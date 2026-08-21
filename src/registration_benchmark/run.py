"""Run the registration benchmark pipeline for a CSV of fixed/moving pairs.

Pipeline:

1. Reslice both images by the same downsampling factor.
2. Skull-strip both images and save SynthSeg labels.
3. Fill holes and keep the largest connected mask component.
4. Rigidly prealign the moving image to the fixed image.
5. Run DIPY SyN and ANTs SyN from the same prealigned inputs.
6. Evaluate both warped outputs with framework-independent metrics.

Example, after installing this repository:

    registration-benchmark \
        --pairs data/oasis2_pairs.csv \
        --config configs/syn_cc_default.yaml \
        --out-dir outputs/oasis2_syn_cc_ds2 \
        --downsample-factor 2 \
        --n 5

Example, run only one pair index:

    registration-benchmark \
        --pairs data/oasis2_pairs.csv \
        --config configs/syn_cc_default.yaml \
        --out-dir outputs/oasis2_syn_cc_ds2 \
        --pair-index 0
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml
from dipy.align.imaffine import (
    AffineRegistration,
    MutualInformationMetric,
    transform_centers_of_mass,
)
from dipy.align.reslice import reslice
from dipy.align.transforms import RigidTransform3D, TranslationTransform3D
from scipy.ndimage import binary_fill_holes, label

from registration_benchmark import __version__
from registration_benchmark.evaluate import evaluate_registration
from registration_benchmark.runners.ants_syn import run_ants_syn
from registration_benchmark.runners.dipy_syn import run_dipy_syn

_SYNTHSEG_MODEL = None
METRIC_NAMES = ("ncc", "nmi")
OVERLAP_METRIC_NAMES = ("dice", "jaccard")
VERSIONED_PACKAGES = (
    "antspyx",
    "dipy",
    "nibabel",
    "numpy",
    "PyYAML",
    "scikit-image",
    "scipy",
    "torch",
)


def runtime_environment() -> dict:
    """Return software and platform metadata needed to reproduce a run."""
    packages = {}
    for package in VERSIONED_PACKAGES:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None

    return {
        "benchmark_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
    }


def load_yaml(path: str | Path) -> dict:
    with Path(path).open() as f:
        return yaml.safe_load(f)


def read_pairs(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f))

    required = {"fixed_path", "moving_path"}
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise ValueError(f"Pair CSV is missing columns: {sorted(missing)}")
    return rows


def select_pairs(rows: list[dict[str, str]], n: int | None) -> list[dict[str, str]]:
    if n is None:
        return rows
    if n < 0:
        raise ValueError(f"Requested n={n}, but n must be non-negative.")
    if n > len(rows):
        raise ValueError(f"Requested n={n}, but only found {len(rows)} pairs.")
    return rows[:n]


def select_pair_indices(
    rows: list[dict[str, str]], pair_index: int | None
) -> list[tuple[int, dict[str, str]]]:
    if pair_index is None:
        return list(enumerate(rows, start=1))
    if pair_index < 0 or pair_index >= len(rows):
        raise ValueError(
            f"Requested pair-index={pair_index}, but found {len(rows)} pairs."
        )
    return [(pair_index + 1, rows[pair_index])]


def get_pair_id(row: dict[str, str], index: int) -> str:
    pair_id = row.get("pair_id", "").strip()
    return pair_id or f"pair_{index:04d}"


def _check_output_run_metadata(path: Path, requested: dict) -> None:
    with path.open() as f:
        existing = json.load(f)

    if existing == requested:
        return

    raise ValueError(
        "The requested preprocessing is incompatible with cached data in "
        f"{path.parent}. Use a new --out-dir (or deliberately remove the old "
        "outputs) before running with a different pair CSV, pair limit, or "
        "preprocessing settings.\nExisting preprocessing contract:\n"
        f"{json.dumps(existing, indent=2)}\n"
        "Requested preprocessing contract:\n"
        f"{json.dumps(requested, indent=2)}"
    )


def ensure_output_run_metadata(
    out_dir: Path,
    requested: dict,
) -> Path:
    metadata_path = out_dir / "run_metadata.json"
    if metadata_path.exists():
        _check_output_run_metadata(metadata_path, requested)
        return metadata_path

    with metadata_path.open("w") as f:
        json.dump(requested, f, indent=2)
        f.write("\n")
    return metadata_path


def warn_about_existing_pair_outputs(pair_id: str, pair_out: Path) -> None:
    if (pair_out / "prepared").exists() or (pair_out / "prealign").exists():
        warnings.warn(
            f"Compatible prepared images already exist for {pair_id}; available "
            "preprocessing outputs will be reused.",
            RuntimeWarning,
            stacklevel=2,
        )

    registration_outputs = (
        pair_out / "dipy",
        pair_out / "ants",
        pair_out / "evaluation.json",
        pair_out / "sample_result.json",
    )
    if any(path.exists() for path in registration_outputs):
        warnings.warn(
            f"Registration or evaluation outputs already exist for {pair_id}; "
            "they will be overwritten by this run.",
            RuntimeWarning,
            stacklevel=2,
        )


def reslice_by_factor(
    in_path: str | Path,
    out_path: str | Path,
    factor: float,
    *,
    order: int = 1,
) -> Path:
    in_path = Path(in_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if factor == 1:
        return in_path
    if out_path.exists():
        print(f"Reusing resliced image: {out_path}")
        return out_path

    print(f"Reslicing image: {in_path}")
    img = nib.load(str(in_path))
    data = np.squeeze(np.asarray(img.dataobj, dtype=np.float32))
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D image, got {data.shape}: {in_path}")

    zooms = img.header.get_zooms()[:3]
    new_zooms = tuple(float(zoom) * factor for zoom in zooms)
    data_rs, affine_rs = reslice(data, img.affine, zooms, new_zooms, order=order)

    out_img = nib.Nifti1Image(data_rs.astype(np.float32), affine_rs)
    nib.save(out_img, str(out_path))
    return out_path


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    components, n_components = label(mask)
    if n_components <= 1:
        return mask.astype(bool)

    counts = np.bincount(components.ravel())
    counts[0] = 0
    return components == int(counts.argmax())


def get_synthseg_model(use_cuda: bool):
    global _SYNTHSEG_MODEL
    if _SYNTHSEG_MODEL is None:
        from dipy.nn.torch.synthseg import SynthSeg

        _SYNTHSEG_MODEL = SynthSeg(verbose=False, use_cuda=use_cuda)
    return _SYNTHSEG_MODEL


def skull_strip(
    in_path: str | Path,
    out_img_path: str | Path,
    out_mask_path: str | Path,
    out_labels_path: str | Path,
    *,
    use_cuda: bool,
    already_skull_stripped: bool,
    labels_path: str | Path | None = None,
) -> Path:
    in_path = Path(in_path)
    out_img_path = Path(out_img_path)
    out_mask_path = Path(out_mask_path)
    out_labels_path = Path(out_labels_path)
    if (
        (already_skull_stripped or out_img_path.exists())
        and out_mask_path.exists()
        and (labels_path is not None or out_labels_path.exists())
    ):
        return in_path if already_skull_stripped else out_img_path

    print(f"Preparing image: {in_path}")
    out_img_path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.load(str(in_path))
    data = np.squeeze(img.get_fdata(dtype=np.float32))
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D image, got {data.shape}: {in_path}")

    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    labels = mask = None
    if not already_skull_stripped or (
        labels_path is None and not out_labels_path.exists()
    ):
        print(f"Running SynthSeg: {in_path}")
        labels, _, mask = get_synthseg_model(use_cuda).predict(data, img.affine)

    if already_skull_stripped:
        mask = data != 0
        brain_path = in_path
    else:
        brain_path = out_img_path
        mask = mask.astype(bool)

    mask = keep_largest_component(binary_fill_holes(mask))
    if not already_skull_stripped:
        brain = data * mask
        nib.save(nib.Nifti1Image(brain.astype(np.float32), img.affine), out_img_path)

    nib.save(nib.Nifti1Image(mask.astype(np.uint8), img.affine), out_mask_path)

    if labels_path is None and not out_labels_path.exists():
        labels = labels.astype(np.int16) * mask.astype(np.int16)
        nib.save(
            nib.Nifti1Image(labels, img.affine),
            out_labels_path,
        )

    return brain_path


def rigid_prealign(
    fixed_path: str | Path,
    moving_path: str | Path,
    out_path: str | Path,
    moving_labels_path: str | Path | None = None,
    out_labels_path: str | Path | None = None,
) -> Path:
    out_path = Path(out_path)
    out_labels_path = Path(out_labels_path) if out_labels_path is not None else None
    if out_path.exists() and (out_labels_path is None or out_labels_path.exists()):
        print(f"Reusing rigid prealignment: {out_path}")
        return out_path

    print("Running rigid prealignment")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fixed_img = nib.load(str(fixed_path))
    moving_img = nib.load(str(moving_path))
    fixed = fixed_img.get_fdata(dtype=np.float32)
    moving = moving_img.get_fdata(dtype=np.float32)

    center = transform_centers_of_mass(
        fixed, fixed_img.affine, moving, moving_img.affine
    )

    metric = MutualInformationMetric(nbins=32, sampling_proportion=None)
    affreg = AffineRegistration(
        metric=metric,
        level_iters=[10000, 1000, 100],
        sigmas=[3.0, 1.0, 0.0],
        factors=[4, 2, 1],
    )

    translation = affreg.optimize(
        fixed,
        moving,
        TranslationTransform3D(),
        params0=None,
        static_grid2world=fixed_img.affine,
        moving_grid2world=moving_img.affine,
        starting_affine=center.affine,
    )

    rigid = affreg.optimize(
        fixed,
        moving,
        RigidTransform3D(),
        params0=None,
        static_grid2world=fixed_img.affine,
        moving_grid2world=moving_img.affine,
        starting_affine=translation.affine,
    )

    prealigned = rigid.transform(moving)

    nib.save(
        nib.Nifti1Image(
            prealigned.astype(np.float32), fixed_img.affine, fixed_img.header
        ),
        str(out_path),
    )

    if moving_labels_path is not None and out_labels_path is not None:
        moving_labels_img = nib.load(str(moving_labels_path))
        moving_labels = np.ascontiguousarray(
            np.squeeze(np.asarray(moving_labels_img.dataobj)).astype(np.float32)
        )
        prealigned_labels = rigid.transform(moving_labels, interpolation="nearest")
        label_header = fixed_img.header.copy()
        label_header.set_data_dtype(np.int16)
        nib.save(
            nib.Nifti1Image(
                prealigned_labels.astype(np.int16),
                fixed_img.affine,
                label_header,
            ),
            str(out_labels_path),
        )

    return out_path


def prepare_pair(
    row: dict[str, str],
    pair_out: Path,
    *,
    downsample_factor: float,
    use_cuda: bool,
    already_skull_stripped: bool,
) -> tuple[Path, Path, Path, Path]:
    prepared_out = pair_out / "prepared"
    inputs_out = prepared_out / "inputs"
    skullstrip_out = prepared_out / "skullstrip"

    print("Preparing inputs")
    fixed_resliced = reslice_by_factor(
        row["fixed_path"],
        inputs_out / "fixed_resliced.nii.gz",
        downsample_factor,
    )
    moving_resliced = reslice_by_factor(
        row["moving_path"],
        inputs_out / "moving_resliced.nii.gz",
        downsample_factor,
    )

    fixed_labels = (row.get("fixed_label_path") or "").strip() or None
    moving_labels = (row.get("moving_label_path") or "").strip() or None
    if fixed_labels is not None:
        fixed_labels = reslice_by_factor(
            fixed_labels,
            inputs_out / "fixed_labels_resliced.nii.gz",
            downsample_factor,
            order=0,
        )
    if moving_labels is not None:
        moving_labels = reslice_by_factor(
            moving_labels,
            inputs_out / "moving_labels_resliced.nii.gz",
            downsample_factor,
            order=0,
        )

    fixed_brain = skull_strip(
        fixed_resliced,
        skullstrip_out / "fixed_brain.nii.gz",
        skullstrip_out / "fixed_mask.nii.gz",
        skullstrip_out / "fixed_labels.nii.gz",
        use_cuda=use_cuda,
        already_skull_stripped=already_skull_stripped,
        labels_path=fixed_labels,
    )
    moving_brain = skull_strip(
        moving_resliced,
        skullstrip_out / "moving_brain.nii.gz",
        skullstrip_out / "moving_mask.nii.gz",
        skullstrip_out / "moving_labels.nii.gz",
        use_cuda=use_cuda,
        already_skull_stripped=already_skull_stripped,
        labels_path=moving_labels,
    )
    fixed_labels = fixed_labels or skullstrip_out / "fixed_labels.nii.gz"
    moving_labels = moving_labels or skullstrip_out / "moving_labels.nii.gz"

    moving_prealigned = rigid_prealign(
        fixed_brain,
        moving_brain,
        pair_out / "prealign" / "moving_rigid_to_fixed.nii.gz",
        moving_labels,
        pair_out / "prealign" / "moving_labels_rigid_to_fixed.nii.gz",
    )
    return (
        fixed_brain,
        moving_prealigned,
        fixed_labels,
        pair_out / "prealign" / "moving_labels_rigid_to_fixed.nii.gz",
    )


def gains_vs_baseline(metrics: dict) -> dict:
    baseline = metrics["baseline"]
    return {
        method: {metric: values[metric] - baseline[metric] for metric in METRIC_NAMES}
        for method, values in metrics.items()
        if method != "baseline"
    }


def summarize(samples: list[dict]) -> dict:
    methods = sorted({method for sample in samples for method in sample["metrics"]})
    summary = {}

    for method in methods:
        method_summary = {"n": len(samples)}
        for metric in METRIC_NAMES:
            values = [
                sample["metrics"][method][metric]
                for sample in samples
                if method in sample["metrics"]
            ]
            method_summary[metric] = {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            }

            if method == "baseline":
                continue

            gain_values = [
                sample["gains_vs_baseline"][method][metric]
                for sample in samples
                if method in sample["gains_vs_baseline"]
            ]
            method_summary[f"{metric}_gain_vs_baseline"] = {
                "mean": statistics.mean(gain_values),
                "std": statistics.stdev(gain_values) if len(gain_values) > 1 else 0.0,
            }
        summary[method] = method_summary

    return summary


def summarize_overlap(samples: list[dict]) -> dict:
    summary = {}
    methods = sorted(
        method for sample in samples for method in sample.get("overlap_metrics", {})
    )

    for method in methods:
        method_samples = [
            sample["overlap_metrics"][method]
            for sample in samples
            if method in sample.get("overlap_metrics", {})
        ]

        summary[method] = {"n": len(method_samples)}

        for section in ("whole_brain", "mean_labels"):
            summary[method][section] = {}
            for metric in ("dice", "jaccard"):
                values = [sample[section][metric] for sample in method_samples]
                summary[method][section][metric] = {
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                }

    return summary


def summarize_timings(samples: list[dict]) -> dict:
    keys = sorted(key for sample in samples for key in sample.get("timings", {}))
    summary = {}

    for key in keys:
        values = [
            sample["timings"][key]
            for sample in samples
            if key in sample.get("timings", {})
        ]
        summary[key] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }

    return summary


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def sample_result(
    pair_id: str,
    row: dict[str, str],
    evaluation: dict,
    run_metadata: dict,
) -> dict:
    return {
        "pair_id": pair_id,
        "fixed_path": row["fixed_path"],
        "moving_path": row["moving_path"],
        "metrics": evaluation["metrics"],
        "overlap_metrics": evaluation.get("overlap_metrics", {}),
        "gains_vs_baseline": gains_vs_baseline(evaluation["metrics"]),
        "timings": evaluation.get("timings", {}),
        "run_metadata": run_metadata,
    }


def run_pair(
    pair_id: str,
    row: dict[str, str],
    out_dir: Path,
    config: dict,
    args: argparse.Namespace,
) -> dict:
    pair_out = out_dir / pair_id
    warn_about_existing_pair_outputs(pair_id, pair_out)

    fixed, moving, fixed_labels, moving_labels = prepare_pair(
        row,
        pair_out,
        downsample_factor=args.downsample_factor,
        use_cuda=args.use_cuda,
        already_skull_stripped=args.already_skull_stripped,
    )

    print("Running DIPY SyN")
    dipy_result = run_dipy_syn(
        fixed,
        moving,
        pair_out / "dipy",
        config,
        moving_labels_path=moving_labels,
    )

    print("Running ANTs SyN")
    ants_result = run_ants_syn(
        fixed,
        moving,
        pair_out / "ants",
        config,
        moving_labels_path=moving_labels,
    )

    print("Evaluating registration outputs")
    evaluation = evaluate_registration(
        pair_id=pair_id,
        fixed_path=fixed,
        moving_path=moving,
        fixed_mask_path=pair_out / "prepared" / "skullstrip" / "fixed_mask.nii.gz",
        warped_ants_path=ants_result["warped_image"],
        warped_dipy_path=dipy_result["warped_image"],
        out_json=pair_out / "evaluation.json",
        fixed_labels_path=fixed_labels,
        moving_labels_path=moving_labels,
        warped_ants_labels_path=ants_result["warped_labels"],
        warped_dipy_labels_path=dipy_result["warped_labels"],
    )

    evaluation["timings"] = {
        "dipy_syn_sec": dipy_result["syn_runtime_sec"],
        "ants_syn_sec": ants_result["syn_runtime_sec"],
    }
    return evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DIPY vs ANTs registration benchmark."
    )
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--out-dir", default=Path("outputs/registration_benchmark"), type=Path
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Run only the first N pairs from the CSV, in file order.",
    )
    parser.add_argument("--downsample-factor", type=float, default=1.0)
    parser.add_argument(
        "--already-skull-stripped",
        action="store_true",
        help=(
            "Treat input images as already skull stripped and derive the "
            "evaluation mask from nonzero voxels."
        ),
    )
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument(
        "--initialize-only",
        action="store_true",
        help="Create or validate run_metadata.json without processing pairs.",
    )
    parser.add_argument(
        "--pair-index",
        type=int,
        default=None,
        help="Run only one zero-based pair index. Useful for cluster jobs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    pairs = select_pairs(read_pairs(args.pairs), args.n)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    preprocessing_contract = {
        "pairs_file": str(args.pairs),
        "n": args.n,
        "downsample_factor": args.downsample_factor,
        "already_skull_stripped": args.already_skull_stripped,
    }
    metadata_path = ensure_output_run_metadata(args.out_dir, preprocessing_contract)
    if args.initialize_only:
        print(f"Run metadata ready: {metadata_path}")
        return

    indexed_pairs = select_pair_indices(pairs, args.pair_index)

    run_metadata = {
        "pairs_file": str(args.pairs),
        "n": args.n,
        "config_file": str(args.config),
        "config": config,
        "downsample_factor": args.downsample_factor,
        "already_skull_stripped": args.already_skull_stripped,
        "use_cuda": args.use_cuda,
        "runtime_environment": runtime_environment(),
    }
    results = {
        "metadata": {
            **run_metadata,
            "n_pairs": len(indexed_pairs),
            "pair_index": args.pair_index,
        },
        "samples": [],
        "summary": {},
        "overlap_summary": {},
        "timing_summary": {},
    }

    for index, row in indexed_pairs:
        pair_id = get_pair_id(row, index)
        print(f"\n=== {pair_id} ===")
        if not row.get("pair_id", "").strip():
            print(f"fixed:  {row['fixed_path']}")
            print(f"moving: {row['moving_path']}")

        evaluation = run_pair(pair_id, row, args.out_dir, config, args)
        sample = sample_result(pair_id, row, evaluation, run_metadata)
        write_json(args.out_dir / pair_id / "sample_result.json", sample)
        results["samples"].append(sample)
        results["summary"] = summarize(results["samples"])
        results["overlap_summary"] = summarize_overlap(results["samples"])
        results["timing_summary"] = summarize_timings(results["samples"])
        if args.pair_index is None:
            write_json(args.out_dir / "benchmark_results.json", results)

    if args.pair_index is not None:
        print(f"\nDone. Pair result saved in: {args.out_dir / pair_id}")
    else:
        print(f"\nDone. Results saved in: {args.out_dir}")


if __name__ == "__main__":
    main()
