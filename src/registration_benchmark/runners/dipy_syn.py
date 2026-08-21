"""Run DIPY SyN registration for one fixed/moving image pair."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric, MIMetric


def load_config(path: str | Path) -> dict:
    with Path(path).open() as f:
        return yaml.safe_load(f)


def as_3d(data: np.ndarray, path: str | Path) -> np.ndarray:
    data = np.squeeze(data)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D image, got {data.shape}: {path}")
    return data


def run_dipy_syn(
    fixed_path: str | Path,
    moving_path: str | Path,
    out_dir: str | Path,
    config: dict,
    moving_labels_path: str | Path | None = None,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fixed_img = nib.load(str(fixed_path))
    moving_img = nib.load(str(moving_path))

    fixed = as_3d(fixed_img.get_fdata(dtype=np.float32), fixed_path)
    moving = as_3d(moving_img.get_fdata(dtype=np.float32), moving_path)

    registration_cfg = config["registration"]
    dipy_cfg = config["dipy"]

    syn_start = time.perf_counter()
    metric_name = registration_cfg["metric"].upper()

    if metric_name == "CC":
        metric = CCMetric(
            3,
            radius=registration_cfg["cc_radius"],
            sigma_diff=dipy_cfg["update_field_sigma"],
        )
    elif metric_name == "MI":
        metric = MIMetric(
            3,
            nbins=registration_cfg.get("mi_nbins", 32),
            smooth=dipy_cfg["update_field_sigma"],
        )
    else:
        raise ValueError(f"Unsupported DIPY metric: {metric_name}")

    sdr = SymmetricDiffeomorphicRegistration(
        metric,
        level_iters=registration_cfg["level_iters"],
        step_length=registration_cfg["grad_step"],
        ss_sigma_factor=dipy_cfg["ss_sigma_factor"],
        opt_tol=registration_cfg["convergence_tol"],
        inv_iter=dipy_cfg["inv_iter"],
        inv_tol=dipy_cfg["inv_tol"],
        num_threads=registration_cfg["num_threads"],
    )
    sdr.energy_window = registration_cfg["convergence_window"]

    mapping = sdr.optimize(
        fixed,
        moving,
        static_grid2world=fixed_img.affine,
        moving_grid2world=moving_img.affine,
        prealign=None,
    )
    warped = mapping.transform(moving)
    syn_runtime_sec = time.perf_counter() - syn_start
    print(f"dipy_syn time: {syn_runtime_sec:.2f} s ({syn_runtime_sec / 60:.2f} min)")

    warped_path = out_dir / "warped_dipy.nii.gz"

    nib.save(nib.Nifti1Image(warped.astype(np.float32), fixed_img.affine), warped_path)

    result = {
        "warped_image": str(warped_path),
        "syn_runtime_sec": syn_runtime_sec,
    }
    if moving_labels_path is not None:
        moving_labels_img = nib.load(str(moving_labels_path))
        moving_labels = np.ascontiguousarray(
            as_3d(np.asarray(moving_labels_img.dataobj), moving_labels_path),
            dtype=np.int16,
        )
        warped_labels = mapping.transform(moving_labels, interpolation="nearest")
        warped_labels_path = out_dir / "warped_dipy_labels.nii.gz"
        nib.save(
            nib.Nifti1Image(warped_labels.astype(np.int16), fixed_img.affine),
            warped_labels_path,
        )
        result["warped_labels"] = str(warped_labels_path)

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DIPY SyN registration.")
    parser.add_argument("--fixed", required=True)
    parser.add_argument("--moving", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dipy_syn(
        fixed_path=args.fixed,
        moving_path=args.moving,
        out_dir=args.out_dir,
        config=load_config(args.config),
    )


if __name__ == "__main__":
    main()
