"""Run ANTsPy SyN registration for one fixed/moving image pair."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with Path(path).open() as f:
        return yaml.safe_load(f)


def remove_transform_files(registration_result: dict) -> None:
    transform_paths = [
        *registration_result.get("fwdtransforms", []),
        *registration_result.get("invtransforms", []),
    ]
    for transform_path in set(transform_paths):
        path = Path(transform_path)
        if path.exists():
            path.unlink()


def run_ants_syn(
    fixed_path: str | Path,
    moving_path: str | Path,
    out_dir: str | Path,
    config: dict,
    moving_labels_path: str | Path | None = None,
) -> dict:
    registration_cfg = config["registration"]
    num_threads = registration_cfg["num_threads"]
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(num_threads)
    os.environ["ANTS_RANDOM_SEED"] = str(registration_cfg["random_seed"])

    import ants

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fixed = ants.image_read(str(fixed_path))
    moving = ants.image_read(str(moving_path))

    ants_cfg = config["ants"]

    metric_name = registration_cfg["metric"].upper()

    if metric_name == "CC":
        syn_metric = "CC"
        syn_sampling = registration_cfg["cc_radius"]
    elif metric_name == "MI":
        syn_metric = "mattes"
        syn_sampling = registration_cfg.get("mi_nbins", 32)
    else:
        raise ValueError(f"Unsupported ANTs metric: {metric_name}")

    syn_start = time.perf_counter()
    reg = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform="SyNOnly",
        initial_transform=registration_cfg["initial_transform"],
        syn_metric=syn_metric,
        syn_sampling=syn_sampling,
        reg_iterations=registration_cfg["level_iters"],
        grad_step=registration_cfg["grad_step"],
        flow_sigma=ants_cfg["flow_sigma"],
        total_sigma=ants_cfg["total_sigma"],
        singleprecision=ants_cfg["singleprecision"],
        use_legacy_histogram_matching=False,
        verbose=True,
    )
    syn_runtime_sec = time.perf_counter() - syn_start
    print(f"ants_syn time: {syn_runtime_sec:.2f} s ({syn_runtime_sec / 60:.2f} min)")

    warped_path = out_dir / "warped_ants.nii.gz"
    ants.image_write(reg["warpedmovout"], str(warped_path))
    result = {
        "warped_image": str(warped_path),
        "syn_runtime_sec": syn_runtime_sec,
    }

    if moving_labels_path is not None:
        moving_labels = ants.image_read(str(moving_labels_path))
        warped_labels = ants.apply_transforms(
            fixed=fixed,
            moving=moving_labels,
            transformlist=reg["fwdtransforms"],
            interpolator="genericLabel",
        )
        warped_labels_path = out_dir / "warped_ants_labels.nii.gz"
        ants.image_write(warped_labels, str(warped_labels_path))
        result["warped_labels"] = str(warped_labels_path)

    remove_transform_files(reg)

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ANTsPy SyN registration.")
    parser.add_argument("--fixed", required=True)
    parser.add_argument("--moving", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_ants_syn(
        fixed_path=args.fixed,
        moving_path=args.moving,
        out_dir=args.out_dir,
        config=load_config(args.config),
    )


if __name__ == "__main__":
    main()
