"""Evaluate registration outputs with framework-independent metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy.stats import pearsonr
from skimage.metrics import normalized_mutual_information

EPS = 1e-8


def load_image(path: str | Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    img = nib.load(str(path))
    data = np.squeeze(img.get_fdata(dtype=np.float32))
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D image, got shape {data.shape}: {path}")
    return img, data


def same_grid(a: nib.Nifti1Image, b: nib.Nifti1Image) -> bool:
    return a.shape == b.shape and np.allclose(a.affine, b.affine, atol=1e-4)


def normalized_values(data: np.ndarray) -> np.ndarray:
    values = data.astype(np.float64).ravel()
    return (values - values.min()) / max(values.max() - values.min(), EPS)


def evaluate_candidate(
    fixed: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    fixed_values = normalized_values(fixed[mask])
    candidate_values = normalized_values(candidate[mask])
    return {
        "ncc": float(pearsonr(fixed_values, candidate_values)[0]),
        "nmi": float(
            normalized_mutual_information(fixed_values, candidate_values, bins=64)
        ),
    }


def dice_jaccard(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    intersection = int(np.count_nonzero(a & b))
    a_count = int(np.count_nonzero(a))
    b_count = int(np.count_nonzero(b))
    union = int(np.count_nonzero(a | b))
    return {
        "dice": float(2 * intersection / max(a_count + b_count, 1)),
        "jaccard": float(intersection / max(union, 1)),
    }


def evaluate_label_overlap(
    fixed_labels: np.ndarray, candidate_labels: np.ndarray
) -> dict:
    whole_brain = dice_jaccard(fixed_labels > 0, candidate_labels > 0)

    label_scores = [
        dice_jaccard(fixed_labels == label_id, candidate_labels == label_id)
        for label_id in np.union1d(fixed_labels, candidate_labels)
        if label_id != 0
    ]

    mean_labels = {
        metric: float(np.mean([score[metric] for score in label_scores]))
        for metric in ("dice", "jaccard")
    }

    return {"whole_brain": whole_brain, "mean_labels": mean_labels}


def evaluate_registration(
    pair_id: str,
    fixed_path: str | Path,
    moving_path: str | Path,
    fixed_mask_path: str | Path,
    warped_ants_path: str | Path,
    warped_dipy_path: str | Path,
    out_json: str | Path,
    fixed_labels_path: str | Path | None = None,
    moving_labels_path: str | Path | None = None,
    warped_ants_labels_path: str | Path | None = None,
    warped_dipy_labels_path: str | Path | None = None,
) -> dict:
    fixed_img, fixed = load_image(fixed_path)
    moving_img, moving = load_image(moving_path)
    mask_img, fixed_mask = load_image(fixed_mask_path)
    ants_img, ants_warped = load_image(warped_ants_path)
    dipy_img, dipy_warped = load_image(warped_dipy_path)

    if not same_grid(fixed_img, mask_img):
        raise ValueError("Fixed mask is not on the fixed image grid.")
    if not same_grid(fixed_img, ants_img):
        raise ValueError("ANTs warped image is not on the fixed image grid.")
    if not same_grid(fixed_img, dipy_img):
        raise ValueError("DIPY warped image is not on the fixed image grid.")

    if not same_grid(fixed_img, moving_img):
        moving = np.squeeze(
            resample_from_to(moving_img, fixed_img, order=1).get_fdata(dtype=np.float32)
        )

    candidates = {
        "baseline": moving,
        "ants": ants_warped,
        "dipy": dipy_warped,
    }
    mask = fixed_mask > 0
    mask &= np.isfinite(fixed)

    if not mask.any():
        raise ValueError("Evaluation mask is empty.")

    output = {
        "pair_id": pair_id,
        "metrics": {
            name: evaluate_candidate(fixed, candidate, mask)
            for name, candidate in candidates.items()
        },
    }

    if all(
        path is not None
        for path in (
            fixed_labels_path,
            moving_labels_path,
            warped_ants_labels_path,
            warped_dipy_labels_path,
        )
    ):
        fixed_labels_img, fixed_labels = load_image(fixed_labels_path)
        moving_labels_img, moving_labels = load_image(moving_labels_path)
        ants_labels_img, ants_labels = load_image(warped_ants_labels_path)
        dipy_labels_img, dipy_labels = load_image(warped_dipy_labels_path)
        for name, labels_img in {
            "Fixed labels": fixed_labels_img,
            "Moving labels": moving_labels_img,
            "ANTs warped labels": ants_labels_img,
            "DIPY warped labels": dipy_labels_img,
        }.items():
            if not same_grid(fixed_img, labels_img):
                raise ValueError(f"{name} are not on the fixed image grid.")

        label_candidates = {
            "baseline": moving_labels.astype(np.int32),
            "ants": ants_labels.astype(np.int32),
            "dipy": dipy_labels.astype(np.int32),
        }
        output["overlap_metrics"] = {
            name: evaluate_label_overlap(fixed_labels.astype(np.int32), labels)
            for name, labels in label_candidates.items()
        }

    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(output, f, indent=2)

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate registration outputs.")
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--fixed", required=True)
    parser.add_argument("--moving", required=True)
    parser.add_argument("--fixed-mask", required=True)
    parser.add_argument("--warped-ants", required=True)
    parser.add_argument("--warped-dipy", required=True)
    parser.add_argument("--fixed-labels")
    parser.add_argument("--moving-labels")
    parser.add_argument("--warped-ants-labels")
    parser.add_argument("--warped-dipy-labels")
    parser.add_argument("--out-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_registration(
        pair_id=args.pair_id,
        fixed_path=args.fixed,
        moving_path=args.moving,
        fixed_mask_path=args.fixed_mask,
        warped_ants_path=args.warped_ants,
        warped_dipy_path=args.warped_dipy,
        fixed_labels_path=args.fixed_labels,
        moving_labels_path=args.moving_labels,
        warped_ants_labels_path=args.warped_ants_labels,
        warped_dipy_labels_path=args.warped_dipy_labels,
        out_json=args.out_json,
    )


if __name__ == "__main__":
    main()
