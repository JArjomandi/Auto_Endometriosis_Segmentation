from pathlib import Path
import sys

import numpy as np
import torch


def add_repo_to_path(repo_path: Path):
    repo_path = Path(repo_path)

    if not repo_path.exists():
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")

    repo_path_str = str(repo_path)

    if repo_path_str not in sys.path:
        sys.path.insert(0, repo_path_str)


def build_surgisam2_predictor(
    sam2_repo_root: Path,
    surgisam2_repo_root: Path,
    model_cfg: str,
    checkpoint_path: Path,
    device: torch.device,
):
    sam2_repo_root = Path(sam2_repo_root)
    surgisam2_repo_root = Path(surgisam2_repo_root)
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing SurgiSAM2 checkpoint: {checkpoint_path}")

    add_repo_to_path(sam2_repo_root)
    add_repo_to_path(surgisam2_repo_root)

    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except Exception as error:
        raise ImportError(
            "Could not import SAM2/SurgiSAM2 modules. "
            "Check F:/Models/SAM2, F:/Models/SurgiSAM2, and the active venv."
        ) from error

    model = build_sam2(
        config_file=model_cfg,
        ckpt_path=str(checkpoint_path),
        device=device,
    )

    predictor = SAM2ImagePredictor(model)

    return predictor


def compute_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray):
    mask_a = mask_a > 0
    mask_b = mask_b > 0

    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()

    if union == 0:
        return 1.0

    return float(intersection / union)


def compute_mask_dice(mask_a: np.ndarray, mask_b: np.ndarray):
    mask_a = mask_a > 0
    mask_b = mask_b > 0

    intersection = np.logical_and(mask_a, mask_b).sum()
    denominator = mask_a.sum() + mask_b.sum()

    if denominator == 0:
        return 1.0

    return float((2.0 * intersection) / denominator)


def dilate_binary_mask(mask_np: np.ndarray, dilation_radius_px: int = 5):
    mask_np = (mask_np > 0).astype(np.uint8)

    if dilation_radius_px <= 0:
        return mask_np

    try:
        from scipy import ndimage
    except Exception:
        return mask_np

    structure_size = 2 * dilation_radius_px + 1
    structure = np.ones((structure_size, structure_size), dtype=bool)

    dilated = ndimage.binary_dilation(
        mask_np > 0,
        structure=structure,
    )

    return dilated.astype(np.uint8)


@torch.no_grad()
def predict_surgisam2_candidates_from_box(
    predictor,
    image_np: np.ndarray,
    box_xyxy,
    multimask_output: bool = True,
    use_bfloat16: bool = True,
):
    """
    Returns all SurgiSAM2 candidate masks for one box prompt.
    """

    if box_xyxy is None:
        return [], []

    input_box = np.array(box_xyxy, dtype=np.float32)

    try:
        model_device = str(next(predictor.model.parameters()).device)
    except Exception:
        model_device = "cuda" if torch.cuda.is_available() else "cpu"

    use_autocast = (
        use_bfloat16
        and torch.cuda.is_available()
        and model_device.startswith("cuda")
    )

    if use_autocast:
        autocast_context = torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        )
    else:
        autocast_context = torch.autocast(
            device_type="cpu",
            enabled=False,
        )

    with autocast_context:
        predictor.set_image(image_np)

        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box,
            multimask_output=multimask_output,
        )

    if masks is None or len(masks) == 0:
        return [], []

    candidate_masks = [
        (mask > 0).astype(np.uint8)
        for mask in masks
    ]

    candidate_scores = [
        float(score)
        for score in np.asarray(scores).tolist()
    ]

    return candidate_masks, candidate_scores


def select_candidate_by_segformer_agreement(
    candidate_masks,
    candidate_scores,
    segformer_component_mask: np.ndarray,
    acceptance_iou_threshold: float = 0.50,
    acceptance_dice_threshold: float = 0.85,
    min_area_ratio: float = 0.70,
    max_area_ratio: float = 1.30,
    clip_to_dilated_segformer: bool = True,
    dilation_radius_px: int = 5,
):
    """
    Safer fallback rule.

    For each exact SegFormer connected component:
        1. Evaluate all SurgiSAM2 candidate masks.
        2. Select the candidate with highest Dice agreement to the SegFormer component.
        3. Accept it only if:
            - Dice agreement >= acceptance_dice_threshold
            - area ratio is within [min_area_ratio, max_area_ratio]
        4. If accepted, optionally clip the SurgiSAM2 candidate to a dilated
           SegFormer component neighborhood.
        5. If rejected, return the original SegFormer component.

    This prevents SurgiSAM2 from replacing a good SegFormer component with a
    mask that is only weakly related to it.
    """

    segformer_component_mask = (segformer_component_mask > 0).astype(np.uint8)
    segformer_area = int(segformer_component_mask.sum())

    if len(candidate_masks) == 0:
        return {
            "accepted": False,
            "selected_mask": segformer_component_mask,
            "selected_candidate_index": None,
            "selected_surgisam2_score": None,
            "best_agreement_iou": None,
            "best_agreement_dice": None,
            "best_area_ratio": None,
            "candidate_agreements": [],
            "decision": "fallback_no_surgisam2_candidates",
        }

    candidate_agreements = []

    for candidate_index, candidate_mask in enumerate(candidate_masks):
        candidate_mask = (candidate_mask > 0).astype(np.uint8)

        candidate_area = int(candidate_mask.sum())

        agreement_iou = compute_mask_iou(
            candidate_mask,
            segformer_component_mask,
        )

        agreement_dice = compute_mask_dice(
            candidate_mask,
            segformer_component_mask,
        )

        if segformer_area > 0:
            area_ratio = candidate_area / segformer_area
        else:
            area_ratio = np.inf if candidate_area > 0 else 1.0

        surgisam2_score = (
            candidate_scores[candidate_index]
            if candidate_index < len(candidate_scores)
            else None
        )

        candidate_agreements.append(
            {
                "candidate_index": int(candidate_index),
                "surgisam2_score": surgisam2_score,
                "agreement_iou_with_segformer_component": agreement_iou,
                "agreement_dice_with_segformer_component": agreement_dice,
                "candidate_area_px": candidate_area,
                "segformer_component_area_px": segformer_area,
                "area_ratio_candidate_over_segformer": float(area_ratio),
            }
        )

    best_candidate = max(
        candidate_agreements,
        key=lambda item: item["agreement_dice_with_segformer_component"],
    )

    best_index = best_candidate["candidate_index"]
    best_iou = best_candidate["agreement_iou_with_segformer_component"]
    best_dice = best_candidate["agreement_dice_with_segformer_component"]
    best_score = best_candidate["surgisam2_score"]
    best_area_ratio = best_candidate["area_ratio_candidate_over_segformer"]

    passes_iou = best_iou >= acceptance_iou_threshold
    passes_dice = best_dice >= acceptance_dice_threshold
    passes_area = min_area_ratio <= best_area_ratio <= max_area_ratio

    if passes_iou and passes_dice and passes_area:
        selected_mask = (candidate_masks[best_index] > 0).astype(np.uint8)

        if clip_to_dilated_segformer:
            dilated_component = dilate_binary_mask(
                segformer_component_mask,
                dilation_radius_px=dilation_radius_px,
            )

            selected_mask = np.logical_and(
                selected_mask > 0,
                dilated_component > 0,
            ).astype(np.uint8)

        return {
            "accepted": True,
            "selected_mask": selected_mask,
            "selected_candidate_index": int(best_index),
            "selected_surgisam2_score": best_score,
            "best_agreement_iou": best_iou,
            "best_agreement_dice": best_dice,
            "best_area_ratio": best_area_ratio,
            "candidate_agreements": candidate_agreements,
            "decision": "accepted_surgisam2_candidate_strict_agreement",
        }

    failed_reasons = []

    if not passes_iou:
        failed_reasons.append("low_iou_agreement")

    if not passes_dice:
        failed_reasons.append("low_dice_agreement")

    if not passes_area:
        failed_reasons.append("area_ratio_out_of_range")

    return {
        "accepted": False,
        "selected_mask": segformer_component_mask,
        "selected_candidate_index": int(best_index),
        "selected_surgisam2_score": best_score,
        "best_agreement_iou": best_iou,
        "best_agreement_dice": best_dice,
        "best_area_ratio": best_area_ratio,
        "candidate_agreements": candidate_agreements,
        "decision": "fallback_to_segformer_component_" + "_".join(failed_reasons),
    }