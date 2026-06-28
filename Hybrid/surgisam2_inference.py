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
    """
    Loads SurgiSAM2 using your YAML:

        sam2_repo_root: F:/Models/SAM2
        surgisam2_repo_root: F:/Models/SurgiSAM2
        checkpoint: F:/Models/SurgiSAM2/checkpoints/Curated400_checkpoint_image_predictor.pt
        model_cfg: configs/sam2/sam2_hiera_b+.yaml

    This assumes SurgiSAM2 uses the SAM2 image predictor API.
    """

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
            "Check that F:/Models/SAM2 and F:/Models/SurgiSAM2 are correct, "
            "and that you are using the same venv where your SurgiSAM2 benchmark worked."
        ) from error

    model = build_sam2(
        config_file=model_cfg,
        ckpt_path=str(checkpoint_path),
        device=device,
    )

    predictor = SAM2ImagePredictor(model)

    return predictor


@torch.no_grad()
def predict_surgisam2_from_box(
    predictor,
    image_np: np.ndarray,
    box_xyxy,
    multimask_output: bool = True,
    mask_selection: str = "highest_score",
    use_bfloat16: bool = True,
):
    """
    Runs SurgiSAM2 with one automatic box prompt.

    box_xyxy:
        [x_min, y_min, x_max, y_max]
    """

    if box_xyxy is None:
        return np.zeros(image_np.shape[:2], dtype=np.uint8), None, None

    input_box = np.array(box_xyxy, dtype=np.float32)

    use_autocast = (
        use_bfloat16
        and torch.cuda.is_available()
        and str(next(predictor.model.parameters()).device).startswith("cuda")
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
        return np.zeros(image_np.shape[:2], dtype=np.uint8), None, None

    scores_np = np.asarray(scores)

    if mask_selection == "highest_score":
        best_index = int(np.argmax(scores_np))
    else:
        best_index = int(np.argmax(scores_np))

    best_mask = masks[best_index]
    best_score = float(scores_np[best_index])

    best_mask = (best_mask > 0).astype(np.uint8)

    return best_mask, best_score, scores_np.tolist()