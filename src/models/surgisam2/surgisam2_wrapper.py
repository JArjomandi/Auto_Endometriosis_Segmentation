from pathlib import Path
import os
import sys
from contextlib import nullcontext

import numpy as np
import torch


class SurgiSAM2FrozenWrapper:
    """
    SurgiSAM2 wrapper.

    SurgiSAM2 is loaded as a SAM2 image predictor with a surgical fine-tuned checkpoint.
    Supports the same image prompt types as SAM2:
      - point
      - box
      - box + point
      - box + positive/negative points
    """

    def __init__(
        self,
        sam2_repo_root: str,
        checkpoint: str,
        model_cfg: str,
        device: str = "cuda",
        multimask_output: bool = True,
        mask_selection: str = "highest_score",
        use_bfloat16: bool = True,
    ):
        self.sam2_repo_root = Path(sam2_repo_root)
        self.checkpoint = Path(checkpoint)
        self.model_cfg = model_cfg
        self.device = device if torch.cuda.is_available() else "cpu"
        self.multimask_output = bool(multimask_output)
        self.mask_selection = mask_selection
        self.use_bfloat16 = bool(use_bfloat16)

        if not self.sam2_repo_root.exists():
            raise FileNotFoundError(f"SAM2 repo not found: {self.sam2_repo_root}")

        if not self.checkpoint.exists():
            raise FileNotFoundError(f"SurgiSAM2 checkpoint not found: {self.checkpoint}")

        repo_root_str = str(self.sam2_repo_root)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        old_cwd = Path.cwd()

        try:
            os.chdir(self.sam2_repo_root)

            self.model = build_sam2(
                config_file=self.model_cfg,
                ckpt_path=str(self.checkpoint),
                device=self.device,
            )

            self.predictor = SAM2ImagePredictor(self.model)

        finally:
            os.chdir(old_cwd)

    def inference_context(self):
        if self.device == "cuda" and self.use_bfloat16:
            return torch.autocast("cuda", dtype=torch.bfloat16)

        return nullcontext()

    def set_image(self, image_rgb: np.ndarray):
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("image_rgb must be an H x W x 3 RGB image.")

        self.predictor.set_image(image_rgb)

    def predict(
        self,
        box=None,
        point_coords=None,
        point_labels=None,
    ):
        if box is not None:
            box = np.array(box, dtype=np.float32)

        if point_coords is not None:
            point_coords = np.array(point_coords, dtype=np.float32)

        if point_labels is not None:
            point_labels = np.array(point_labels, dtype=np.int32)

        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=self.multimask_output,
        )

        if self.mask_selection != "highest_score":
            raise ValueError(f"Unsupported mask_selection: {self.mask_selection}")

        selected_index = int(np.argmax(scores))
        selected_mask = masks[selected_index]
        selected_score = float(scores[selected_index])

        binary_mask = selected_mask.astype(np.uint8) * 255

        return binary_mask, selected_score, selected_index