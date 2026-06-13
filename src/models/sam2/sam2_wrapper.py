import os
import sys
from pathlib import Path

import numpy as np
import torch


class SAM2FrozenWrapper:
    def __init__(
        self,
        sam2_repo_root: str,
        checkpoint: str,
        model_cfg: str,
        device: str = "cuda",
        multimask_output: bool = True,
        use_bfloat16: bool = True,
    ):
        self.sam2_repo_root = Path(sam2_repo_root)
        self.checkpoint = str(checkpoint)
        self.model_cfg = str(model_cfg)
        self.device = device
        self.multimask_output = multimask_output
        self.use_bfloat16 = use_bfloat16

        if not self.sam2_repo_root.exists():
            raise FileNotFoundError(f"SAM2 repo root not found: {self.sam2_repo_root}")

        if not Path(self.checkpoint).exists():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {self.checkpoint}")

        if str(self.sam2_repo_root) not in sys.path:
            sys.path.insert(0, str(self.sam2_repo_root))

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        # SAM2 config resolution is easier if current working directory is SAM2 repo.
        old_cwd = os.getcwd()
        os.chdir(str(self.sam2_repo_root))

        try:
            model = build_sam2(
                config_file=self.model_cfg,
                ckpt_path=self.checkpoint,
                device=self.device,
            )
        finally:
            os.chdir(old_cwd)

        self.predictor = SAM2ImagePredictor(model)

    def set_image(self, image_rgb: np.ndarray):
        self.predictor.set_image(image_rgb)

    def predict(
        self,
        box=None,
        point_coords=None,
        point_labels=None,
    ):
        if point_coords is not None:
            point_coords = np.asarray(point_coords, dtype=np.float32)

        if point_labels is not None:
            point_labels = np.asarray(point_labels, dtype=np.int32)

        if box is not None:
            box = np.asarray(box, dtype=np.float32)

        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=self.multimask_output,
        )

        scores = np.asarray(scores)
        selected_index = int(np.argmax(scores))

        selected_mask = masks[selected_index].astype(np.uint8)
        selected_score = float(scores[selected_index])

        return selected_mask, selected_score, selected_index

    def inference_context(self):
        if self.device == "cuda" and self.use_bfloat16:
            return torch.autocast("cuda", dtype=torch.bfloat16)

        return torch.autocast("cpu", enabled=False)