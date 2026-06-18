from pathlib import Path
import sys
import importlib

import cv2
import numpy as np
import torch
import torch.nn.functional as torch_functional


class MedSAMFrozenWrapper:
    """
    Frozen MedSAM wrapper for box-prompt inference.

    Prompt:
        box_xyxy = [x1, y1, x2, y2]
        Coordinates are in original image pixel space.

    Output:
        binary mask in original image size, uint8 values {0, 255}.
    """

    def __init__(
        self,
        medsam_repo_root: str,
        checkpoint: str,
        device: str = "cuda:0",
        image_size: int = 1024,
    ):
        self.medsam_repo_root = Path(medsam_repo_root)
        self.checkpoint = Path(checkpoint)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.image_size = int(image_size)

        if not self.medsam_repo_root.exists():
            raise FileNotFoundError(f"MedSAM repo not found: {self.medsam_repo_root}")

        if not self.checkpoint.exists():
            raise FileNotFoundError(f"MedSAM checkpoint not found: {self.checkpoint}")

        repo_root_str = str(self.medsam_repo_root)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)

        segment_anything_module = importlib.import_module("segment_anything")
        sam_model_registry = getattr(segment_anything_module, "sam_model_registry")

        self.model = sam_model_registry["vit_b"](checkpoint=str(self.checkpoint))
        self.model.to(self.device)
        self.model.eval()

        self.image_embedding = None
        self.original_size = None

    def set_image(self, image_rgb: np.ndarray):
        """
        Precompute MedSAM image embedding.

        Args:
            image_rgb:
                RGB image as H x W x 3 uint8 numpy array.
        """

        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("image_rgb must be an H x W x 3 RGB image.")

        original_h, original_w = image_rgb.shape[:2]
        self.original_size = (original_h, original_w)

        image_1024 = cv2.resize(
            image_rgb,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_CUBIC,
        )

        image_1024 = image_1024.astype(np.float32) / 255.0

        image_tensor = torch.tensor(image_1024, dtype=torch.float32)
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            self.image_embedding = self.model.image_encoder(image_tensor)

    def predict(self, box_xyxy):
        """
        Run MedSAM box-prompt inference.

        Args:
            box_xyxy:
                [x1, y1, x2, y2] in original image coordinates.

        Returns:
            binary_mask:
                uint8 mask in original image size, values {0, 255}.
            mean_probability:
                Mean probability of predicted probability map.
        """

        if self.image_embedding is None or self.original_size is None:
            raise RuntimeError("Call set_image(image_rgb) before predict().")

        original_h, original_w = self.original_size

        box_np = np.array(box_xyxy, dtype=np.float32)

        if box_np.shape != (4,):
            raise ValueError(f"box_xyxy must have shape (4,), got {box_np.shape}")

        box_1024 = box_np / np.array(
            [original_w, original_h, original_w, original_h],
            dtype=np.float32,
        ) * self.image_size

        box_torch = torch.tensor(
            box_1024,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )

            low_res_logits, _ = self.model.mask_decoder(
                image_embeddings=self.image_embedding,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            mask_prob = torch.sigmoid(low_res_logits)

            mask_prob = torch_functional.interpolate(
                mask_prob,
                size=(original_h, original_w),
                mode="bilinear",
                align_corners=False,
            )

        mask_prob_np = mask_prob.squeeze().detach().cpu().numpy()
        binary_mask = (mask_prob_np > 0.5).astype(np.uint8) * 255

        return binary_mask, float(mask_prob_np.mean())