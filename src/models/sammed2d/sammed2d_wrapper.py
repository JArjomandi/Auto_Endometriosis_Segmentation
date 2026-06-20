from pathlib import Path
import sys
import importlib

import cv2
import numpy as np
import torch
import torch.nn.functional as torch_functional


class SAMMed2DFrozenWrapper:
    """
    Frozen SAM-Med2D wrapper.

    Supports:
        GT_point
        GT_box

    Input prompts use original image coordinates.
    Output mask is resized back to original image size.
    """

    def __init__(
        self,
        sammed2d_repo_root: str,
        checkpoint: str,
        device: str = "cuda",
        model_type: str = "vit_b",
        image_size: int = 256,
        encoder_adapter: bool = True,
        multimask_output: bool = True,
        mask_selection: str = "highest_score",
    ):
        self.sammed2d_repo_root = Path(sammed2d_repo_root)
        self.checkpoint = Path(checkpoint)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.image_size = int(image_size)
        self.encoder_adapter = bool(encoder_adapter)
        self.multimask_output = bool(multimask_output)
        self.mask_selection = mask_selection

        if not self.sammed2d_repo_root.exists():
            raise FileNotFoundError(f"SAM-Med2D repo not found: {self.sammed2d_repo_root}")

        if not self.checkpoint.exists():
            raise FileNotFoundError(f"SAM-Med2D checkpoint not found: {self.checkpoint}")

        repo_root_str = str(self.sammed2d_repo_root)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)

        segment_anything_module = importlib.import_module("segment_anything")
        sam_model_registry = getattr(segment_anything_module, "sam_model_registry")

        class Args:
            pass

        args = Args()
        args.model_type = self.model_type
        args.image_size = self.image_size
        args.sam_checkpoint = str(self.checkpoint)
        args.encoder_adapter = self.encoder_adapter

        self.model = sam_model_registry[self.model_type](args)
        self.model.to(self.device)
        self.model.eval()

        self.image_embedding = None
        self.original_size = None
        self.input_size = None

    def _preprocess_image(self, image_rgb: np.ndarray):
        original_h, original_w = image_rgb.shape[:2]
        self.original_size = (original_h, original_w)

        image_resized = cv2.resize(
            image_rgb,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_CUBIC,
        )

        image_resized = image_resized.astype(np.float32) / 255.0

        image_tensor = torch.tensor(image_resized, dtype=torch.float32)
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        return image_tensor

    def set_image(self, image_rgb: np.ndarray):
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("image_rgb must be an H x W x 3 RGB image.")

        image_tensor = self._preprocess_image(image_rgb)

        with torch.no_grad():
            self.image_embedding = self.model.image_encoder(image_tensor)

    def _scale_box_to_model_space(self, box_xyxy):
        original_h, original_w = self.original_size

        box_np = np.array(box_xyxy, dtype=np.float32)

        box_scaled = box_np / np.array(
            [original_w, original_h, original_w, original_h],
            dtype=np.float32,
        ) * self.image_size

        return torch.tensor(
            box_scaled,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

    def _scale_points_to_model_space(self, point_coords):
        original_h, original_w = self.original_size

        points_np = np.array(point_coords, dtype=np.float32)

        points_scaled = points_np / np.array(
            [original_w, original_h],
            dtype=np.float32,
        ) * self.image_size

        return torch.tensor(
            points_scaled,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

    def predict(
        self,
        box=None,
        point_coords=None,
        point_labels=None,
    ):
        if self.image_embedding is None or self.original_size is None:
            raise RuntimeError("Call set_image(image_rgb) before predict().")

        original_h, original_w = self.original_size

        box_torch = None
        points_tuple = None

        if box is not None:
            box_torch = self._scale_box_to_model_space(box)

        if point_coords is not None and point_labels is not None:
            point_coords_torch = self._scale_points_to_model_space(point_coords)
            point_labels_torch = torch.tensor(
                point_labels,
                dtype=torch.int64,
                device=self.device,
            ).unsqueeze(0)

            points_tuple = (point_coords_torch, point_labels_torch)

        with torch.no_grad():
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=points_tuple,
                boxes=box_torch,
                masks=None,
            )

            low_res_logits, _ = self.model.mask_decoder(
                image_embeddings=self.image_embedding,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=self.multimask_output,
            )

            mask_prob = torch.sigmoid(low_res_logits)

            if self.multimask_output:
                scores = mask_prob.flatten(2).mean(dim=-1).squeeze(0)
                selected_index = int(torch.argmax(scores).item())
                selected_prob = mask_prob[:, selected_index:selected_index + 1, :, :]
                selected_score = float(scores[selected_index].item())
            else:
                selected_index = 0
                selected_prob = mask_prob
                selected_score = float(mask_prob.mean().item())

            selected_prob = torch_functional.interpolate(
                selected_prob,
                size=(original_h, original_w),
                mode="bilinear",
                align_corners=False,
            )

        mask_prob_np = selected_prob.squeeze().detach().cpu().numpy()
        binary_mask = (mask_prob_np > 0.5).astype(np.uint8) * 255

        return binary_mask, selected_score, selected_index