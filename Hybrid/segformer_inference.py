from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation


def build_segformer_model(
    checkpoint_path: Path,
    device: torch.device,
    pretrained_model_name: str = "nvidia/segformer-b2-finetuned-ade-512-512",
    num_labels: int = 1,
):
    """
    Loads your trained Hugging Face SegFormer checkpoint.

    Your YAML:
        pretrained_model_name: nvidia/segformer-b2-finetuned-ade-512-512
        num_labels: 1
        image_size: 512
        threshold: 0.5
    """

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing SegFormer checkpoint: {checkpoint_path}")

    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )

    checkpoint = torch.load(
        str(checkpoint_path),
        map_location=device,
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        raise ValueError(
            f"Unsupported SegFormer checkpoint format: {type(checkpoint)}"
        )

    cleaned_state_dict = {}

    for key, value in state_dict.items():
        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[len("module."):]

        if new_key.startswith("model."):
            new_key = new_key[len("model."):]

        cleaned_state_dict[new_key] = value

    missing_keys, unexpected_keys = model.load_state_dict(
        cleaned_state_dict,
        strict=False,
    )

    if missing_keys:
        print("WARNING: Missing SegFormer keys:")
        print(missing_keys[:30])

    if unexpected_keys:
        print("WARNING: Unexpected SegFormer keys:")
        print(unexpected_keys[:30])

    model.to(device)
    model.eval()

    return model


def preprocess_for_segformer(
    image_np: np.ndarray,
    input_size: int = 512,
    device: torch.device = torch.device("cuda"),
):
    """
    SegFormer preprocessing matching your YAML image_size=512.

    Uses ImageNet normalization, which is standard for SegFormer ADE-pretrained
    backbones unless your training script used different normalization.
    """

    image = Image.fromarray(image_np.astype(np.uint8)).convert("RGB")
    image = image.resize(
        (input_size, input_size),
        resample=Image.BILINEAR,
    )

    image_np_resized = np.array(image).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    image_np_resized = (image_np_resized - mean) / std

    tensor = torch.from_numpy(image_np_resized).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device=device, dtype=torch.float32)

    return tensor


@torch.no_grad()
def predict_segformer_mask(
    model,
    image_np: np.ndarray,
    device: torch.device,
    input_size: int = 512,
    threshold: float = 0.5,
):
    original_height, original_width = image_np.shape[:2]

    input_tensor = preprocess_for_segformer(
        image_np=image_np,
        input_size=input_size,
        device=device,
    )

    outputs = model(pixel_values=input_tensor)
    logits = outputs.logits

    logits = F.interpolate(
        logits,
        size=(original_height, original_width),
        mode="bilinear",
        align_corners=False,
    )

    if logits.shape[1] == 1:
        probability = torch.sigmoid(logits[:, 0, :, :])
    else:
        probability = torch.softmax(logits, dim=1)[:, 1, :, :]

    probability_np = probability.squeeze(0).detach().cpu().numpy()
    mask_np = (probability_np >= threshold).astype(np.uint8)

    return mask_np, probability_np