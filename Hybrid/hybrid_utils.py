from pathlib import Path
import json
import time

import numpy as np
from PIL import Image, ImageDraw


IMAGE_EXTENSIONS = [
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
]


def get_nearest_resampling():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.NEAREST
    return Image.NEAREST


def get_bilinear_resampling():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.BILINEAR
    return Image.BILINEAR


def find_file_by_root(folder: Path, root_name: str):
    for extension in IMAGE_EXTENSIONS:
        candidate = folder / f"{root_name}{extension}"
        if candidate.exists():
            return candidate

    matches = []

    for extension in IMAGE_EXTENSIONS:
        matches.extend(sorted(folder.glob(f"{root_name}*{extension}")))

    if matches:
        return matches[0]

    return None


def list_image_files(image_folder: Path):
    image_files = []

    for extension in IMAGE_EXTENSIONS:
        image_files.extend(sorted(image_folder.glob(f"*{extension}")))

    return sorted(image_files)


def read_rgb_image(image_path: Path):
    image = Image.open(image_path).convert("RGB")
    return np.array(image)


def read_binary_mask(mask_path: Path):
    mask = Image.open(mask_path).convert("L")
    mask_np = np.array(mask)
    return (mask_np > 0).astype(np.uint8)


def save_binary_mask(mask_np: np.ndarray, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_uint8 = (mask_np > 0).astype(np.uint8) * 255
    Image.fromarray(mask_uint8).save(output_path)


def save_grayscale_uint8(image_np: np.ndarray, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_np = np.asarray(image_np)
    image_np = np.clip(image_np, 0, 255).astype(np.uint8)
    Image.fromarray(image_np).save(output_path)


def save_rgb_image(image_np: np.ndarray, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_np.astype(np.uint8)).save(output_path)


def save_json(data: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def resize_mask_to_image(mask_np: np.ndarray, image_shape):
    target_height, target_width = image_shape[:2]

    if mask_np.shape[:2] == (target_height, target_width):
        return (mask_np > 0).astype(np.uint8)

    mask_image = Image.fromarray((mask_np > 0).astype(np.uint8) * 255)
    mask_image = mask_image.resize(
        (target_width, target_height),
        resample=get_nearest_resampling(),
    )

    return (np.array(mask_image) > 0).astype(np.uint8)


def remove_small_components(mask_np: np.ndarray, min_area_px: int):
    mask_np = (mask_np > 0).astype(np.uint8)

    if min_area_px <= 0 or mask_np.sum() == 0:
        return mask_np

    try:
        from scipy import ndimage
    except Exception:
        return mask_np

    labeled, num_labels = ndimage.label(mask_np)

    if num_labels == 0:
        return mask_np

    output = np.zeros_like(mask_np, dtype=np.uint8)
    component_sizes = np.bincount(labeled.ravel())

    for label_id in range(1, num_labels + 1):
        if component_sizes[label_id] >= min_area_px:
            output[labeled == label_id] = 1

    return output


def mask_to_component_boxes(
    mask_np: np.ndarray,
    image_shape,
    padding_px: int = 0,
    padding_ratio: float = 0.0,
    min_component_area_px: int = 0,
    max_components=None,
    return_component_masks: bool = False,
):
    """
    Converts one SegFormer binary mask into multiple tight boxes.

    Each connected component becomes one box.

    Returns:
        boxes_xyxy:
            list of [x_min, y_min, x_max, y_max]

        component_mask:
            binary mask containing all components used for prompting

        component_infos:
            list of dicts with label_id, area, and box info

        component_masks:
            optional list of exact binary masks, one per connected component.
            Returned only if return_component_masks=True.
    """

    mask_np = resize_mask_to_image(mask_np, image_shape)
    mask_np = (mask_np > 0).astype(np.uint8)

    image_height, image_width = image_shape[:2]

    if mask_np.sum() == 0:
        if return_component_masks:
            return [], mask_np, [], []
        return [], mask_np, []

    try:
        from scipy import ndimage
    except Exception as error:
        raise ImportError(
            "scipy is required for multi-component box extraction. "
            "Install it with: pip install scipy"
        ) from error

    labeled, num_labels = ndimage.label(mask_np)

    if num_labels == 0:
        empty_mask = np.zeros_like(mask_np, dtype=np.uint8)

        if return_component_masks:
            return [], empty_mask, [], []

        return [], empty_mask, []

    component_infos = []

    for label_id in range(1, num_labels + 1):
        component = labeled == label_id
        area_px = int(component.sum())

        if area_px < min_component_area_px:
            continue

        ys, xs = np.where(component)

        if len(xs) == 0 or len(ys) == 0:
            continue

        x_min = int(xs.min())
        x_max = int(xs.max())
        y_min = int(ys.min())
        y_max = int(ys.max())

        ratio_pad_x = int(round(padding_ratio * image_width))
        ratio_pad_y = int(round(padding_ratio * image_height))

        total_pad_x = int(padding_px) + ratio_pad_x
        total_pad_y = int(padding_px) + ratio_pad_y

        padded_x_min = max(0, x_min - total_pad_x)
        padded_y_min = max(0, y_min - total_pad_y)
        padded_x_max = min(image_width - 1, x_max + total_pad_x)
        padded_y_max = min(image_height - 1, y_max + total_pad_y)

        if padded_x_max <= padded_x_min or padded_y_max <= padded_y_min:
            continue

        box_xyxy = [
            padded_x_min,
            padded_y_min,
            padded_x_max,
            padded_y_max,
        ]

        box_area_px = int(
            (padded_x_max - padded_x_min + 1)
            * (padded_y_max - padded_y_min + 1)
        )

        component_infos.append(
            {
                "label_id": int(label_id),
                "area_px": area_px,
                "box_xyxy": box_xyxy,
                "box_area_px": box_area_px,
                "tight_box_xyxy_without_padding": [
                    x_min,
                    y_min,
                    x_max,
                    y_max,
                ],
            }
        )

    component_infos = sorted(
        component_infos,
        key=lambda item: item["area_px"],
        reverse=True,
    )

    if max_components is not None:
        component_infos = component_infos[:max_components]

    filtered_component_mask = np.zeros_like(mask_np, dtype=np.uint8)
    boxes_xyxy = []
    component_masks = []

    for component_info in component_infos:
        label_id = component_info["label_id"]

        exact_component_mask = (labeled == label_id).astype(np.uint8)

        filtered_component_mask = np.logical_or(
            filtered_component_mask > 0,
            exact_component_mask > 0,
        ).astype(np.uint8)

        boxes_xyxy.append(component_info["box_xyxy"])
        component_masks.append(exact_component_mask)

    if return_component_masks:
        return boxes_xyxy, filtered_component_mask, component_infos, component_masks

    return boxes_xyxy, filtered_component_mask, component_infos


def compute_binary_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray):
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)

    if pred.shape != gt.shape:
        pred = resize_mask_to_image(pred, gt.shape)

    pred_bool = pred > 0
    gt_bool = gt > 0

    tp = int(np.logical_and(pred_bool, gt_bool).sum())
    fp = int(np.logical_and(pred_bool, ~gt_bool).sum())
    fn = int(np.logical_and(~pred_bool, gt_bool).sum())
    tn = int(np.logical_and(~pred_bool, ~gt_bool).sum())

    dice_denominator = (2 * tp + fp + fn)
    iou_denominator = (tp + fp + fn)
    precision_denominator = (tp + fp)
    recall_denominator = (tp + fn)

    dice = (2 * tp / dice_denominator) if dice_denominator > 0 else 1.0
    iou = (tp / iou_denominator) if iou_denominator > 0 else 1.0
    precision = (tp / precision_denominator) if precision_denominator > 0 else 1.0
    recall = (tp / recall_denominator) if recall_denominator > 0 else 1.0

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "pred_area": int(pred_bool.sum()),
        "gt_area": int(gt_bool.sum()),
    }


def make_comparison_overlay(
    image_np: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    alpha: float = 0.55,
    boxes_xyxy=None,
    box_color=(180, 0, 255),
    box_width: int = 4,
):
    """
    Green  = GT only / missed lesion
    Red    = prediction only / false positive
    Yellow = overlap / true positive
    Purple boxes = SegFormer-derived prompt boxes
    """

    gt_mask = resize_mask_to_image(gt_mask, image_np.shape)
    pred_mask = resize_mask_to_image(pred_mask, image_np.shape)

    gt_bool = gt_mask > 0
    pred_bool = pred_mask > 0

    overlap_bool = gt_bool & pred_bool
    gt_only_bool = gt_bool & (~pred_bool)
    pred_only_bool = pred_bool & (~gt_bool)

    image_float = image_np.astype(np.float32)
    overlay = image_float.copy()

    gt_only_color = np.array([0, 255, 0], dtype=np.float32)
    pred_only_color = np.array([255, 0, 0], dtype=np.float32)
    overlap_color = np.array([255, 255, 0], dtype=np.float32)

    overlay[gt_only_bool] = (
        (1.0 - alpha) * image_float[gt_only_bool]
        + alpha * gt_only_color
    )

    overlay[pred_only_bool] = (
        (1.0 - alpha) * image_float[pred_only_bool]
        + alpha * pred_only_color
    )

    overlay[overlap_bool] = (
        (1.0 - alpha) * image_float[overlap_bool]
        + alpha * overlap_color
    )

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    if boxes_xyxy is not None and len(boxes_xyxy) > 0:
        overlay_pil = Image.fromarray(overlay)
        draw = ImageDraw.Draw(overlay_pil)

        for current_box in boxes_xyxy:
            x_min, y_min, x_max, y_max = [int(v) for v in current_box]

            for offset in range(box_width):
                draw.rectangle(
                    [
                        x_min - offset,
                        y_min - offset,
                        x_max + offset,
                        y_max + offset,
                    ],
                    outline=tuple(box_color),
                )

        overlay = np.array(overlay_pil)

    return overlay


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start