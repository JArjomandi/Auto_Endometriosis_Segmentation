from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from Hybrid.hybrid_utils import (
    find_file_by_root,
    list_image_files,
    read_rgb_image,
    read_binary_mask,
    save_binary_mask,
    save_rgb_image,
    save_grayscale_uint8,
    save_json,
    mask_to_tight_box,
    compute_binary_metrics,
    make_comparison_overlay,
    Timer,
)
from Hybrid.segformer_inference import (
    build_segformer_model,
    predict_segformer_mask,
)
from Hybrid.surgisam2_inference import (
    build_surgisam2_predictor,
    predict_surgisam2_from_box,
)


# =============================================================================
# Configuration
# =============================================================================

RESULTS_ROOT = Path(r"F:\Results\SAM_Benchmarking")

STANDARDIZED_DATASETS = {
    "ENID": Path(r"F:\Datasets\Standardized datasets\ENID\ENID 60_20_20 Split"),
    "GLENDA": Path(r"F:\Datasets\Standardized datasets\GLENDA\GLENDA 60_20_20 split"),
    "GLENDA_clean": Path(r"F:\Datasets\Standardized datasets\GLENDA_clean\GLENDA_clean 60_20_20 split"),
}


SEGFORMER_CHECKPOINTS = {
    "ENID": Path(r"F:\Results\SAM_Benchmarking\ENID\SegFormer\trained\checkpoints\best_SegFormer.pt"),
    "GLENDA": Path(r"F:\Results\SAM_Benchmarking\GLENDA\SegFormer\trained\checkpoints\best_SegFormer.pt"),
    "GLENDA_clean": Path(r"F:\Results\SAM_Benchmarking\GLENDA_clean\SegFormer\trained\checkpoints\best_SegFormer.pt"),
}


SEGFORMER_PRETRAINED_MODEL_NAME = "nvidia/segformer-b2-finetuned-ade-512-512"
SEGFORMER_INPUT_SIZE = 512
SEGFORMER_THRESHOLD = 0.5


SURGISAM2_SAM2_REPO_ROOT = Path(r"F:\Models\SAM2")
SURGISAM2_REPO_ROOT = Path(r"F:\Models\SurgiSAM2")
SURGISAM2_CHECKPOINT_PATH = Path(
    r"F:\Models\SurgiSAM2\checkpoints\Curated400_checkpoint_image_predictor.pt"
)
SURGISAM2_MODEL_CFG = "configs/sam2/sam2_hiera_b+.yaml"

SURGISAM2_MULTIMASK_OUTPUT = True
SURGISAM2_MASK_SELECTION = "highest_score"
SURGISAM2_USE_BFLOAT16 = True


METHOD_FOLDER_NAME = "SegFormer_SurgiSAM2_AutoBox"
TRAINING_STATE_FOLDER = "hybrid"
PROMPT_MODE_FOLDER = "Auto_box"


DATASETS_TO_RUN = [
    "ENID",
    "GLENDA",
    "GLENDA_clean",
]


SPLITS_TO_RUN = [
    "val",
    "test",
]


# Set to 5 for first debug run.
# Set to None for full val/test run.
MAX_IMAGES_DEBUG = None


# Tight prompt settings.
# For the tightest possible box, keep both values at zero.
BOX_PADDING_PX = 0
BOX_PADDING_RATIO = 0.0

# Recommended for avoiding huge boxes caused by tiny disconnected false-positive islands.
BOX_LARGEST_COMPONENT_ONLY = True

# Keep 0 first. Later you may try 20, 50, or 100 if SegFormer produces tiny islands.
BOX_MIN_COMPONENT_AREA_PX = 0


SAVE_PROBABILITY_MAPS = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Folder helpers
# =============================================================================

def make_output_dirs(dataset_key: str, split_key: str):
    split_output_dir = (
        RESULTS_ROOT
        / dataset_key
        / METHOD_FOLDER_NAME
        / TRAINING_STATE_FOLDER
        / PROMPT_MODE_FOLDER
        / split_key
    )

    dirs = {
        "root": split_output_dir,
        "final_masks": split_output_dir / "merged_masks",
        "final_overlays": split_output_dir / "overlays",
        "segformer_masks": split_output_dir / "segformer_initial_masks",
        "segformer_prompt_masks": split_output_dir / "segformer_prompt_masks_largest_component",
        "segformer_overlays": split_output_dir / "segformer_initial_overlays",
        "probability_maps": split_output_dir / "segformer_probability_maps",
        "prompts": split_output_dir / "auto_prompts",
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


def save_run_config(output_dir: Path):
    config = {
        "method_folder_name": METHOD_FOLDER_NAME,
        "training_state_folder": TRAINING_STATE_FOLDER,
        "prompt_mode_folder": PROMPT_MODE_FOLDER,
        "datasets_to_run": DATASETS_TO_RUN,
        "splits_to_run": SPLITS_TO_RUN,
        "max_images_debug": MAX_IMAGES_DEBUG,
        "segformer_pretrained_model_name": SEGFORMER_PRETRAINED_MODEL_NAME,
        "segformer_input_size": SEGFORMER_INPUT_SIZE,
        "segformer_threshold": SEGFORMER_THRESHOLD,
        "box_padding_px": BOX_PADDING_PX,
        "box_padding_ratio": BOX_PADDING_RATIO,
        "box_largest_component_only": BOX_LARGEST_COMPONENT_ONLY,
        "box_min_component_area_px": BOX_MIN_COMPONENT_AREA_PX,
        "save_probability_maps": SAVE_PROBABILITY_MAPS,
        "device": str(DEVICE),
        "surgisam2_sam2_repo_root": str(SURGISAM2_SAM2_REPO_ROOT),
        "surgisam2_repo_root": str(SURGISAM2_REPO_ROOT),
        "surgisam2_checkpoint_path": str(SURGISAM2_CHECKPOINT_PATH),
        "surgisam2_model_cfg": SURGISAM2_MODEL_CFG,
        "surgisam2_multimask_output": SURGISAM2_MULTIMASK_OUTPUT,
        "surgisam2_mask_selection": SURGISAM2_MASK_SELECTION,
        "surgisam2_use_bfloat16": SURGISAM2_USE_BFLOAT16,
    }

    config_path = output_dir / "run_config.json"

    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)


def box_area(box_xyxy):
    if box_xyxy is None:
        return 0

    x_min, y_min, x_max, y_max = box_xyxy
    return max(0, x_max - x_min + 1) * max(0, y_max - y_min + 1)


# =============================================================================
# Main split runner
# =============================================================================

def run_dataset_split(
    dataset_key: str,
    split_key: str,
    segformer_model,
    surgisam2_predictor,
):
    dataset_root = STANDARDIZED_DATASETS[dataset_key]

    image_folder = dataset_root / split_key / "images"
    mask_folder = dataset_root / split_key / "masks"

    if not image_folder.exists():
        raise FileNotFoundError(f"Missing image folder: {image_folder}")

    if not mask_folder.exists():
        raise FileNotFoundError(f"Missing mask folder: {mask_folder}")

    output_dirs = make_output_dirs(
        dataset_key=dataset_key,
        split_key=split_key,
    )

    save_run_config(output_dirs["root"])

    image_files = list_image_files(image_folder)

    if MAX_IMAGES_DEBUG is not None:
        image_files = image_files[:MAX_IMAGES_DEBUG]

    print("\n" + "=" * 100)
    print(f"Running hybrid: {dataset_key} | {split_key}")
    print(f"Images: {len(image_files)}")
    print(f"Output: {output_dirs['root']}")
    print("=" * 100)

    metric_rows = []
    time_rows = []

    for image_index, image_path in enumerate(image_files, start=1):
        root_name = image_path.stem

        mask_path = find_file_by_root(mask_folder, root_name)

        if mask_path is None:
            print(f"WARNING: Missing GT mask for {image_path.name}. Skipping.")
            continue

        print(f"[{image_index}/{len(image_files)}] {root_name}")

        image_np = read_rgb_image(image_path)
        gt_mask = read_binary_mask(mask_path)

        with Timer() as total_timer:
            with Timer() as segformer_timer:
                segformer_mask, segformer_probability = predict_segformer_mask(
                    model=segformer_model,
                    image_np=image_np,
                    device=DEVICE,
                    input_size=SEGFORMER_INPUT_SIZE,
                    threshold=SEGFORMER_THRESHOLD,
                )

            with Timer() as prompt_timer:
                box_xyxy, segformer_prompt_mask = mask_to_tight_box(
                    mask_np=segformer_mask,
                    image_shape=image_np.shape,
                    padding_px=BOX_PADDING_PX,
                    padding_ratio=BOX_PADDING_RATIO,
                    largest_component_only=BOX_LARGEST_COMPONENT_ONLY,
                    min_component_area_px=BOX_MIN_COMPONENT_AREA_PX,
                )

            with Timer() as surgisam2_timer:
                if box_xyxy is None:
                    final_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
                    surgisam2_score = None
                    surgisam2_scores_all = None
                    prompt_status = "empty_segformer_mask"
                else:
                    final_mask, surgisam2_score, surgisam2_scores_all = predict_surgisam2_from_box(
                        predictor=surgisam2_predictor,
                        image_np=image_np,
                        box_xyxy=box_xyxy,
                        multimask_output=SURGISAM2_MULTIMASK_OUTPUT,
                        mask_selection=SURGISAM2_MASK_SELECTION,
                        use_bfloat16=SURGISAM2_USE_BFLOAT16,
                    )
                    prompt_status = "tight_box_prompt_used"

        final_metrics = compute_binary_metrics(
            pred_mask=final_mask,
            gt_mask=gt_mask,
        )

        segformer_metrics = compute_binary_metrics(
            pred_mask=segformer_mask,
            gt_mask=gt_mask,
        )

        segformer_prompt_metrics = compute_binary_metrics(
            pred_mask=segformer_prompt_mask,
            gt_mask=gt_mask,
        )

        final_overlay = make_comparison_overlay(
            image_np=image_np,
            gt_mask=gt_mask,
            pred_mask=final_mask,
            box_xyxy=box_xyxy,
            box_color=(180, 0, 255),
            box_width=4,
        )

        segformer_overlay = make_comparison_overlay(
            image_np=image_np,
            gt_mask=gt_mask,
            pred_mask=segformer_mask,
            box_xyxy=box_xyxy,
            box_color=(180, 0, 255),
            box_width=4,
        )

        final_mask_path = output_dirs["final_masks"] / f"{root_name}.png"
        final_overlay_path = output_dirs["final_overlays"] / f"{root_name}_overlay.png"

        segformer_mask_path = output_dirs["segformer_masks"] / f"{root_name}.png"
        segformer_prompt_mask_path = output_dirs["segformer_prompt_masks"] / f"{root_name}.png"
        segformer_overlay_path = output_dirs["segformer_overlays"] / f"{root_name}_overlay.png"

        prompt_path = output_dirs["prompts"] / f"{root_name}.json"

        save_binary_mask(final_mask, final_mask_path)
        save_rgb_image(final_overlay, final_overlay_path)

        save_binary_mask(segformer_mask, segformer_mask_path)
        save_binary_mask(segformer_prompt_mask, segformer_prompt_mask_path)
        save_rgb_image(segformer_overlay, segformer_overlay_path)

        if SAVE_PROBABILITY_MAPS:
            probability_uint8 = np.clip(
                segformer_probability * 255,
                0,
                255,
            ).astype(np.uint8)

            probability_path = output_dirs["probability_maps"] / f"{root_name}.png"
            save_grayscale_uint8(probability_uint8, probability_path)

        prompt_data = {
            "image_name": image_path.name,
            "root_name": root_name,
            "prompt_type": "tight_box",
            "prompt_status": prompt_status,
            "box_xyxy": box_xyxy,
            "box_area_px": box_area(box_xyxy),
            "box_padding_px": BOX_PADDING_PX,
            "box_padding_ratio": BOX_PADDING_RATIO,
            "box_largest_component_only": BOX_LARGEST_COMPONENT_ONLY,
            "box_min_component_area_px": BOX_MIN_COMPONENT_AREA_PX,
            "segformer_pred_area": int((segformer_mask > 0).sum()),
            "segformer_prompt_component_area": int((segformer_prompt_mask > 0).sum()),
            "final_pred_area": int((final_mask > 0).sum()),
            "gt_area": int((gt_mask > 0).sum()),
            "surgisam2_best_score": surgisam2_score,
            "surgisam2_scores_all": surgisam2_scores_all,
        }

        save_json(prompt_data, prompt_path)

        metric_row = {
            "image_name": image_path.name,
            "mask_name": mask_path.name,
            "case_id": root_name,
            "prediction_name": final_mask_path.name,
            "overlay_name": final_overlay_path.name,
            "prompt_status": prompt_status,
            "box_xmin": box_xyxy[0] if box_xyxy is not None else "",
            "box_ymin": box_xyxy[1] if box_xyxy is not None else "",
            "box_xmax": box_xyxy[2] if box_xyxy is not None else "",
            "box_ymax": box_xyxy[3] if box_xyxy is not None else "",
            "box_area_px": box_area(box_xyxy),
            "surgisam2_best_score": surgisam2_score if surgisam2_score is not None else "",
            "dice": final_metrics["dice"],
            "iou": final_metrics["iou"],
            "precision": final_metrics["precision"],
            "recall": final_metrics["recall"],
            "tp": final_metrics["tp"],
            "fp": final_metrics["fp"],
            "fn": final_metrics["fn"],
            "tn": final_metrics["tn"],
            "pred_area": final_metrics["pred_area"],
            "gt_area": final_metrics["gt_area"],
            "segformer_initial_dice": segformer_metrics["dice"],
            "segformer_initial_iou": segformer_metrics["iou"],
            "segformer_initial_precision": segformer_metrics["precision"],
            "segformer_initial_recall": segformer_metrics["recall"],
            "segformer_initial_pred_area": segformer_metrics["pred_area"],
            "segformer_prompt_component_dice": segformer_prompt_metrics["dice"],
            "segformer_prompt_component_iou": segformer_prompt_metrics["iou"],
            "segformer_prompt_component_precision": segformer_prompt_metrics["precision"],
            "segformer_prompt_component_recall": segformer_prompt_metrics["recall"],
            "segformer_prompt_component_area": segformer_prompt_metrics["pred_area"],
        }

        time_row = {
            "image_name": image_path.name,
            "case_id": root_name,
            "total_time_sec": total_timer.elapsed,
            "segformer_time_sec": segformer_timer.elapsed,
            "prompt_generation_time_sec": prompt_timer.elapsed,
            "surgisam2_time_sec": surgisam2_timer.elapsed,
        }

        metric_rows.append(metric_row)
        time_rows.append(time_row)

    metrics_df = pd.DataFrame(metric_rows)
    times_df = pd.DataFrame(time_rows)

    metrics_path = output_dirs["root"] / "metrics_image_level.csv"
    times_path = output_dirs["root"] / "inference_times.csv"
    summary_path = output_dirs["root"] / "metrics_summary.csv"

    metrics_df.to_csv(metrics_path, index=False)
    times_df.to_csv(times_path, index=False)

    summary_rows = []

    metric_columns = [
        "dice",
        "iou",
        "precision",
        "recall",
        "segformer_initial_dice",
        "segformer_initial_iou",
        "segformer_initial_precision",
        "segformer_initial_recall",
        "segformer_prompt_component_dice",
        "segformer_prompt_component_iou",
        "segformer_prompt_component_precision",
        "segformer_prompt_component_recall",
    ]

    for metric_name in metric_columns:
        if metric_name not in metrics_df.columns:
            continue

        values = pd.to_numeric(metrics_df[metric_name], errors="coerce").dropna()

        if len(values) == 0:
            continue

        summary_rows.append(
            {
                "metric": metric_name,
                "n": len(values),
                "mean": values.mean(),
                "std": values.std(),
                "median": values.median(),
                "q1": values.quantile(0.25),
                "q3": values.quantile(0.75),
                "min": values.min(),
                "max": values.max(),
            }
        )

    time_columns = [
        "total_time_sec",
        "segformer_time_sec",
        "prompt_generation_time_sec",
        "surgisam2_time_sec",
    ]

    for time_name in time_columns:
        if time_name not in times_df.columns:
            continue

        values = pd.to_numeric(times_df[time_name], errors="coerce").dropna()

        if len(values) == 0:
            continue

        summary_rows.append(
            {
                "metric": time_name,
                "n": len(values),
                "mean": values.mean(),
                "std": values.std(),
                "median": values.median(),
                "q1": values.quantile(0.25),
                "q3": values.quantile(0.75),
                "min": values.min(),
                "max": values.max(),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved metrics: {metrics_path}")
    print(f"Saved times:   {times_path}")
    print(f"Saved summary: {summary_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    print(f"Device: {DEVICE}")

    surgisam2_predictor = build_surgisam2_predictor(
        sam2_repo_root=SURGISAM2_SAM2_REPO_ROOT,
        surgisam2_repo_root=SURGISAM2_REPO_ROOT,
        model_cfg=SURGISAM2_MODEL_CFG,
        checkpoint_path=SURGISAM2_CHECKPOINT_PATH,
        device=DEVICE,
    )

    for dataset_key in DATASETS_TO_RUN:
        checkpoint_path = SEGFORMER_CHECKPOINTS[dataset_key]

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing SegFormer checkpoint: {checkpoint_path}")

        print("\n" + "#" * 100)
        print(f"Loading SegFormer for {dataset_key}")
        print(f"Checkpoint: {checkpoint_path}")
        print("#" * 100)

        segformer_model = build_segformer_model(
            checkpoint_path=checkpoint_path,
            device=DEVICE,
            pretrained_model_name=SEGFORMER_PRETRAINED_MODEL_NAME,
            num_labels=1,
        )

        for split_key in SPLITS_TO_RUN:
            run_dataset_split(
                dataset_key=dataset_key,
                split_key=split_key,
                segformer_model=segformer_model,
                surgisam2_predictor=surgisam2_predictor,
            )

        del segformer_model
        torch.cuda.empty_cache()

    print("\nDONE.")


if __name__ == "__main__":
    main()