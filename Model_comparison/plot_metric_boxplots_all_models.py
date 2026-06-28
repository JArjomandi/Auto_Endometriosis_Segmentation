from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_ROOT = Path(r"F:\Results\SAM_Benchmarking")
OUTPUT_ROOT = RESULTS_ROOT / "Model_comparison" / "metric_boxplots"


DATASETS = [
    {
        "key": "ENID",
        "title": "ENID dataset",
    },
    {
        "key": "GLENDA",
        "title": "GLENDA dataset",
    },
    {
        "key": "GLENDA_clean",
        "title": "cleaned GLENDA dataset",
    },
]


SPLITS = [
    {
        "key": "val",
        "title": "Validation",
    },
    {
        "key": "test",
        "title": "Test",
    },
]


SAM_PROMPT_MODE_TO_COMPARE = None

SAM_PROMPT_MODE_CANDIDATES = [
    "GT_box",
    "GT_box_point",
    "GT_point",
    "Box_prompt",
    "Point_prompt",
    "No_prompt",
    "Auto_YOLO_box",
]


SAM_CYAN = "#00BDD6"

MODEL_COLORS = {
    "SAM2": SAM_CYAN,
    "MedSAM": SAM_CYAN,
    "SAM-Med2D": SAM_CYAN,
    "SurgiSAM2": SAM_CYAN,
    "YOLO11s-seg": "#9467bd",
    "DeepLabV3+": "#ff7f0e",
    "SegFormer": "#2ca02c",
    "UNet++": "#1f77b4",
    "nnU-Net v2 2D": "#7f7f7f",
}


MODEL_SPECS_TOP_TO_BOTTOM = [
    {
        "display_name": "SAM2",
        "folder_candidates": ["SAM2"],
        "training_state": "frozen",
        "prompt_mode": SAM_PROMPT_MODE_TO_COMPARE,
        "is_sam": True,
    },
    {
        "display_name": "MedSAM",
        "folder_candidates": ["MedSAM"],
        "training_state": "frozen",
        "prompt_mode": SAM_PROMPT_MODE_TO_COMPARE,
        "is_sam": True,
    },
    {
        "display_name": "SAM-Med2D",
        "folder_candidates": ["SAM-Med2D", "SAMMed2D", "SAM_Med2D"],
        "training_state": "frozen",
        "prompt_mode": SAM_PROMPT_MODE_TO_COMPARE,
        "is_sam": True,
    },
    {
        "display_name": "SurgiSAM2",
        "folder_candidates": ["SurgiSAM2"],
        "training_state": "frozen",
        "prompt_mode": SAM_PROMPT_MODE_TO_COMPARE,
        "is_sam": True,
    },
    {
        "display_name": "YOLO11s-seg",
        "folder_candidates": ["YOLO11s_seg"],
        "training_state": "trained",
        "prompt_mode": "No_prompt",
        "is_sam": False,
    },
    {
        "display_name": "DeepLabV3+",
        "folder_candidates": ["DeepLabV3Plus"],
        "training_state": "trained",
        "prompt_mode": "No_prompt",
        "is_sam": False,
    },
    {
        "display_name": "SegFormer",
        "folder_candidates": ["SegFormer"],
        "training_state": "trained",
        "prompt_mode": "No_prompt",
        "is_sam": False,
    },
    {
        "display_name": "UNet++",
        "folder_candidates": ["UNetPP"],
        "training_state": "trained",
        "prompt_mode": "No_prompt",
        "is_sam": False,
    },
    {
        "display_name": "nnU-Net v2 2D",
        "folder_candidates": ["nnUNetV2_2D_100ep", "nnUNetV2_2D"],
        "training_state": "trained",
        "prompt_mode": "No_prompt",
        "is_sam": False,
    },
]


DPI = 600


NON_METRIC_COLUMNS = {
    "dataset",
    "split",
    "model_name",
    "training_state",
    "prompt_mode",
    "image_name",
    "mask_name",
    "case_id",
    "prediction_name",
    "merged_mask_name",
    "overlay_name",
    "source_image",
    "source_mask",
    "image_path",
    "mask_path",
    "pred_path",
    "gt_path",
    "num_prompt_instances",
    "inference_time_ms",
    "threshold",
}


EXCLUDE_METRIC_SUBSTRINGS = [
    "path",
    "name",
    "time",
    "prompt",
    "threshold",
    "pixel",
    "area",
    "component",
    "instance",
    "tp",
    "tn",
    "fp",
    "fn",
]


PREFERRED_METRIC_ORDER = [
    "dice",
    "iou",
    "precision",
    "recall",
    "specificity",
    "accuracy",
    "f1",
    "balanced_accuracy",
    "hd95",
    "assd",
]


def setup_matplotlib():
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 18,
            "legend.fontsize": 15,
            "figure.titlesize": 20,
        }
    )


def metric_display_name(metric_name: str):
    mapping = {
        "dice": "Dice score",
        "iou": "IoU score",
        "precision": "Precision",
        "recall": "Recall",
        "specificity": "Specificity",
        "accuracy": "Accuracy",
        "f1": "F1 score",
        "balanced_accuracy": "Balanced accuracy",
        "hd95": "HD95",
        "assd": "ASSD",
    }

    return mapping.get(metric_name.lower(), metric_name.replace("_", " ").title())


def safe_filename(text: str):
    return (
        text.replace(" ", "_")
        .replace("+", "Plus")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("-", "_")
        .replace("__", "_")
    )


def find_metrics_file_for_model(dataset_key: str, split_key: str, model_spec: dict):
    for folder_name in model_spec["folder_candidates"]:
        model_root = (
            RESULTS_ROOT
            / dataset_key
            / folder_name
            / model_spec["training_state"]
        )

        if not model_root.exists():
            continue

        if model_spec["is_sam"]:
            if model_spec["prompt_mode"] is not None:
                prompt_candidates = [model_spec["prompt_mode"]]
            else:
                prompt_candidates = SAM_PROMPT_MODE_CANDIDATES

            for prompt_mode in prompt_candidates:
                candidate = (
                    model_root
                    / prompt_mode
                    / split_key
                    / "metrics_image_level.csv"
                )

                if candidate.exists():
                    return candidate, prompt_mode, folder_name

            fallback_candidates = sorted(
                model_root.glob(f"*/{split_key}/metrics_image_level.csv")
            )

            if fallback_candidates:
                path = fallback_candidates[0]
                prompt_mode = path.parents[1].name
                return path, prompt_mode, folder_name

        else:
            prompt_mode = model_spec["prompt_mode"]

            candidate = (
                model_root
                / prompt_mode
                / split_key
                / "metrics_image_level.csv"
            )

            if candidate.exists():
                return candidate, prompt_mode, folder_name

            fallback_candidates = sorted(
                model_root.glob(f"*/{split_key}/metrics_image_level.csv")
            )

            if fallback_candidates:
                path = fallback_candidates[0]
                prompt_mode = path.parents[1].name
                return path, prompt_mode, folder_name

    return None, None, None


def load_all_metrics_for_dataset_split(dataset_key: str, split_key: str):
    all_rows = []
    loaded_info = []

    for model_spec in MODEL_SPECS_TOP_TO_BOTTOM:
        display_name = model_spec["display_name"]

        metrics_path, prompt_mode, folder_name = find_metrics_file_for_model(
            dataset_key=dataset_key,
            split_key=split_key,
            model_spec=model_spec,
        )

        if metrics_path is None:
            print(f"WARNING: Missing metrics for {dataset_key} | {split_key} | {display_name}")
            continue

        df = pd.read_csv(metrics_path)

        df["comparison_model_name"] = display_name
        df["comparison_model_folder"] = folder_name
        df["comparison_prompt_mode"] = prompt_mode
        df["comparison_metrics_path"] = str(metrics_path)

        all_rows.append(df)

        loaded_info.append(
            {
                "dataset": dataset_key,
                "split": split_key,
                "display_name": display_name,
                "folder_name": folder_name,
                "prompt_mode": prompt_mode,
                "metrics_path": str(metrics_path),
                "num_rows": len(df),
            }
        )

        print(
            f"Loaded {dataset_key} | {split_key} | {display_name} "
            f"| prompt={prompt_mode} | rows={len(df)}"
        )

    if len(all_rows) == 0:
        return None, pd.DataFrame(loaded_info)

    combined_df = pd.concat(all_rows, ignore_index=True)

    return combined_df, pd.DataFrame(loaded_info)


def is_metric_column(column_name: str, series: pd.Series):
    lower = column_name.lower()

    if lower in NON_METRIC_COLUMNS:
        return False

    if lower.startswith("comparison_"):
        return False

    for substring in EXCLUDE_METRIC_SUBSTRINGS:
        if substring in lower:
            return False

    numeric_series = pd.to_numeric(series, errors="coerce")

    if numeric_series.notna().sum() == 0:
        return False

    return True


def get_metric_columns(df: pd.DataFrame):
    metric_columns = []

    for column in df.columns:
        if is_metric_column(column, df[column]):
            metric_columns.append(column)

    lower_map = {col.lower(): col for col in metric_columns}

    ordered = []

    for preferred in PREFERRED_METRIC_ORDER:
        if preferred in lower_map:
            ordered.append(lower_map[preferred])

    remaining = [
        col for col in metric_columns
        if col not in ordered
    ]

    ordered.extend(sorted(remaining))

    return ordered


def is_unit_interval_metric(metric_name: str):
    lower = metric_name.lower()

    unit_metric_keywords = [
        "dice",
        "iou",
        "precision",
        "recall",
        "specificity",
        "accuracy",
        "f1",
        "balanced_accuracy",
    ]

    return any(keyword in lower for keyword in unit_metric_keywords)


def get_clean_metric_values(
    combined_df: pd.DataFrame,
    model_name: str,
    metric_name: str,
    dataset_key: str,
    split_key: str,
):
    values = pd.to_numeric(
        combined_df.loc[
            combined_df["comparison_model_name"] == model_name,
            metric_name,
        ],
        errors="coerce",
    ).dropna()

    if is_unit_interval_metric(metric_name):
        invalid_values = values[(values < 0.0) | (values > 1.0)]

        if len(invalid_values) > 0:
            print(
                f"WARNING: {dataset_key} | {split_key} | {model_name} | {metric_name} "
                f"has {len(invalid_values)} values outside [0, 1]. "
                f"These are clipped for plotting only."
            )

        values = values.clip(lower=0.0, upper=1.0)

    return values


def plot_metric_boxplot(
    combined_df: pd.DataFrame,
    dataset_key: str,
    dataset_title: str,
    split_key: str,
    split_title: str,
    metric_name: str,
):
    model_names_available = set(combined_df["comparison_model_name"].unique())

    ordered_model_names = [
        spec["display_name"]
        for spec in MODEL_SPECS_TOP_TO_BOTTOM
        if spec["display_name"] in model_names_available
    ]

    data = []
    labels = []
    colors = []
    means = []
    medians = []

    for model_name in ordered_model_names:
        values = get_clean_metric_values(
            combined_df=combined_df,
            model_name=model_name,
            metric_name=metric_name,
            dataset_key=dataset_key,
            split_key=split_key,
        )

        if len(values) == 0:
            continue

        data.append(values.to_numpy())
        labels.append(model_name)
        colors.append(MODEL_COLORS.get(model_name, "#333333"))
        means.append(float(values.mean()))
        medians.append(float(values.median()))

    if len(data) == 0:
        print(f"WARNING: No data for {dataset_key} | {split_key} | {metric_name}")
        return

    n_models = len(data)

    positions = list(range(n_models, 0, -1))

    fig_height = max(8, 0.72 * n_models + 2.5)
    fig, axis = plt.subplots(figsize=(13.5, fig_height))

    boxplot = axis.boxplot(
        data,
        vert=False,
        positions=positions,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        showfliers=False,
        widths=0.55,
        medianprops={
            "color": "black",
            "linewidth": 2.4,
        },
        meanprops={
            "color": "red",
            "linewidth": 2.4,
            "linestyle": "-",
        },
        whiskerprops={
            "color": "black",
            "linewidth": 1.5,
        },
        capprops={
            "color": "black",
            "linewidth": 1.5,
        },
    )

    for patch, color in zip(boxplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.4)

    axis.set_yticks(positions)
    axis.set_yticklabels(labels, fontsize=18)

    display_metric = metric_display_name(metric_name)

    axis.set_xlabel(display_metric, fontsize=18)
    axis.set_title(f"{dataset_title} - {split_title} set", fontsize=20)

    axis.tick_params(axis="x", labelsize=14)
    axis.tick_params(axis="y", labelsize=18)

    axis.grid(True, axis="x", alpha=0.3)

    if is_unit_interval_metric(metric_name):
        axis.set_xlim(-0.05, 1.0)
        axis.set_xticks(np.arange(0.0, 1.01, 0.1))

    median_handle = Line2D(
        [0],
        [0],
        color="black",
        linewidth=2.4,
        label="Median",
    )

    mean_handle = Line2D(
        [0],
        [0],
        color="red",
        linewidth=2.4,
        label="Mean",
    )

    axis.legend(
        handles=[median_handle, mean_handle],
        loc="best",
        frameon=True,
        fontsize=15,
    )

    fig.tight_layout()

    output_dir = OUTPUT_ROOT / dataset_key / split_key
    output_dir.mkdir(parents=True, exist_ok=True)

    output_name = f"{dataset_key}_{split_key}_{safe_filename(metric_name)}_models.png"
    output_path = output_dir / output_name

    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")

    summary_df = pd.DataFrame(
        {
            "model": labels,
            "mean": means,
            "median": medians,
        }
    )

    summary_path = output_dir / f"{dataset_key}_{split_key}_{safe_filename(metric_name)}_summary.csv"
    summary_df.to_csv(summary_path, index=False)


def save_loaded_info(loaded_info_df: pd.DataFrame, dataset_key: str, split_key: str):
    output_dir = OUTPUT_ROOT / dataset_key / split_key
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{dataset_key}_{split_key}_loaded_metric_files.csv"
    loaded_info_df.to_csv(output_path, index=False)

    print(f"Saved loaded-file report: {output_path}")


def save_combined_metrics(combined_df: pd.DataFrame, dataset_key: str, split_key: str):
    output_dir = OUTPUT_ROOT / dataset_key / split_key
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{dataset_key}_{split_key}_combined_image_level_metrics.csv"
    combined_df.to_csv(output_path, index=False)

    print(f"Saved combined metrics: {output_path}")


def process_dataset_split(dataset_cfg: dict, split_cfg: dict):
    dataset_key = dataset_cfg["key"]
    dataset_title = dataset_cfg["title"]

    split_key = split_cfg["key"]
    split_title = split_cfg["title"]

    print("\n" + "=" * 100)
    print(f"Processing {dataset_key} | {split_key}")
    print("=" * 100)

    combined_df, loaded_info_df = load_all_metrics_for_dataset_split(
        dataset_key=dataset_key,
        split_key=split_key,
    )

    save_loaded_info(
        loaded_info_df=loaded_info_df,
        dataset_key=dataset_key,
        split_key=split_key,
    )

    if combined_df is None or combined_df.empty:
        print(f"WARNING: No metrics found for {dataset_key} | {split_key}")
        return

    save_combined_metrics(
        combined_df=combined_df,
        dataset_key=dataset_key,
        split_key=split_key,
    )

    metric_columns = get_metric_columns(combined_df)

    if len(metric_columns) == 0:
        print(f"WARNING: No numeric metric columns found for {dataset_key} | {split_key}")
        return

    print(f"Metric columns to plot: {metric_columns}")

    for metric_name in metric_columns:
        plot_metric_boxplot(
            combined_df=combined_df,
            dataset_key=dataset_key,
            dataset_title=dataset_title,
            split_key=split_key,
            split_title=split_title,
            metric_name=metric_name,
        )


def main():
    setup_matplotlib()

    for dataset_cfg in DATASETS:
        for split_cfg in SPLITS:
            process_dataset_split(dataset_cfg, split_cfg)


if __name__ == "__main__":
    main()