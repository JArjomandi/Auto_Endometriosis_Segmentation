from pathlib import Path
import argparse

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Metric visualization for prompt-comparison experiments
# ============================================================

DEFAULT_RESULTS_ROOT = Path(r"F:\Results\SAM_Benchmarking")

PROMPT_ORDER = [
    "GT_point",
    "GT_box",
    "GT_box_point",
    "GT_box_posneg",
]

PROMPT_LABELS = {
    "GT_point": "GT point",
    "GT_box": "GT box",
    "GT_box_point": "GT box + point",
    "GT_box_posneg": "GT box + point + neg points",
}

METRICS_TO_PLOT = [
    "dice",
    "iou",
    "precision",
    "recall",
    "specificity",
    "false_positive_area_px",
    "false_negative_area_px",
    "gt_area_px",
    "pred_area_px",
]

RATIO_METRICS = {
    "dice",
    "iou",
    "precision",
    "recall",
    "specificity",
}


def metric_display_name(metric: str) -> str:
    names = {
        "dice": "Dice score",
        "iou": "IoU",
        "precision": "Precision",
        "recall": "Recall / sensitivity",
        "specificity": "Specificity",
        "false_positive_area_px": "False-positive area (pixels)",
        "false_negative_area_px": "False-negative area (pixels)",
        "gt_area_px": "Ground-truth area (pixels)",
        "pred_area_px": "Predicted area (pixels)",
    }
    return names.get(metric, metric)


def collect_metrics(
    results_root: Path,
    dataset_name: str,
    model_name: str,
    training_state: str,
    split: str,
) -> pd.DataFrame:
    """
    Reads CSVs like:

    F:/Results/SAM_Benchmarking/ENID/SAM2/frozen/GT_box/val/metrics_image_level.csv
    """

    base_dir = results_root / dataset_name / model_name / training_state

    if not base_dir.exists():
        raise FileNotFoundError(f"Results folder not found: {base_dir}")

    dfs = []

    for prompt_mode in PROMPT_ORDER:
        csv_path = base_dir / prompt_mode / split / "metrics_image_level.csv"

        if not csv_path.exists():
            print(f"WARNING: missing metrics CSV, skipping: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        df["dataset"] = dataset_name
        df["model_name"] = model_name
        df["training_state"] = training_state
        df["split"] = split
        df["prompt_mode"] = prompt_mode
        df["prompt_label"] = PROMPT_LABELS.get(prompt_mode, prompt_mode)
        df["source_csv"] = str(csv_path)

        dfs.append(df)

    if not dfs:
        raise RuntimeError(
            f"No metrics CSVs found for {dataset_name}/{model_name}/{training_state}/{split}"
        )

    return pd.concat(dfs, ignore_index=True)


def make_prompt_comparison_boxplot(
    df: pd.DataFrame,
    metric: str,
    output_dir: Path,
    dataset_name: str,
    model_name: str,
    training_state: str,
    split: str,
) -> Path:
    """
    Saves one boxplot comparing prompt modes for one metric and one split.
    Median is the default orange/black line.
    Mean is drawn as a dashed line across each box.
    """

    available_prompt_modes = [
        prompt for prompt in PROMPT_ORDER
        if prompt in set(df["prompt_mode"].unique())
    ]

    if not available_prompt_modes:
        raise RuntimeError("No prompt modes available for plotting.")

    data = []
    labels = []

    for prompt_mode in available_prompt_modes:
        values = df.loc[df["prompt_mode"] == prompt_mode, metric].dropna().values
        if len(values) == 0:
            continue

        data.append(values)
        labels.append(PROMPT_LABELS.get(prompt_mode, prompt_mode))

    if not data:
        raise RuntimeError(f"No valid data found for metric: {metric}")

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))

    box = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        widths=0.6,
        medianprops={
            "linewidth": 2.2,
        },
        meanprops={
            "linestyle": "--",
            "linewidth": 2.0,
        },
        boxprops={
            "linewidth": 1.5,
        },
        whiskerprops={
            "linewidth": 1.3,
        },
        capprops={
            "linewidth": 1.3,
        },
        flierprops={
            "marker": "o",
            "markersize": 3,
            "alpha": 0.5,
        },
    )

    # Light gray fill for all boxes.
    for patch in box["boxes"]:
        patch.set_facecolor("#EAEAEA")

    # Add individual sample points.
    for i, values in enumerate(data, start=1):
        x_positions = [i] * len(values)
        ax.scatter(
            x_positions,
            values,
            s=12,
            alpha=0.35,
            zorder=3,
        )

    # Add mean and median text.
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin

    for i, values in enumerate(data, start=1):
        mean_val = values.mean()
        median_val = pd.Series(values).median()

        text_y = mean_val + 0.03 * yrange
        if text_y > ymax:
            text_y = ymax - 0.05 * yrange

        ax.text(
            i,
            text_y,
            f"mean={mean_val:.3f}\nmedian={median_val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    title = (
        f"{dataset_name} | {model_name} {training_state} | "
        f"{split} | {metric_display_name(metric)}"
    )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Prompt mode", fontsize=12)
    ax.set_ylabel(metric_display_name(metric), fontsize=12)

    if metric in RATIO_METRICS:
        ax.set_ylim(0, 1.05)

    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=15)

    fig.tight_layout()

    out_name = f"{dataset_name}_{metric}_{split}_prompt_compare.png"
    out_path = output_dir / out_name

    fig.savefig(out_path, dpi=500)
    plt.close(fig)

    return out_path


def visualize_dataset_model_split(
    results_root: Path,
    dataset_name: str,
    model_name: str,
    training_state: str,
    split: str,
):
    print("\n" + "=" * 100)
    print(f"Visualizing: {dataset_name} | {model_name} | {training_state} | {split}")
    print("=" * 100)

    df = collect_metrics(
        results_root=results_root,
        dataset_name=dataset_name,
        model_name=model_name,
        training_state=training_state,
        split=split,
    )

    output_dir = (
        results_root
        / dataset_name
        / model_name
        / training_state
        / "visualize_metrics"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    combined_csv = output_dir / f"{dataset_name}_{split}_combined_prompt_metrics.csv"
    df.to_csv(combined_csv, index=False)
    print(f"Saved combined CSV: {combined_csv}")

    for metric in METRICS_TO_PLOT:
        if metric not in df.columns:
            print(f"WARNING: metric not found, skipping: {metric}")
            continue

        out_path = make_prompt_comparison_boxplot(
            df=df,
            metric=metric,
            output_dir=output_dir,
            dataset_name=dataset_name,
            model_name=model_name,
            training_state=training_state,
            split=split,
        )

        print(f"Saved plot: {out_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results_root",
        type=str,
        default=str(DEFAULT_RESULTS_ROOT),
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ENID", "GLENDA"],
    )

    parser.add_argument(
        "--model",
        type=str,
        default="SAM2",
    )

    parser.add_argument(
        "--training_state",
        type=str,
        default="frozen",
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val", "test"],
    )

    args = parser.parse_args()

    results_root = Path(args.results_root)

    for dataset_name in args.datasets:
        for split in args.splits:
            visualize_dataset_model_split(
                results_root=results_root,
                dataset_name=dataset_name,
                model_name=args.model,
                training_state=args.training_state,
                split=split,
            )

    print("\nDone visualizing metrics.")


if __name__ == "__main__":
    main()