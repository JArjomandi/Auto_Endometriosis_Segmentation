from pathlib import Path
import argparse

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Generic metric visualization and summary tables
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

# Only these metrics are plotted.
# The Excel file will summarize all numeric columns it can find.
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
        "inference_time_ms": "Inference time (ms)",
        "sam_score": "SAM score",
        "selected_mask_index": "Selected mask index",
        "num_prompt_instances": "Number of prompt instances",
    }
    return names.get(metric, metric)


def collect_csvs(
    results_root: Path,
    dataset_name: str,
    model_name: str,
    training_state: str,
    split: str,
):
    """
    Reads the prompt-specific result files:

    F:/Results/SAM_Benchmarking/ENID/SAM2/frozen/GT_box/val/metrics_image_level.csv
    F:/Results/SAM_Benchmarking/ENID/SAM2/frozen/GT_box/val/inference_results.csv

    Returns:
      image_metrics_df
      inference_df
    """

    base_dir = results_root / dataset_name / model_name / training_state

    if not base_dir.exists():
        raise FileNotFoundError(f"Results folder not found: {base_dir}")

    image_metric_dfs = []
    inference_dfs = []

    for prompt_mode in PROMPT_ORDER:
        prompt_dir = base_dir / prompt_mode / split

        metrics_csv = prompt_dir / "metrics_image_level.csv"
        inference_csv = prompt_dir / "inference_results.csv"

        if metrics_csv.exists():
            df = pd.read_csv(metrics_csv)
            df["dataset"] = dataset_name
            df["model_name"] = model_name
            df["training_state"] = training_state
            df["split"] = split
            df["prompt_mode"] = prompt_mode
            df["prompt_label"] = PROMPT_LABELS.get(prompt_mode, prompt_mode)
            df["source_table"] = "metrics_image_level"
            df["source_csv"] = str(metrics_csv)
            image_metric_dfs.append(df)
        else:
            print(f"WARNING: missing metrics CSV, skipping: {metrics_csv}")

        if inference_csv.exists():
            df = pd.read_csv(inference_csv)
            df["dataset"] = dataset_name
            df["model_name"] = model_name
            df["training_state"] = training_state
            df["split"] = split
            df["prompt_mode"] = prompt_mode
            df["prompt_label"] = PROMPT_LABELS.get(prompt_mode, prompt_mode)
            df["source_table"] = "inference_results"
            df["source_csv"] = str(inference_csv)
            inference_dfs.append(df)
        else:
            print(f"WARNING: missing inference CSV, skipping: {inference_csv}")

    if not image_metric_dfs:
        raise RuntimeError(
            f"No metrics_image_level.csv files found for "
            f"{dataset_name}/{model_name}/{training_state}/{split}"
        )

    image_metrics_df = pd.concat(image_metric_dfs, ignore_index=True)

    if inference_dfs:
        inference_df = pd.concat(inference_dfs, ignore_index=True)
    else:
        inference_df = pd.DataFrame()

    return image_metrics_df, inference_df


def make_prompt_comparison_boxplot(
    df: pd.DataFrame,
    metric: str,
    output_dir: Path,
    dataset_name: str,
    model_name: str,
    training_state: str,
    split: str,
) -> Path:
    available_prompt_modes = [
        prompt for prompt in PROMPT_ORDER
        if prompt in set(df["prompt_mode"].unique())
    ]

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

    fig, ax = plt.subplots(figsize=(9, 5.5))

    box = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        widths=0.35,
        medianprops={
            "color": "orange",
            "linewidth": 2.5,
        },
        meanprops={
            "color": "red",
            "linestyle": "-",
            "linewidth": 2.2,
        },
        boxprops={
            "linewidth": 1.2,
            "color": "black",
        },
        whiskerprops={
            "linewidth": 1.1,
            "color": "black",
        },
        capprops={
            "linewidth": 1.1,
            "color": "black",
        },
        flierprops={
            "marker": "o",
            "markersize": 3,
            "alpha": 0.45,
            "markerfacecolor": "black",
            "markeredgecolor": "black",
        },
    )

    for patch in box["boxes"]:
        patch.set_facecolor("#BFDDF2")  # light blue

    title = (
        f"{dataset_name} | {model_name} {training_state} | "
        f"{split} | {metric_display_name(metric)}"
    )

    ax.set_title(title, fontsize=17, fontweight="bold")
    ax.set_xlabel("Prompt mode", fontsize=15)
    ax.set_ylabel(metric_display_name(metric), fontsize=15)

    ax.tick_params(axis="x", labelsize=13, rotation=15)
    ax.tick_params(axis="y", labelsize=13)

    if metric in RATIO_METRICS:
        ax.set_ylim(0, 1.05)

    ax.grid(axis="y", alpha=0.25)

    # Simple legend for line meanings.
    ax.plot([], [], color="red", linewidth=2.2, label="Mean")
    ax.plot([], [], color="orange", linewidth=2.5, label="Median")
    ax.legend(fontsize=12, frameon=False, loc="best")

    fig.tight_layout()

    out_name = f"{dataset_name}_{metric}_{split}_prompt_compare.png"
    out_path = output_dir / out_name

    fig.savefig(out_path, dpi=500)
    plt.close(fig)

    return out_path


def summarize_numeric_columns(df: pd.DataFrame, source_table_name: str) -> pd.DataFrame:
    """
    Generates a long-format summary table.

    Columns:
      parameter
      source_table
      prompt_mode
      n
      min
      max
      Q1
      Q3
      median
      mean

    This includes every numeric column found in the dataframe.
    """

    if df.empty:
        return pd.DataFrame()

    excluded_numeric = {
        # IDs are numeric but not meaningful as metrics.
        "lesion_id",
    }

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in excluded_numeric]

    rows = []

    for parameter in numeric_cols:
        for prompt_mode in PROMPT_ORDER:
            sub = df.loc[df["prompt_mode"] == prompt_mode, parameter].dropna()

            if len(sub) == 0:
                continue

            rows.append({
                "parameter": parameter,
                "parameter_display_name": metric_display_name(parameter),
                "source_table": source_table_name,
                "prompt_mode": prompt_mode,
                "prompt_label": PROMPT_LABELS.get(prompt_mode, prompt_mode),
                "n": int(len(sub)),
                "min": float(sub.min()),
                "max": float(sub.max()),
                "Q1": float(sub.quantile(0.25)),
                "Q3": float(sub.quantile(0.75)),
                "median": float(sub.median()),
                "mean": float(sub.mean()),
                "std": float(sub.std()) if len(sub) > 1 else 0.0,
            })

    summary_df = pd.DataFrame(rows)

    if not summary_df.empty:
        summary_df = summary_df[
            [
                "parameter",
                "parameter_display_name",
                "source_table",
                "prompt_mode",
                "prompt_label",
                "n",
                "min",
                "max",
                "Q1",
                "Q3",
                "median",
                "mean",
                "std",
            ]
        ]

    return summary_df


def save_excel_summary(
    image_metrics_df: pd.DataFrame,
    inference_df: pd.DataFrame,
    output_dir: Path,
    dataset_name: str,
    model_name: str,
    training_state: str,
    split: str,
) -> Path:
    image_summary = summarize_numeric_columns(
        image_metrics_df,
        source_table_name="metrics_image_level",
    )

    inference_summary = summarize_numeric_columns(
        inference_df,
        source_table_name="inference_results",
    )

    summary_all = pd.concat(
        [image_summary, inference_summary],
        ignore_index=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    out_xlsx = output_dir / f"{dataset_name}_{split}_summary_statistics.xlsx"

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        summary_all.to_excel(writer, sheet_name="summary_all", index=False)
        image_summary.to_excel(writer, sheet_name="image_level_summary", index=False)

        if not inference_summary.empty:
            inference_summary.to_excel(writer, sheet_name="inference_summary", index=False)

        image_metrics_df.to_excel(writer, sheet_name="image_level_raw", index=False)

        if not inference_df.empty:
            inference_df.to_excel(writer, sheet_name="inference_raw", index=False)

        # Basic column width formatting.
        workbook = writer.book

        for sheet_name in writer.sheets:
            ws = writer.sheets[sheet_name]

            for column_cells in ws.columns:
                max_length = 0
                column_letter = column_cells[0].column_letter

                for cell in column_cells:
                    try:
                        value_length = len(str(cell.value))
                        if value_length > max_length:
                            max_length = value_length
                    except Exception:
                        pass

                adjusted_width = min(max(max_length + 2, 10), 35)
                ws.column_dimensions[column_letter].width = adjusted_width

            ws.freeze_panes = "A2"

            # Header style.
            for cell in ws[1]:
                cell.font = cell.font.copy(bold=True)

    return out_xlsx


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

    image_metrics_df, inference_df = collect_csvs(
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
    image_metrics_df.to_csv(combined_csv, index=False)
    print(f"Saved combined image-level CSV: {combined_csv}")

    if not inference_df.empty:
        combined_inference_csv = output_dir / f"{dataset_name}_{split}_combined_inference_results.csv"
        inference_df.to_csv(combined_inference_csv, index=False)
        print(f"Saved combined inference CSV: {combined_inference_csv}")

    excel_path = save_excel_summary(
        image_metrics_df=image_metrics_df,
        inference_df=inference_df,
        output_dir=output_dir,
        dataset_name=dataset_name,
        model_name=model_name,
        training_state=training_state,
        split=split,
    )
    print(f"Saved Excel summary: {excel_path}")

    for metric in METRICS_TO_PLOT:
        if metric not in image_metrics_df.columns:
            print(f"WARNING: metric not found in image-level CSVs, skipping plot: {metric}")
            continue

        out_path = make_prompt_comparison_boxplot(
            df=image_metrics_df,
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