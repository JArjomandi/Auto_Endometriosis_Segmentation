from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_ROOT = Path(r"F:\Results\SAM_Benchmarking")
OUTPUT_ROOT = RESULTS_ROOT / "Model_comparison" / "training_loss_plots"

NNUNET_RESULTS_ROOT = RESULTS_ROOT / "nnUNet" / "nnUNet_results"


DATASETS = [
    {
        "key": "ENID",
        "title": "ENID dataset",
        "nnunet_dataset_folder": "Dataset501_ENID",
    },
    {
        "key": "GLENDA",
        "title": "GLENDA dataset",
        "nnunet_dataset_folder": "Dataset502_GLENDA",
    },
    {
        "key": "GLENDA_clean",
        "title": "cleaned GLENDA dataset",
        "nnunet_dataset_folder": "Dataset503_GLENDA_clean",
    },
]


MODEL_COLORS = {
    "UNet++": "#1f77b4",
    "DeepLabV3+": "#ff7f0e",
    "SegFormer": "#2ca02c",
    "YOLO11s-seg": "#9467bd",
    "nnU-Net v2 2D": "#7f7f7f",
}


SEMANTIC_LOSS_MODELS = [
    {
        "display_name": "UNet++",
        "folder_name": "UNetPP",
        "type": "standard_csv",
    },
    {
        "display_name": "DeepLabV3+",
        "folder_name": "DeepLabV3Plus",
        "type": "standard_csv",
    },
    {
        "display_name": "SegFormer",
        "folder_name": "SegFormer",
        "type": "standard_csv",
    },
    {
        "display_name": "nnU-Net v2 2D",
        "folder_name": "nnUNetV2_2D_100ep",
        "type": "nnunet_log",
    },
]


ALL_TRAINABLE_MODELS = [
    {
        "display_name": "UNet++",
        "folder_name": "UNetPP",
        "type": "standard_csv",
    },
    {
        "display_name": "DeepLabV3+",
        "folder_name": "DeepLabV3Plus",
        "type": "standard_csv",
    },
    {
        "display_name": "SegFormer",
        "folder_name": "SegFormer",
        "type": "standard_csv",
    },
    {
        "display_name": "YOLO11s-seg",
        "folder_name": "YOLO11s_seg",
        "type": "yolo_csv",
    },
    {
        "display_name": "nnU-Net v2 2D",
        "folder_name": "nnUNetV2_2D_100ep",
        "type": "nnunet_log",
    },
]


MAX_EPOCHS = 100
DPI = 600


def setup_matplotlib():
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "figure.titlesize": 18,
        }
    )


def find_existing_file(candidates):
    for path in candidates:
        if path.exists():
            return path

    return None


def normalize_epoch_column(df: pd.DataFrame) -> pd.DataFrame:
    lower_to_original = {col.lower().strip(): col for col in df.columns}

    if "epoch" in lower_to_original:
        epoch_col = lower_to_original["epoch"]
        df["epoch_for_plot"] = pd.to_numeric(df[epoch_col], errors="coerce")
    else:
        df["epoch_for_plot"] = range(1, len(df) + 1)

    if df["epoch_for_plot"].min() == 0:
        df["epoch_for_plot"] = df["epoch_for_plot"] + 1

    return df


def load_standard_training_history(dataset_key: str, model_folder: str):
    candidates = [
        RESULTS_ROOT / dataset_key / model_folder / "trained" / "training_history.csv",
        RESULTS_ROOT / dataset_key / model_folder / "trained" / "training_history.xlsx",
    ]

    history_path = find_existing_file(candidates)

    if history_path is None:
        print(f"WARNING: No training history found for {dataset_key} | {model_folder}")
        return None

    if history_path.suffix.lower() == ".xlsx":
        df = pd.read_excel(history_path)
    else:
        df = pd.read_csv(history_path)

    df = normalize_epoch_column(df)

    lower_to_original = {col.lower().strip(): col for col in df.columns}

    train_col = lower_to_original.get("train_loss")
    val_col = lower_to_original.get("val_loss")

    if train_col is None or val_col is None:
        print(
            f"WARNING: Missing train_loss/val_loss in {history_path}. "
            f"Columns: {list(df.columns)}"
        )
        return None

    out = pd.DataFrame(
        {
            "epoch": df["epoch_for_plot"],
            "train_loss": pd.to_numeric(df[train_col], errors="coerce"),
            "val_loss": pd.to_numeric(df[val_col], errors="coerce"),
        }
    )

    out = out.dropna(subset=["epoch"])
    out = out[out["epoch"] <= MAX_EPOCHS]

    return out


def load_yolo_training_history(dataset_key: str, model_folder: str):
    candidates = [
        RESULTS_ROOT / dataset_key / model_folder / "trained" / "training_history.csv",
        RESULTS_ROOT / dataset_key / model_folder / "trained" / "training_history.xlsx",
        RESULTS_ROOT / dataset_key / model_folder / "trained" / "ultralytics" / "train" / "results.csv",
    ]

    history_path = find_existing_file(candidates)

    if history_path is None:
        print(f"WARNING: No YOLO training history found for {dataset_key} | {model_folder}")
        return None

    if history_path.suffix.lower() == ".xlsx":
        df = pd.read_excel(history_path)
    else:
        df = pd.read_csv(history_path)

    df.columns = [str(col).strip() for col in df.columns]
    df = normalize_epoch_column(df)

    train_loss_cols = [
        col for col in df.columns
        if col.startswith("train/") and col.endswith("_loss")
    ]

    val_loss_cols = [
        col for col in df.columns
        if col.startswith("val/") and col.endswith("_loss")
    ]

    if len(train_loss_cols) == 0 or len(val_loss_cols) == 0:
        print(
            f"WARNING: Could not find YOLO train/val loss columns in {history_path}. "
            f"Columns: {list(df.columns)}"
        )
        return None

    train_loss = df[train_loss_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    val_loss = df[val_loss_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)

    out = pd.DataFrame(
        {
            "epoch": df["epoch_for_plot"],
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
    )

    out = out.dropna(subset=["epoch"])
    out = out[out["epoch"] <= MAX_EPOCHS]

    print(f"YOLO loss columns from {history_path}:")
    print(f"  train loss columns: {train_loss_cols}")
    print(f"  val loss columns:   {val_loss_cols}")

    return out


def parse_float_from_line(line: str):
    matches = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line)

    if not matches:
        return None

    return float(matches[-1])


def load_nnunet_training_log(nnunet_dataset_folder: str):
    dataset_results_dir = NNUNET_RESULTS_ROOT / nnunet_dataset_folder

    if not dataset_results_dir.exists():
        print(f"WARNING: nnU-Net results folder not found: {dataset_results_dir}")
        return None

    log_paths = sorted(
        dataset_results_dir.rglob("training_log*.txt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not log_paths:
        print(f"WARNING: No nnU-Net training_log*.txt found in {dataset_results_dir}")
        return None

    log_path = log_paths[0]
    print(f"Using nnU-Net log: {log_path}")

    rows = []
    current_epoch = None
    current_train_loss = None
    current_val_loss = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            stripped = line.strip()
            lower = stripped.lower()

            epoch_match = re.search(r"epoch\s+(\d+)", stripped, flags=re.IGNORECASE)

            if epoch_match:
                if current_epoch is not None:
                    rows.append(
                        {
                            "epoch": current_epoch,
                            "train_loss": current_train_loss,
                            "val_loss": current_val_loss,
                        }
                    )

                current_epoch = int(epoch_match.group(1)) + 1
                current_train_loss = None
                current_val_loss = None
                continue

            if "train_loss" in lower or "training loss" in lower:
                current_train_loss = parse_float_from_line(stripped)

            if "val_loss" in lower or "validation loss" in lower:
                current_val_loss = parse_float_from_line(stripped)

    if current_epoch is not None:
        rows.append(
            {
                "epoch": current_epoch,
                "train_loss": current_train_loss,
                "val_loss": current_val_loss,
            }
        )

    if not rows:
        print(f"WARNING: Could not parse nnU-Net losses from {log_path}")
        return None

    df = pd.DataFrame(rows)
    df = df[df["epoch"] <= MAX_EPOCHS]

    if df["train_loss"].isna().all() and df["val_loss"].isna().all():
        print(f"WARNING: Parsed nnU-Net log but no losses were found: {log_path}")
        return None

    return df


def load_model_losses(dataset_cfg: dict, model_cfg: dict):
    dataset_key = dataset_cfg["key"]

    if model_cfg["type"] == "standard_csv":
        return load_standard_training_history(
            dataset_key=dataset_key,
            model_folder=model_cfg["folder_name"],
        )

    if model_cfg["type"] == "yolo_csv":
        return load_yolo_training_history(
            dataset_key=dataset_key,
            model_folder=model_cfg["folder_name"],
        )

    if model_cfg["type"] == "nnunet_log":
        return load_nnunet_training_log(
            nnunet_dataset_folder=dataset_cfg["nnunet_dataset_folder"],
        )

    raise ValueError(f"Unknown model type: {model_cfg['type']}")


def prepare_for_log_scale(df: pd.DataFrame, loss_column: str):
    out = df[["epoch", loss_column]].copy()
    out[loss_column] = pd.to_numeric(out[loss_column], errors="coerce")
    out = out.replace([float("inf"), float("-inf")], pd.NA)
    out = out.dropna(subset=[loss_column])
    out = out[out[loss_column] > 0]

    return out


def plot_dataset_losses(
    dataset_cfg: dict,
    model_list: list,
    output_suffix: str,
    use_log_scale: bool,
    figure_note: str,
):
    dataset_key = dataset_cfg["key"]
    dataset_title = dataset_cfg["title"]

    fig, axes = plt.subplots(1, 2, figsize=(19, 7.5), sharex=True)

    plotted_any = False

    for model_cfg in model_list:
        display_name = model_cfg["display_name"]
        color = MODEL_COLORS.get(display_name, "#333333")

        loss_df = load_model_losses(dataset_cfg, model_cfg)

        if loss_df is None or loss_df.empty:
            continue

        if use_log_scale:
            train_df = prepare_for_log_scale(loss_df, "train_loss")
            val_df = prepare_for_log_scale(loss_df, "val_loss")
        else:
            train_df = loss_df[["epoch", "train_loss"]].copy()
            val_df = loss_df[["epoch", "val_loss"]].copy()

        if not train_df.empty and not train_df["train_loss"].isna().all():
            axes[0].plot(
                train_df["epoch"],
                train_df["train_loss"],
                label=display_name,
                color=color,
                linewidth=2.4,
            )
            plotted_any = True

        if not val_df.empty and not val_df["val_loss"].isna().all():
            axes[1].plot(
                val_df["epoch"],
                val_df["val_loss"],
                label=display_name,
                color=color,
                linewidth=2.4,
            )
            plotted_any = True

    axes[0].set_title(f"Training losses on {dataset_title}")
    axes[1].set_title(f"Validation losses on {dataset_title}")

    axes[0].set_xlabel("Epoch")
    axes[1].set_xlabel("Epoch")

    axes[0].set_ylabel("Training loss")
    axes[1].set_ylabel("Validation loss")

    if use_log_scale:
        axes[0].set_yscale("log")
        axes[1].set_yscale("log")
        axes[0].set_ylabel("Training loss, log scale")
        axes[1].set_ylabel("Validation loss, log scale")

    for axis in axes:
        axis.set_xlim(1, MAX_EPOCHS)
        axis.grid(True, alpha=0.3, which="both")
        axis.legend(loc="best", frameon=True)

    fig.text(
        0.5,
        0.01,
        figure_note,
        ha="center",
        va="bottom",
        fontsize=14,
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_ROOT / f"{dataset_key}_training_validation_losses_{output_suffix}.png"

    if plotted_any:
        fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        print(f"WARNING: Nothing plotted for {dataset_key} | {output_suffix}")

    plt.close(fig)


def main():
    setup_matplotlib()

    for dataset_cfg in DATASETS:
        plot_dataset_losses(
            dataset_cfg=dataset_cfg,
            model_list=SEMANTIC_LOSS_MODELS,
            output_suffix="semantic_only",
            use_log_scale=False,
            # figure_note=(
            #     "YOLO11s-seg is excluded because its multi-component detection-segmentation "
            #     "loss is not directly comparable to Dice/BCE-style semantic segmentation losses."
            # ),
            figure_note=(
                " "
            ),
        )

        plot_dataset_losses(
            dataset_cfg=dataset_cfg,
            model_list=ALL_TRAINABLE_MODELS,
            output_suffix="all_models_log_scale",
            use_log_scale=True,
            # figure_note=(
            #     "Log-scale view including YOLO11s-seg. Raw loss magnitudes are not directly "
            #     "comparable across different loss formulations."
            # ),
            figure_note=(
                " "
            ),
        )


if __name__ == "__main__":
    main()