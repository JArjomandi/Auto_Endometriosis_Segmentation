from pathlib import Path
import shutil

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.datasets.segmentation_dataset import BinarySegmentationDataset
from src.models.segformer.segformer_model import build_segformer_model
from src.runners.run_segformer import (
    collect_probabilities_for_split,
    estimate_inference_time_per_image,
    evaluate_records_and_save,
    run_threshold_sweep,
    save_threshold_sweep,
)


def load_yaml(path):
    path = Path(path)

    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def copy_training_artifacts_if_available(source_output_root: Path, output_root: Path):
    output_root.mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        "training_history.csv",
        "training_history.xlsx",
    ]

    for file_name in files_to_copy:
        source_path = source_output_root / file_name
        target_path = output_root / file_name

        if source_path.exists():
            shutil.copy2(source_path, target_path)

    source_curves_dir = source_output_root / "training_curves"
    target_curves_dir = output_root / "training_curves"

    if source_curves_dir.exists():
        target_curves_dir.mkdir(parents=True, exist_ok=True)

        for source_file in source_curves_dir.glob("*"):
            if source_file.is_file():
                shutil.copy2(source_file, target_curves_dir / source_file.name)


def write_calibration_metadata(
    output_root: Path,
    source_checkpoint: Path,
    final_threshold: float,
    threshold_df: pd.DataFrame,
    choose_metric: str,
):
    output_root.mkdir(parents=True, exist_ok=True)

    metadata_path = output_root / "calibration_metadata.csv"

    best_row = threshold_df.sort_values(
        by=[choose_metric, "dice"],
        ascending=False,
    ).iloc[0]

    metadata_df = pd.DataFrame(
        [
            {
                "source_checkpoint": str(source_checkpoint),
                "selection_split": "val",
                "choose_metric": choose_metric,
                "selected_threshold": final_threshold,
                "selected_val_dice": float(best_row["dice"]),
                "selected_val_iou": float(best_row["iou"]),
                "selected_val_precision": float(best_row["precision"]),
                "selected_val_recall": float(best_row["recall"]),
            }
        ]
    )

    metadata_df.to_csv(metadata_path, index=False)

    print(f"Saved calibration metadata: {metadata_path}")


def evaluate_calibrated_checkpoint(experiment_config_path):
    experiment_config_path = Path(experiment_config_path)
    project_root = Path(__file__).resolve().parents[2]

    exp_cfg = load_yaml(experiment_config_path)

    dataset_cfg = load_yaml(project_root / exp_cfg["dataset_config"])
    model_cfg = load_yaml(project_root / exp_cfg["model_config"])

    dataset_name = dataset_cfg["dataset_name"]
    dataset_root = Path(dataset_cfg["dataset_root"])

    source_checkpoint = Path(exp_cfg["source_checkpoint"])
    source_output_root = Path(exp_cfg["source_output_root"])
    output_root = Path(exp_cfg["output_root"])

    output_root.mkdir(parents=True, exist_ok=True)

    if not source_checkpoint.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {source_checkpoint}")

    device = torch.device(
        model_cfg.get("device", "cuda")
        if torch.cuda.is_available()
        else "cpu"
    )

    image_size = int(model_cfg.get("image_size", 512))
    batch_size = int(model_cfg.get("batch_size", 4))
    num_workers = int(model_cfg.get("num_workers", 4))
    postprocessing_cfg = model_cfg.get("postprocessing", {})

    val_split = exp_cfg["splits"]["val"]
    test_split = exp_cfg["splits"]["test"]

    val_cfg = dataset_cfg["splits"][val_split]
    test_cfg = dataset_cfg["splits"][test_split]

    val_dataset = BinarySegmentationDataset(
        images_dir=dataset_root / val_cfg["images"],
        masks_dir=dataset_root / val_cfg["masks"],
        image_size=image_size,
        augment=False,
    )

    test_dataset = BinarySegmentationDataset(
        images_dir=dataset_root / test_cfg["images"],
        masks_dir=dataset_root / test_cfg["masks"],
        image_size=image_size,
        augment=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = build_segformer_model(
        pretrained_model_name=model_cfg.get(
            "pretrained_model_name",
            "nvidia/segformer-b2-finetuned-ade-512-512",
        ),
        num_labels=int(model_cfg.get("num_labels", 1)),
    )

    checkpoint = torch.load(
        source_checkpoint,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    model_name = "SegFormer_calibrated"

    print("=" * 100)
    print(f"Evaluating calibrated SegFormer on {dataset_name}")
    print(f"Dataset root:       {dataset_root}")
    print(f"Source checkpoint:  {source_checkpoint}")
    print(f"Output root:        {output_root}")
    print(f"Device:             {device}")
    print(f"Val images:         {len(val_dataset)}")
    print(f"Test images:        {len(test_dataset)}")
    print("=" * 100)

    copy_training_artifacts_if_available(
        source_output_root=source_output_root,
        output_root=output_root,
    )

    threshold_sweep_cfg = exp_cfg.get("threshold_sweep", {})
    threshold_values = threshold_sweep_cfg.get(
        "values",
        [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
    )

    choose_metric = threshold_sweep_cfg.get("choose_metric", "dice")

    val_records = collect_probabilities_for_split(
        model=model,
        dataloader=val_loader,
        dataset_root=dataset_root,
        split_cfg=val_cfg,
        device=device,
    )

    threshold_df = run_threshold_sweep(
        records=val_records,
        thresholds=threshold_values,
        postprocessing_cfg=postprocessing_cfg,
    )

    save_threshold_sweep(threshold_df, output_root)

    if choose_metric not in threshold_df.columns:
        raise ValueError(f"choose_metric not found in threshold sweep: {choose_metric}")

    best_row = threshold_df.sort_values(
        by=[choose_metric, "dice"],
        ascending=False,
    ).iloc[0]

    final_threshold = float(best_row["threshold"])

    write_calibration_metadata(
        output_root=output_root,
        source_checkpoint=source_checkpoint,
        final_threshold=final_threshold,
        threshold_df=threshold_df,
        choose_metric=choose_metric,
    )

    print("=" * 100)
    print(f"Selected validation threshold by {choose_metric}: {final_threshold:.2f}")
    print("Final calibrated evaluation with saved masks and overlays")
    print("=" * 100)

    save_cfg = exp_cfg.get("save", {})

    val_inference_time = estimate_inference_time_per_image(
        model=model,
        dataloader=val_loader,
        device=device,
    )

    test_inference_time = estimate_inference_time_per_image(
        model=model,
        dataloader=test_loader,
        device=device,
    )

    evaluate_records_and_save(
        records=val_records,
        dataset_name=dataset_name,
        model_name=model_name,
        output_root=output_root,
        split_name="val",
        threshold=final_threshold,
        postprocessing_cfg=postprocessing_cfg,
        save_cfg=save_cfg,
        inference_time_ms_per_image=val_inference_time,
    )

    test_records = collect_probabilities_for_split(
        model=model,
        dataloader=test_loader,
        dataset_root=dataset_root,
        split_cfg=test_cfg,
        device=device,
    )

    evaluate_records_and_save(
        records=test_records,
        dataset_name=dataset_name,
        model_name=model_name,
        output_root=output_root,
        split_name="test",
        threshold=final_threshold,
        postprocessing_cfg=postprocessing_cfg,
        save_cfg=save_cfg,
        inference_time_ms_per_image=test_inference_time,
    )

    print(f"Calibrated SegFormer evaluation finished for {dataset_name}.")