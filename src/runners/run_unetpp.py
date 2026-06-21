from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as torch_functional
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from src.datasets.segmentation_dataset import BinarySegmentationDataset
from src.models.unetpp.unetpp_model import build_unetpp_model
from src.evaluation.metrics import compute_binary_metrics
from src.utils.visualization import save_overlay


def load_yaml(path):
    path = Path(path)

    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def dice_loss_from_logits(logits, targets, eps=1e-7):
    probs = torch.sigmoid(logits)

    dims = (1, 2, 3)
    intersection = torch.sum(probs * targets, dims)
    cardinality = torch.sum(probs + targets, dims)

    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice.mean()


def combined_loss(logits, targets, dice_weight=0.5, bce_weight=0.5):
    dice = dice_loss_from_logits(logits, targets)
    bce = torch_functional.binary_cross_entropy_with_logits(logits, targets)

    return dice_weight * dice + bce_weight * bce


def save_binary_mask(mask: np.ndarray, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8)).save(output_path)


def find_original_image_path(images_dir: Path, image_name: str) -> Path:
    direct = images_dir / image_name

    if direct.exists():
        return direct

    stem = Path(image_name).stem

    for extension in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
        candidate = images_dir / f"{stem}{extension}"

        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Image not found: {image_name}")


def find_original_mask_path(masks_dir: Path, image_name: str) -> Path:
    stem = Path(image_name).stem

    for extension in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
        candidate = masks_dir / f"{stem}{extension}"

        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Mask not found for: {image_name}")


def load_binary_mask(mask_path: Path):
    mask = Image.open(mask_path).convert("L")
    mask_np = np.array(mask)
    return (mask_np > 0).astype(np.uint8) * 255


def train_one_epoch(model, dataloader, optimizer, device, loss_cfg):
    model.train()

    losses = []

    for batch in dataloader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = combined_loss(
            logits=logits,
            targets=masks,
            dice_weight=loss_cfg.get("dice_weight", 0.5),
            bce_weight=loss_cfg.get("bce_weight", 0.5),
        )

        loss.backward()
        optimizer.step()

        losses.append(float(loss.detach().cpu().item()))

    return float(np.mean(losses))


@torch.no_grad()
def evaluate_split(
    model,
    dataloader,
    dataset_root: Path,
    split_cfg: dict,
    output_root: Path,
    split_name: str,
    device,
    threshold: float,
    image_size: int,
    save_cfg: dict,
):
    model.eval()

    prompt_mode = "No_prompt"

    out_dir = output_root / prompt_mode / split_name
    merged_dir = out_dir / "merged_masks"
    overlay_dir = out_dir / "overlays"

    merged_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    images_dir = dataset_root / split_cfg["images"]
    masks_dir = dataset_root / split_cfg["masks"]

    inference_rows = []
    metric_rows = []

    for batch in dataloader:
        images = batch["image"].to(device)
        image_names = batch["image_name"]

        start_time = time.perf_counter()
        logits = model(images)
        probs = torch.sigmoid(logits)
        elapsed_ms_batch = (time.perf_counter() - start_time) * 1000.0

        probs_np = probs.detach().cpu().numpy()

        batch_size = probs_np.shape[0]
        elapsed_per_image_ms = elapsed_ms_batch / max(batch_size, 1)

        for i in range(batch_size):
            image_name = image_names[i]
            prob = probs_np[i, 0]

            image_path = find_original_image_path(images_dir, image_name)
            mask_path = find_original_mask_path(masks_dir, image_name)

            gt_mask = load_binary_mask(mask_path)
            original_h, original_w = gt_mask.shape

            prob_resized = cv2.resize(
                prob,
                (original_w, original_h),
                interpolation=cv2.INTER_LINEAR,
            )

            pred_mask = (prob_resized >= threshold).astype(np.uint8) * 255

            merged_name = f"{Path(image_name).stem}.png"
            merged_path = merged_dir / merged_name

            if save_cfg.get("merged_masks", True):
                save_binary_mask(pred_mask, merged_path)

            if save_cfg.get("overlays", True):
                overlay_path = overlay_dir / f"{Path(image_name).stem}_overlay.png"

                save_overlay(
                    image_path=image_path,
                    gt_mask=gt_mask,
                    pred_mask=pred_mask,
                    output_path=overlay_path,
                )

            metrics = compute_binary_metrics(
                pred_mask=pred_mask,
                gt_mask=gt_mask,
            )

            inference_rows.append(
                {
                    "dataset": dataset_root.name,
                    "split": split_name,
                    "model_name": "UNetPP",
                    "training_state": "trained",
                    "prompt_mode": prompt_mode,
                    "image_name": image_name,
                    "mask_name": Path(mask_path).name,
                    "threshold": threshold,
                    "inference_time_ms": elapsed_per_image_ms,
                    "merged_mask_name": merged_name,
                }
            )

            metric_row = {
                "dataset": dataset_root.name,
                "split": split_name,
                "model_name": "UNetPP",
                "training_state": "trained",
                "prompt_mode": prompt_mode,
                "image_name": image_name,
                "mask_name": Path(mask_path).name,
                "num_prompt_instances": 0,
            }

            metric_row.update(metrics)
            metric_rows.append(metric_row)

    inference_df = pd.DataFrame(inference_rows)
    metrics_df = pd.DataFrame(metric_rows)

    inference_csv = out_dir / "inference_results.csv"
    metrics_csv = out_dir / "metrics_image_level.csv"
    summary_csv = out_dir / "metrics_summary.csv"

    inference_df.to_csv(inference_csv, index=False)
    metrics_df.to_csv(metrics_csv, index=False)

    numeric_cols = metrics_df.select_dtypes(include="number").columns
    summary_df = metrics_df[numeric_cols].agg(["mean", "std", "median", "min", "max"]).T
    summary_df.to_csv(summary_csv)

    return metrics_df


def train_and_evaluate(experiment_config_path):
    experiment_config_path = Path(experiment_config_path)
    project_root = Path(__file__).resolve().parents[2]

    exp_cfg = load_yaml(experiment_config_path)

    dataset_cfg = load_yaml(project_root / exp_cfg["dataset_config"])
    model_cfg = load_yaml(project_root / exp_cfg["model_config"])

    dataset_name = dataset_cfg["dataset_name"]
    dataset_root = Path(dataset_cfg["dataset_root"])
    output_root = Path(exp_cfg["output_root"])

    device = torch.device(
        model_cfg.get("device", "cuda")
        if torch.cuda.is_available()
        else "cpu"
    )

    image_size = int(model_cfg.get("image_size", 512))
    batch_size = int(model_cfg.get("batch_size", 4))
    num_workers = int(model_cfg.get("num_workers", 4))
    epochs = int(model_cfg.get("epochs", 80))
    threshold = float(model_cfg.get("threshold", 0.5))

    output_root.mkdir(parents=True, exist_ok=True)

    train_split = exp_cfg["splits"]["train"]
    val_split = exp_cfg["splits"]["val"]
    test_split = exp_cfg["splits"]["test"]

    train_cfg = dataset_cfg["splits"][train_split]
    val_cfg = dataset_cfg["splits"][val_split]
    test_cfg = dataset_cfg["splits"][test_split]

    train_dataset = BinarySegmentationDataset(
        images_dir=dataset_root / train_cfg["images"],
        masks_dir=dataset_root / train_cfg["masks"],
        image_size=image_size,
        augment=True,
    )

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
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

    model = build_unetpp_model(
        encoder_name=model_cfg.get("encoder_name", "resnet34"),
        encoder_weights=model_cfg.get("encoder_weights", "imagenet"),
        in_channels=int(model_cfg.get("in_channels", 3)),
        classes=int(model_cfg.get("classes", 1)),
    )

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(model_cfg.get("weight_decay", 1e-5)),
    )

    best_val_dice = -1.0
    best_checkpoint_path = output_root / "checkpoints" / "best_unetpp.pt"
    last_checkpoint_path = output_root / "checkpoints" / "last_unetpp.pt"

    best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history_rows = []

    save_cfg = exp_cfg.get("save", {})

    print("=" * 100)
    print(f"Training UNet++ on {dataset_name}")
    print(f"Dataset root: {dataset_root}")
    print(f"Output root:  {output_root}")
    print(f"Device:       {device}")
    print("=" * 100)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_cfg=model_cfg.get("loss", {}),
        )

        val_metrics = evaluate_split(
            model=model,
            dataloader=val_loader,
            dataset_root=dataset_root,
            split_cfg=val_cfg,
            output_root=output_root,
            split_name="val",
            device=device,
            threshold=threshold,
            image_size=image_size,
            save_cfg={"merged_masks": False, "overlays": False},
        )

        val_dice = float(val_metrics["dice"].mean())

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_dice": val_dice,
                "val_iou": float(val_metrics["iou"].mean()),
                "val_precision": float(val_metrics["precision"].mean()),
                "val_recall": float(val_metrics["recall"].mean()),
            }
        )

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_dice={val_dice:.4f}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_cfg": model_cfg,
                    "epoch": epoch,
                    "best_val_dice": best_val_dice,
                },
                best_checkpoint_path,
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_cfg": model_cfg,
            "epoch": epochs,
            "best_val_dice": best_val_dice,
        },
        last_checkpoint_path,
    )

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(output_root / "training_history.csv", index=False)

    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("=" * 100)
    print(f"Loaded best checkpoint: {best_checkpoint_path}")
    print(f"Best val dice: {best_val_dice:.4f}")
    print("Final evaluation with saved masks and overlays")
    print("=" * 100)

    evaluate_split(
        model=model,
        dataloader=val_loader,
        dataset_root=dataset_root,
        split_cfg=val_cfg,
        output_root=output_root,
        split_name="val",
        device=device,
        threshold=threshold,
        image_size=image_size,
        save_cfg=save_cfg,
    )

    evaluate_split(
        model=model,
        dataloader=test_loader,
        dataset_root=dataset_root,
        split_cfg=test_cfg,
        output_root=output_root,
        split_name="test",
        device=device,
        threshold=threshold,
        image_size=image_size,
        save_cfg=save_cfg,
    )

    print("UNet++ training and evaluation finished.")