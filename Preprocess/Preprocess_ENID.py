from pathlib import Path
import shutil
import pandas as pd
from PIL import Image
import numpy as np


# ============================================================
# ENID 60/20/20 split preparation script
# ============================================================

# --- Input folders ---
FRAMES_DIR = Path(r"F:\Datasets\Original raw\ENID_v1.0_dataset\ENID_v1.0_dataset\frames")
ANNOTS_DIR = Path(r"F:\Datasets\Original raw\ENID_v1.0_dataset\ENID_v1.0_dataset\annots")

# --- Official ENID split CSV files ---
TRAIN_CSV = Path(r"F:\Datasets\Original raw\ENID_v1.0_split_60_20_20\ENID_v1.0_dataset\train.csv")
VAL_CSV   = Path(r"F:\Datasets\Original raw\ENID_v1.0_split_60_20_20\ENID_v1.0_dataset\val.csv")
TEST_CSV  = Path(r"F:\Datasets\Original raw\ENID_v1.0_split_60_20_20\ENID_v1.0_dataset\test.csv")

# --- Output folder ---
OUT_ROOT = Path(r"F:\Datasets\Standardized datasets\ENID\ENID 60_20_20 Split")

# If True, deletes existing output split folders before recreating them.
# Keep False unless you intentionally want to overwrite everything.
OVERWRITE_EXISTING = False


def read_split_csv(csv_path: Path) -> pd.DataFrame:
    """
    Reads an official ENID split CSV.

    The provided CSV files are headerless and contain two columns:
      column 0: annots/xxx.png
      column 1: frames/xxx.jpg
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path, header=None)

    if df.shape[1] != 2:
        raise ValueError(
            f"Expected 2 columns in {csv_path}, but found {df.shape[1]} columns."
        )

    df.columns = ["mask_relpath", "image_relpath"]

    # Remove possible whitespace
    df["mask_relpath"] = df["mask_relpath"].astype(str).str.strip()
    df["image_relpath"] = df["image_relpath"].astype(str).str.strip()

    return df


def binary_convert_mask(mask_path: Path) -> Image.Image:
    """
    Converts ENID colored annotation masks to binary segmentation masks.

    ENID colors:
      background: 0 0 0
      lesion:     204 81 81

    Output:
      background = 0   black
      lesion     = 255 white

    Uses non-black thresholding to be robust against minor color variation.
    """
    mask = Image.open(mask_path).convert("RGB")
    mask_np = np.array(mask)

    # Any non-black pixel becomes lesion.
    binary = np.any(mask_np > 0, axis=-1).astype(np.uint8) * 255

    return Image.fromarray(binary, mode="L")


def prepare_output_dirs(split_name: str):
    split_dir = OUT_ROOT / split_name
    images_out = split_dir / "images"
    masks_out = split_dir / "masks"

    if OVERWRITE_EXISTING and split_dir.exists():
        shutil.rmtree(split_dir)

    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    return images_out, masks_out


def resolve_image_path(image_relpath: str) -> Path:
    """
    Converts CSV image path like:
      frames/c_3_v_(video_24.mp4)_f_137.jpg
    into:
      FRAMES_DIR/c_3_v_(video_24.mp4)_f_137.jpg
    """
    filename = Path(image_relpath).name
    return FRAMES_DIR / filename


def resolve_mask_path(mask_relpath: str) -> Path:
    """
    Converts CSV mask path like:
      annots/c_3_v_(video_24.mp4)_f_137.png
    into:
      ANNOTS_DIR/c_3_v_(video_24.mp4)_f_137.png
    """
    filename = Path(mask_relpath).name
    return ANNOTS_DIR / filename


def process_split(split_name: str, csv_path: Path):
    print(f"\nProcessing split: {split_name}")
    print(f"CSV: {csv_path}")

    df = read_split_csv(csv_path)
    images_out, masks_out = prepare_output_dirs(split_name)

    missing_images = []
    missing_masks = []
    processed = 0

    for _, row in df.iterrows():
        image_src = resolve_image_path(row["image_relpath"])
        mask_src = resolve_mask_path(row["mask_relpath"])

        if not image_src.exists():
            missing_images.append(str(image_src))
            continue

        if not mask_src.exists():
            missing_masks.append(str(mask_src))
            continue

        # Keep original image filename.
        image_dst = images_out / image_src.name

        # Save binary mask with matching stem and .png extension.
        mask_dst = masks_out / f"{image_src.stem}.png"

        # Copy RGB laparoscopic image.
        shutil.copy2(image_src, image_dst)

        # Convert colored ENID mask to binary black-white mask.
        binary_mask = binary_convert_mask(mask_src)
        binary_mask.save(mask_dst)

        processed += 1

    print(f"Expected items from CSV: {len(df)}")
    print(f"Processed successfully:  {processed}")
    print(f"Missing images:          {len(missing_images)}")
    print(f"Missing masks:           {len(missing_masks)}")

    if missing_images:
        print("\nMissing image files:")
        for p in missing_images[:20]:
            print(f"  {p}")
        if len(missing_images) > 20:
            print(f"  ... and {len(missing_images) - 20} more")

    if missing_masks:
        print("\nMissing mask files:")
        for p in missing_masks[:20]:
            print(f"  {p}")
        if len(missing_masks) > 20:
            print(f"  ... and {len(missing_masks) - 20} more")

    return processed, missing_images, missing_masks


def sanity_check_output():
    """
    Prints final file counts and checks image-mask name matching.
    """
    print("\n" + "=" * 60)
    print("Final output sanity check")
    print("=" * 60)

    for split_name in ["train", "val", "test"]:
        images_dir = OUT_ROOT / split_name / "images"
        masks_dir = OUT_ROOT / split_name / "masks"

        image_files = sorted(images_dir.glob("*.jpg"))
        mask_files = sorted(masks_dir.glob("*.png"))

        image_stems = {p.stem for p in image_files}
        mask_stems = {p.stem for p in mask_files}

        missing_masks = sorted(image_stems - mask_stems)
        missing_images = sorted(mask_stems - image_stems)

        print(f"\n{split_name}")
        print(f"  images: {len(image_files)}")
        print(f"  masks:  {len(mask_files)}")

        if missing_masks:
            print(f"  WARNING: images without masks: {len(missing_masks)}")
            for x in missing_masks[:10]:
                print(f"    {x}")

        if missing_images:
            print(f"  WARNING: masks without images: {len(missing_images)}")
            for x in missing_images[:10]:
                print(f"    {x}")

        if not missing_masks and not missing_images:
            print("  OK: image-mask stems match")


def main():
    print("Preparing ENID 60/20/20 standardized dataset")
    print(f"Frames folder: {FRAMES_DIR}")
    print(f"Annots folder: {ANNOTS_DIR}")
    print(f"Output folder: {OUT_ROOT}")

    if not FRAMES_DIR.exists():
        raise FileNotFoundError(f"Frames folder not found: {FRAMES_DIR}")

    if not ANNOTS_DIR.exists():
        raise FileNotFoundError(f"Annots folder not found: {ANNOTS_DIR}")

    process_split("train", TRAIN_CSV)
    process_split("val", VAL_CSV)
    process_split("test", TEST_CSV)

    sanity_check_output()

    print("\nDone.")


if __name__ == "__main__":
    main()