from pathlib import Path
import torch


INPUT_CHECKPOINT = Path(
    r"F:\Models\SurgiSAM2\checkpoints\Curated400_checkpoint.pt"
)

OUTPUT_CHECKPOINT = Path(
    r"F:\Models\SurgiSAM2\checkpoints\Curated400_checkpoint_image_predictor.pt"
)

KEYS_TO_REMOVE = [
    "no_obj_embed_spatial",
    "obj_ptr_tpos_proj.weight",
    "obj_ptr_tpos_proj.bias",
]


def main():
    if not INPUT_CHECKPOINT.exists():
        raise FileNotFoundError(f"Input checkpoint not found: {INPUT_CHECKPOINT}")

    print("=" * 100)
    print("Preparing SurgiSAM2 checkpoint for SAM2 image predictor")
    print("=" * 100)
    print(f"Input:  {INPUT_CHECKPOINT}")
    print(f"Output: {OUTPUT_CHECKPOINT}")

    checkpoint = torch.load(
        INPUT_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        save_as_full_checkpoint = True
    else:
        state_dict = checkpoint
        save_as_full_checkpoint = False

    print(f"Original number of keys: {len(state_dict)}")

    removed = []

    for key in KEYS_TO_REMOVE:
        if key in state_dict:
            state_dict.pop(key)
            removed.append(key)

    print(f"Removed keys: {removed}")
    print(f"Remaining number of keys: {len(state_dict)}")

    OUTPUT_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

    if save_as_full_checkpoint:
        checkpoint["model"] = state_dict
        torch.save(checkpoint, OUTPUT_CHECKPOINT)
    else:
        torch.save(state_dict, OUTPUT_CHECKPOINT)

    print("=" * 100)
    print("Saved cleaned checkpoint.")
    print("=" * 100)


if __name__ == "__main__":
    main()