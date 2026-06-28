from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.visualize_metrics import visualize_dataset_model_split


RESULTS_ROOT = Path(r"F:\Results\SAM_Benchmarking")

DATASETS_TO_VISUALIZE = [
    "ENID",
    "GLENDA",
    "GLENDA_clean",
]

SPLITS_TO_VISUALIZE = [
    "val",
    "test",
]


def main():
    for dataset_name in DATASETS_TO_VISUALIZE:
        for split in SPLITS_TO_VISUALIZE:
            try:
                visualize_dataset_model_split(
                    results_root=RESULTS_ROOT,
                    dataset_name=dataset_name,
                    model_name="nnUNetV2_2D",
                    training_state="trained",
                    split=split,
                )
            except Exception as error:
                print(
                    f"WARNING: Visualization failed for "
                    f"{dataset_name} | nnUNetV2_2D | {split}: {error}"
                )


if __name__ == "__main__":
    main()