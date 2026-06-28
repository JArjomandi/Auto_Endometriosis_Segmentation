from pathlib import Path
import subprocess
import sys

# '''
# # Run order:
# #
# # 1. preprocess\prepare_nnunetv2_datasets.py
# # 2. preprocess\check_nnunet_masks.py
# # 3. run_models\run_nnunetv2_2d_all.py
# # 4. run_models\predict_nnunetv2_2d_all.py
# # 5. src\runners\evaluate_nnunetv2_predictions.py
# # 6. run_models\visualize_nnunetv2_2d_all.py
#
# '''

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

SCRIPTS = [
    PROJECT_ROOT / "run_models" / "run_nnunetv2_2d_all.py",
    PROJECT_ROOT / "run_models" / "predict_nnunetv2_2d_all.py",
    PROJECT_ROOT / "src" / "runners" / "evaluate_nnunetv2_predictions.py",
    PROJECT_ROOT / "run_models" / "visualize_nnunetv2_2d_all.py",
]


def run_script(script_path: Path):
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    print("\n" + "=" * 100)
    print(f"Running: {script_path}")
    print("=" * 100)

    result = subprocess.run(
        [PYTHON, str(script_path)],
        cwd=str(PROJECT_ROOT),
        shell=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Script failed with exit code {result.returncode}: {script_path}"
        )


def main():
    for script_path in SCRIPTS:
        run_script(script_path)

    print("\nFull nnU-Net v2 pipeline finished.")


if __name__ == "__main__":
    main()