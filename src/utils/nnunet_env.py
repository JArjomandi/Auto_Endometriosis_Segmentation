import os
from pathlib import Path


NNUNET_BASE = Path(r"F:\Results\SAM_Benchmarking\nnUNet")

NNUNET_RAW = NNUNET_BASE / "nnUNet_raw"
NNUNET_PREPROCESSED = NNUNET_BASE / "nnUNet_preprocessed"
NNUNET_RESULTS = NNUNET_BASE / "nnUNet_results"
NNUNET_EXPORTS = NNUNET_BASE / "evaluation_exports"


def setup_nnunet_environment():
    """
    Set nnU-Net v2 environment variables from inside Python.

    This makes nnU-Net runnable from PyCharm without .bat files.
    """

    NNUNET_RAW.mkdir(parents=True, exist_ok=True)
    NNUNET_PREPROCESSED.mkdir(parents=True, exist_ok=True)
    NNUNET_RESULTS.mkdir(parents=True, exist_ok=True)
    NNUNET_EXPORTS.mkdir(parents=True, exist_ok=True)

    os.environ["nnUNet_raw"] = str(NNUNET_RAW)
    os.environ["nnUNet_preprocessed"] = str(NNUNET_PREPROCESSED)
    os.environ["nnUNet_results"] = str(NNUNET_RESULTS)

    print("nnU-Net environment variables:")
    print(f"nnUNet_raw={os.environ['nnUNet_raw']}")
    print(f"nnUNet_preprocessed={os.environ['nnUNet_preprocessed']}")
    print(f"nnUNet_results={os.environ['nnUNet_results']}")


def get_nnunet_environment():
    setup_nnunet_environment()
    return os.environ.copy()