# Auto_Endometriosis_Segmentation
Benchmarking SAM based models and comparison with supervised and weakly supervised models as well as hybrid combinations for zero-shot, few shot and automated endometriosis lesion segmentation on laparascopic images

Research code for automatic endometriosis lesion segmentation in laparoscopic images.

This repository contains preprocessing, training, inference, benchmarking, prompt-comparison, hybrid segmentation, and model-comparison scripts for evaluating conventional segmentation models and SAM-based foundation models on endometriosis segmentation datasets.

The project focuses on automatic lesion segmentation in laparoscopic images, with experiments on datasets such as ENID, GLENDA, and GLENDA-clean.

---

## Repository Structure

```text
Auto_Endometriosis_Segmentation/
│
├── configs/
│   └── Configuration files and path/model settings
│
├── Hybrid/
│   └── Hybrid SegFormer + SAM/SurgiSAM2 refinement utilities
│
├── Model_comparison/
│   └── Scripts for comparing models, prompts, statistics, inference time, and figures
│
├── Preprocess/
│   └── Dataset preprocessing and standardization scripts
│
├── run_models/
│   └── Main executable scripts for running training/inference pipelines
│
├── scripts/
│   └── Additional utility scripts
│
├── src/
│   └── Core source code, model utilities, datasets, metrics, and helper functions
│
├── LICENSE
│
└── README.md
```

---

## Project Overview

This repository benchmarks multiple approaches for endometriosis lesion segmentation from laparoscopic images.

The code supports:

* Dataset preprocessing and standardization
* Supervised model training and inference
* SAM-based promptable segmentation
* Prompt comparison across SAM-based models
* Hybrid segmentation pipelines
* Metric calculation from predicted masks
* Statistical comparison between models
* Inference-time comparison
* Publication-ready plots and Excel summaries

The main evaluation metrics are:

* Dice score
* Intersection over Union
* Precision
* Recall

---

## Supported Model Groups

### Supervised segmentation models

The repository is designed to support conventional supervised segmentation models such as:

```text
nnU-Net / nnU-Net v2
UNet++
DeepLabV3+
SegFormer
YOLO-based segmentation models
```

### SAM-based models

The prompt-comparison pipeline supports:

```text
SAM2
MedSAM
SAM-Med2D
SurgiSAM / SurgiSAM2
```

### Hybrid models

Hybrid experiments combine supervised segmentation models with SAM-based refinement.

The main hybrid pipeline is:

```text
Input image
    -> SegFormer lesion prediction
    -> connected-component extraction
    -> automatic box prompt generation
    -> SurgiSAM2 mask refinement
    -> optional fallback to SegFormer prediction
```

---

## Folder Descriptions

### `configs/`

Contains configuration files and path/model settings.

Use this folder for:

* Dataset path configuration
* Model checkpoint paths
* Training configuration
* Inference configuration
* Experiment-specific settings

If running the repository on a new machine, update paths in the relevant config files or at the top of each script.

---

### `Preprocess/`

Contains scripts for preparing datasets before training or evaluation.

Typical preprocessing tasks include:

* Converting raw datasets into a standardized folder structure
* Resizing or checking images and masks
* Splitting datasets into train/validation/test sets
* Verifying image-mask correspondence
* Preparing binary lesion masks

Expected standardized dataset format:

```text
<DATASET_ROOT>/
│
├── train/
│   ├── images/
│   └── masks/
│
├── val/
│   ├── images/
│   └── masks/
│
└── test/
    ├── images/
    └── masks/
```

Example local dataset paths used during development:

```text
F:\Datasets\Standardized datasets\ENID\ENID 60_20_20 Split
F:\Datasets\Standardized datasets\GLENDA\GLENDA 60_20_20 split
F:\Datasets\Standardized datasets\GLENDA_clean\GLENDA_clean 60_20_20 split
```

---

### `src/`

Contains core source code used by the training, inference, and evaluation scripts.

This folder may include:

* Dataset loaders
* Model definitions
* Training utilities
* Inference utilities
* Metric functions
* Mask-processing functions
* Visualization helpers

Scripts in `run_models/`, `Hybrid/`, and `Model_comparison/` may import functions from `src/`.

---

### `run_models/`

Contains executable scripts for running model experiments.

Use this folder to launch:

* Model training
* Model inference
* SAM-based inference
* Hybrid inference
* Dataset-specific model runs

Example usage pattern:

```cmd
C:\Venvs\sam2-env\Scripts\python.exe "F:\GitHub repos\Auto_Endometriosis_Segmentation\run_models\<script_name>.py"
```

Many scripts define key settings near the top, for example:

```python
DATASETS_TO_RUN = ["ENID", "GLENDA", "GLENDA_clean"]
SPLITS_TO_RUN = ["val", "test"]
MAX_IMAGES_DEBUG = None
```

For a quick debug run:

```python
MAX_IMAGES_DEBUG = 5
```

For full evaluation:

```python
MAX_IMAGES_DEBUG = None
```

---

### `Hybrid/`

Contains code for hybrid segmentation experiments.

The main hybrid strategy is to use a supervised model, such as SegFormer, to generate an initial lesion mask and then refine it using a SAM-based model.

Typical hybrid process:

```text
Image
    -> SegFormer prediction
    -> connected components
    -> automatic bounding boxes
    -> SurgiSAM2 prediction from boxes
    -> candidate selection
    -> fallback if refinement is not accepted
```

Important utilities may include:

* Connected-component extraction
* Automatic box generation
* SegFormer inference
* SurgiSAM2 inference
* Candidate mask selection
* Fallback rules
* Dice/IoU agreement calculation
* Area-ratio filtering

The fallback hybrid is useful when SAM refinement occasionally removes valid lesion pixels. In that case, the script can keep the original SegFormer component unless the SAM candidate satisfies the acceptance criteria.

Example acceptance criteria:

```text
Minimum IoU agreement
Minimum Dice agreement
Area-ratio range
Optional clipping to a dilated SegFormer component
```

---

### `Model_comparison/`

Contains scripts for comparing model outputs and generating final analysis files.

This folder is used for:

* Prompt comparison
* Model metric comparison
* Statistical testing
* Inference-time comparison
* Qualitative figure generation
* Excel summary generation
* Publication-ready plots

Important outputs usually go to:

```text
F:\Results\SAM_Benchmarking\Model_comparison
```

---

### `scripts/`

Contains additional utility scripts.

This folder can be used for:

* One-time data checks
* File conversion
* Debugging scripts
* Helper scripts for organizing outputs
* Miscellaneous project utilities

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/JArjomandi/Auto_Endometriosis_Segmentation.git
cd Auto_Endometriosis_Segmentation
```

---

### 2. Create a Python environment

On Windows:

```cmd
python -m venv C:\Venvs\sam2-env
C:\Venvs\sam2-env\Scripts\activate
```

Upgrade pip:

```cmd
python -m pip install --upgrade pip setuptools wheel
```

---

### 3. Install core dependencies

```cmd
pip install numpy pandas matplotlib pillow openpyxl scipy scikit-image scikit-learn tqdm opencv-python
```

Install PyTorch according to your CUDA version.

Example for CUDA 12.8:

```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Additional dependencies may be needed depending on which models are used:

```cmd
pip install transformers segmentation-models-pytorch albumentations
```

Some external models, such as SAM2, MedSAM, SAM-Med2D, and SurgiSAM/SurgiSAM2, may require installation from their original repositories.

---

## External Model Repositories

Some SAM-based models are expected to be installed separately.

Example local paths used during development:

```text
F:\Models\SAM2
F:\Models\SurgiSAM2
```

Example SurgiSAM2 checkpoint:

```text
F:\Models\SurgiSAM2\checkpoints\Curated400_checkpoint_image_predictor.pt
```

Example SAM2/SurgiSAM2 config:

```text
configs/sam2/sam2_hiera_b+.yaml
```

If a script requires an external model repository, update the path variables at the top of the script.

---

## Expected Results Folder Structure

The main results root used by the project is usually:

```text
F:\Results\SAM_Benchmarking
```

A typical SAM-based inference result folder follows this format:

```text
F:\Results\SAM_Benchmarking\<DATASET>\<MODEL>\frozen\<PROMPT>\<SPLIT>\inference_results.csv
```

Example:

```text
F:\Results\SAM_Benchmarking\ENID\SAM2\frozen\GT_box\test\inference_results.csv
```

---

## SAM Prompt Folder Structure

The prompt-comparison script expects prompt folders such as:

```text
GT_point
GT_box
GT_box_point
GT_box_posneg
```

Example SAM2 structure:

```text
F:\Results\SAM_Benchmarking\ENID\SAM2\frozen\GT_point\test\inference_results.csv
F:\Results\SAM_Benchmarking\ENID\SAM2\frozen\GT_box\test\inference_results.csv
F:\Results\SAM_Benchmarking\ENID\SAM2\frozen\GT_box_point\test\inference_results.csv
F:\Results\SAM_Benchmarking\ENID\SAM2\frozen\GT_box_posneg\test\inference_results.csv
```

Example MedSAM structure:

```text
F:\Results\SAM_Benchmarking\ENID\MedSAM\frozen\GT_box\test\inference_results.csv
```

Example SAM-Med2D structure:

```text
F:\Results\SAM_Benchmarking\ENID\SAM-Med2D\frozen\GT_point\test\inference_results.csv
F:\Results\SAM_Benchmarking\ENID\SAM-Med2D\frozen\GT_box\test\inference_results.csv
```

Example SurgiSAM2 structure:

```text
F:\Results\SAM_Benchmarking\ENID\SurgiSAM2\frozen\GT_point\test\inference_results.csv
F:\Results\SAM_Benchmarking\ENID\SurgiSAM2\frozen\GT_box\test\inference_results.csv
F:\Results\SAM_Benchmarking\ENID\SurgiSAM2\frozen\GT_box_point\test\inference_results.csv
F:\Results\SAM_Benchmarking\ENID\SurgiSAM2\frozen\GT_box_posneg\test\inference_results.csv
```

---

## SAM Prompt Support

The prompt comparison follows this support table:

| Model used | Point             | Box | Point + Box       | Box + positive + negative points |
| ---------- | ----------------- | --- | ----------------- | -------------------------------- |
| SAM2       | Yes               | Yes | Yes               | Yes                              |
| MedSAM     | No / not standard | Yes | No / not standard | No / not standard                |
| SAM-Med2D  | Yes               | Yes | No / not standard | No / not standard                |
| SurgiSAM   | Yes               | Yes | Yes               | Yes                              |

Prompt folders are mapped as:

| Prompt type                      | Folder name     |
| -------------------------------- | --------------- |
| Point                            | `GT_point`      |
| Box                              | `GT_box`        |
| Box + Point                      | `GT_box_point`  |
| Box + positive + negative points | `GT_box_posneg` |

---

## `inference_results.csv` Format

For SAM prompt comparison, each `inference_results.csv` should contain predicted mask information and ground-truth mask information.

Expected or supported columns include:

```text
image_name
mask_name
gt_mask_name
instance_mask_name
pred_mask_name
lesion_id
bbox_xyxy
bbox_x1
bbox_y1
bbox_x2
bbox_y2
prompt_mode
inference_time_sec
```

At minimum, the script needs:

```text
mask_name or gt_mask_name
instance_mask_name or pred_mask_name
```

These are used to find the ground-truth mask and predicted mask, then calculate Dice, IoU, precision, and recall.

---

## Usage

### Run SAM prompt comparison

This script compares SAM2, MedSAM, SAM-Med2D, and SurgiSAM/SurgiSAM2 across available prompt types.

```cmd
C:\Venvs\sam2-env\Scripts\python.exe "F:\GitHub repos\Auto_Endometriosis_Segmentation\Model_comparison\prompt_comparison.py"
```

Outputs:

```text
F:\Results\SAM_Benchmarking\Model_comparison\SAM_prompt_comparison
```

Important output files:

```text
figures_600dpi/
sam_prompt_comparison_summary.xlsx
calculated_prompt_metrics_image_level.csv
metric_calculation_check.csv
discovered_prompt_files.csv
```

The script creates one boxplot per:

```text
Dataset
Split
Metric
```

For example:

```text
ENID_test_dice_sam_prompt_boxplots_600dpi.png
ENID_test_iou_sam_prompt_boxplots_600dpi.png
ENID_test_precision_sam_prompt_boxplots_600dpi.png
ENID_test_recall_sam_prompt_boxplots_600dpi.png
```

Each plot shows:

* Prompt type on the x-axis
* Metric score on the y-axis
* One boxplot per available SAM-based model
* Mean as a red line
* Median as a black line

---

### Run hybrid inference

Example:

```cmd
C:\Venvs\sam2-env\Scripts\python.exe "F:\GitHub repos\Auto_Endometriosis_Segmentation\run_models\run_hybrid_segformer_surgisam2_autobox_fallback.py"
```

Typical output folder:

```text
F:\Results\SAM_Benchmarking\<DATASET>\SegFormer_SurgiSAM2_AutoBox_Fallback\hybrid\Auto_box_fallback_dice_0p85_area_0p70_1p30\<SPLIT>
```

The hybrid pipeline uses:

```text
SegFormer prediction
Automatic box prompt generation
SurgiSAM2 refinement
Fallback if the refined mask is not accepted
```

---

### Run statistical comparison

Example:

```cmd
C:\Venvs\sam2-env\Scripts\python.exe "F:\GitHub repos\Auto_Endometriosis_Segmentation\Model_comparison\compare_segformer_vs_fallback_hybrid_surgisam2_statistics.py"
```

Typical output:

```text
F:\Results\SAM_Benchmarking\Model_comparison\statistical_comparison\segformer_vs_fallback_hybrid_surgisam2_statistics.xlsx
```

---

### Run inference-time comparison

Example:

```cmd
C:\Venvs\sam2-env\Scripts\python.exe "F:\GitHub repos\Auto_Endometriosis_Segmentation\Model_comparison\calculate_all_model_inference_times.py"
```

Typical output:

```text
F:\Results\SAM_Benchmarking\Model_comparison\inference_time_comparison\all_models_mean_inference_times.xlsx
```

---

## Metrics

The repository uses standard binary segmentation metrics.

### Dice

```text
Dice = 2TP / (2TP + FP + FN)
```

### Intersection over Union

```text
IoU = TP / (TP + FP + FN)
```

### Precision

```text
Precision = TP / (TP + FP)
```

### Recall

```text
Recall = TP / (TP + FN)
```

For prompt-based SAM experiments, metrics are calculated directly from predicted masks and ground-truth masks.

---

## Output Summary Files

The prompt-comparison script creates:

```text
sam_prompt_comparison_summary.xlsx
```

Important sheets:

| Sheet                      | Description                                                                  |
| -------------------------- | ---------------------------------------------------------------------------- |
| `simple_mean_std`          | Mean ± standard deviation for each dataset, split, model, prompt, and metric |
| `wide_mean_std`            | Compact summary with models as columns                                       |
| `prompt_support`           | Prompt support table                                                         |
| `saved_plots`              | Paths to generated PNG figures                                               |
| `image_level_metrics`      | Image-level metric values                                                    |
| `discovered_files`         | Found and missing result files                                               |
| `metric_calculation_check` | Diagnostic information for mask matching and metric calculation              |

---

## Reproducibility Notes

Results depend on:

```text
Dataset version
Train/validation/test split
Model checkpoint
Prompt type
Inference script version
Metric calculation script version
Post-processing settings
CUDA/PyTorch version
```

Recommended practice:

```text
Save all inference_results.csv files.
Save all predicted masks.
Save generated Excel summaries.
Save plotting scripts used for final figures.
Record model checkpoints and dataset splits.
```

---

## Hardware and Environment

Development environment example:

```text
OS: Windows 11
Python: 3.11
GPU: NVIDIA GeForce RTX 5090
CUDA: 12.8
PyTorch: CUDA-enabled build
```

The code can be adapted to Linux by changing paths and environment commands.

---

## Project Status

This repository is under active research development. Scripts are designed for benchmarking, ablation studies, and manuscript figure generation. Paths and external model dependencies may need to be edited before running on another system.

---

## License

See the `LICENSE` file.

---

## Citation

If this repository is used in academic work, cite the repository as:

```text
Arjomandi, J. Auto Endometriosis Segmentation. GitHub repository.
https://github.com/JArjomandi/Auto_Endometriosis_Segmentation
```

If a related manuscript is published, cite the corresponding paper.

---

## Acknowledgements

This project builds on open-source segmentation frameworks and foundation models, including:

```text
PyTorch
nnU-Net
SegFormer
SAM / SAM2
MedSAM
SAM-Med2D
SurgiSAM / SurgiSAM2
```

Datasets and third-party model checkpoints are not redistributed in this repository. Refer to the original dataset and model sources for licensing and access conditions.
