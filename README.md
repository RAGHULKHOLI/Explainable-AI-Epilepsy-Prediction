# Explainable AI for Reliable Epilepsy Prediction

A deep-learning project for classifying brain MRI images into **Healthy** and **Epilepsy** categories using **EfficientNet-B0 transfer learning**, with **Grad-CAM** visualizations to make model decisions more interpretable.

> **Important:** This repository is a clean, reproducible implementation scaffold based on the project report. The report does not contain the original training source code or the original dataset, so the code here is a reconstruction of the methodology described in the report rather than a claim that it is the exact original training script.

## Project Overview

Epilepsy can be associated with subtle structural abnormalities that may be difficult to identify consistently from MRI scans. This project explores deep learning for MRI-based classification and adds Explainable AI so that a prediction can be accompanied by a heatmap showing which image regions contributed most to the model's decision.

The reported project uses:
- EfficientNet-B0 with ImageNet transfer learning
- 224 × 224 image input
- Adam optimizer
- Learning rate: 0.001
- CrossEntropyLoss
- 10 epochs
- Random horizontal flip and 10° rotation for training augmentation
- MRI classification into Healthy / Epilepsy
- Grad-CAM for visual explanation

The project report states an overall accuracy of **88.54%** for the implemented EfficientNet-B0 system. It also presents a performance table showing 94% accuracy, so the manuscript contains an internal metric inconsistency that should be resolved using the original experiment logs before treating one number as the definitive final result.

## Research Paper / Report

The accompanying report is included in `report/`.

**Title:** Explainable AI for Reliable Epilepsy Prediction

The report describes MRI preprocessing, 2D/3D CNN concepts, a hybrid CNN-LSTM discussion, EfficientNet-B0 implementation, performance evaluation, and Grad-CAM visualization.

## Repository Structure

```text
Explainable-AI-Epilepsy-Prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── src/
│   ├── train.py
│   ├── evaluate.py
│   └── gradcam.py
├── notebooks/
│   └── epilepsy_prediction.ipynb
├── results/
│   ├── README.md
│   └── gradcam_examples/
├── data/
│   ├── train/
│   │   ├── healthy/
│   │   └── epilepsy/
│   └── test/
│       ├── healthy/
│       └── epilepsy/
└── report/
    └── project_report.docx
```

## Dataset

The report describes an MRI dataset organized into **healthy** and **epilepsy** categories.

Do **not** commit a medical imaging dataset to GitHub unless its license explicitly permits redistribution.

Place an appropriately licensed dataset locally using:

```text
data/
├── train/
│   ├── healthy/
│   └── epilepsy/
└── test/
    ├── healthy/
    └── epilepsy/
```

If you only have one training directory, the training script can create a validation split from it.

## Methodology

### 1. Data preparation
The report describes:
- Standardization
- Skull stripping
- Bias-field correction
- Intensity normalization
- Image resizing
- Data augmentation

The reconstruction code assumes preprocessing has already produced readable image files. It applies resizing, ImageNet normalization, and training augmentation.

### 2. Transfer learning

EfficientNet-B0 is initialized with ImageNet-pretrained weights. The final classifier is replaced with a two-class output layer.

### 3. Training

Default configuration:

| Parameter | Value |
|---|---|
| Model | EfficientNet-B0 |
| Input size | 224 × 224 |
| Classes | Healthy, Epilepsy |
| Train/validation split | 70% / 30% |
| Batch size | 32 |
| Epochs | 10 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | CrossEntropyLoss |
| Augmentation | Horizontal flip + 10° rotation |
| Normalization mean | [0.485, 0.456, 0.406] |
| Normalization std | [0.229, 0.224, 0.225] |

These values follow the configuration reported in the project document.

## Installation

Python 3.10+ is recommended.

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Explainable-AI-Epilepsy-Prediction

python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux/macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

Put your images into:

```text
data/train/healthy/
data/train/epilepsy/
```

Then:

```bash
python src/train.py --data_dir data/train --output_dir checkpoints --epochs 10 --batch_size 32 --lr 0.001
```

The script saves the best model and training history.

## Evaluation

If you have a separate test set:

```bash
python src/evaluate.py --data_dir data/test --checkpoint checkpoints/best_model.pth
```

The evaluation script reports:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## Grad-CAM

Generate an explanation for an MRI image:

```bash
python src/gradcam.py \
    --checkpoint checkpoints/best_model.pth \
    --image path/to/mri_image.jpg \
    --output results/gradcam_examples/example.png
```

The resulting heatmap is overlaid on the MRI image.

Warm regions indicate areas that contributed more strongly to the model's selected prediction, while cooler regions indicate lower contribution.

## Results Reported in the Project Document

The report states:
- Healthy precision: 0.92
- Healthy recall: 0.91
- Healthy F1: 0.91
- Epilepsy precision: 0.95
- Epilepsy recall: 0.96
- Epilepsy F1: 0.96
- Reported table accuracy: 0.94
- Separately stated test-set accuracy: 88.54%

Because these values are not perfectly consistent, the original experiment outputs should be checked before publishing a final GitHub results table.

## Explainable AI

Grad-CAM is used to visualize image regions that influence the prediction. The project report shows examples involving healthy MRI and MRI cases with epileptic features, and discusses cases containing FCD, left hippocampal sclerosis, brain tumors such as meningioma, glioma and pituitary tumors, and stroke.

**Important:** A Grad-CAM heatmap is an explanation of model attention/contribution, not proof that a highlighted region is a clinically confirmed lesion or tumor.

## Limitations

- The reported final classifier is binary: Healthy vs Epilepsy.
- The project is based on MRI images and may be sensitive to image quality and acquisition protocols.
- Dataset diversity can affect generalization.
- The report does not provide enough information to independently reproduce the original dataset and exact experiment.
- Medical AI predictions must not be treated as a standalone clinical diagnosis.

## Future Work

Potential extensions described in the report include moving beyond binary epilepsy classification toward specific epilepsy types such as temporal-lobe and generalized epilepsy.

Other engineering extensions could include:
- Better external validation
- Patient-level dataset splitting
- Multi-modal MRI fusion
- 3D volumetric modeling
- Robust preprocessing pipelines
- Calibration and uncertainty estimation
- Prospective clinical validation

## Citation

If you use this repository in academic work, cite the project report and the original papers listed in `report/`.

## Authors

**Raghul S.** and project team  
Department of Biomedical Engineering  
Rajalakshmi Engineering College, Chennai, India

## Disclaimer

This repository is for **research and educational purposes only**. It is not a medical device and should not be used to diagnose or treat patients.
