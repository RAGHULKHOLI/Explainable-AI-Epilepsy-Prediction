# Project Results

## Overall Test Accuracy
**88.54%**

The report states that the final EfficientNetB0 model achieved 88.54% accuracy on the unseen test set.

## Class-wise Metrics

| Class | F1-score | Recall | Precision |
|---|---:|---:|---:|
| Healthy | 90% | 92% | 89% |
| Epilepsy | 86% | 84% | 88% |
| Weighted average | 89% | 89% | 89% |
| Macro average | 88% | 88% | 89% |

## Model Accuracy Comparison

| Model | Accuracy |
|---|---:|
| EfficientNetB0 (our model) | 88.54% |
| ResNet-50 | 87.50% |
| InceptionV3 | 87.00% |
| MobileNetV2 | 85.00% |
| VGG16 | 80.00% |

## Explainable AI Results

Grad-CAM was used to generate heatmaps showing regions that contributed to the model's epilepsy prediction.

The report includes:
- Healthy MRI prediction
- Moderate epilepsy prediction with severity score **0.53**
- Moderate epilepsy prediction with severity score **0.56**

### Important
The report describes the severity score as a conceptual score based on Grad-CAM heatmap properties and model confidence. It should **not** be presented as a clinically validated disease-severity measurement.

## Source
Extracted from the project's completed report, Chapter 8 and the "Outputs Achieved" section.
