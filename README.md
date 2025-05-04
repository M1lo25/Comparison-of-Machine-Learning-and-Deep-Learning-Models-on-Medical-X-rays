# Comparison of Machine Learning and Deep Learning Models on Medical X-rays

---

This project comparatively analyzes the performance of traditional Machine Learning models and deep neural networks for the classification of radiographic images, using the [MURA (Musculoskeletal Radiographs)](https://stanfordmlgroup.github.io/competitions/mura/) dataset. The goal is to evaluate the effectiveness of the models in detecting bone anomalies, contributing to the development of AI-assisted diagnostic systems.

## Objectives

- Compare SVM, KNN, Decision Tree, MLP, VGG-16 and DenseNet169
- Evaluate metrics such as Accuracy, Precision, Recall, F1-score, AUC
- Perform Ablation Study to understand the impact of hyperparameters and architectures
- Analyze the effect of resolution (64x64 vs 128x128)
- Position the best model within the context of the MURA leaderboard

## Models Analyzed

| Type              | Models                                         |
|-------------------|------------------------------------------------|
| Machine Learning  | SVM, K-Nearest Neighbors (KNN), Decision Tree  |
| Deep Learning     | Multi-Layer Perceptron (MLP), VGG-16, DenseNet169 |

## Dataset

- **Source**: [MURA - Stanford ML Group](https://stanfordmlgroup.github.io/competitions/mura/)
- **Content**: radiographic images labeled as "normal" or "abnormal"
- **Format**: `.png` images, binary labels (`0 = normal`, `1 = abnormal`)
- **Structure**: `train`, `valid` folders and CSVs with paths and labels

## Methodology

- Preprocessing:
  - Grayscale, resizing to 64x64 / 128x128, normalization
  - Flattening (for traditional ML models)
- Training:
  - Balanced dataset, caching of preprocessed data
- Metrics:
  - Accuracy, Precision, Recall, F1-score, ROC-AUC

## Main Results

### Machine Learning

| Model          | Accuracy | F1-score (0) | F1-score (1) |
|----------------|----------|--------------|--------------|
| Decision Tree  | 57.21%   | 58%          | 57%          |
| KNN            | 57.08%   | 59%          | 55%          |
| SVM            | 56.15%   | 59%          | 53%          |

### Deep Learning

| Model          | Accuracy     | F1-score (0) | F1-score (1) |
|----------------|--------------|--------------|--------------|
| DenseNet169    | 67.94% → 70.94% (128x128) | 71% | 69% |
| VGG-16         | 64.59%       | 62%          | 67%          |
| MLP            | 55.74%       | 51%          | 59%          |

**Best model:** DenseNet169 (AUC = 0.74)

## Ablation Study

The following were analyzed:
- Different criteria and depths for Decision Tree
- Parameters C and kernel in SVM
- K, metric and normalization for KNN
- Structure and optimizers in MLP
- Dropout and optimizers in VGG-16
- Number of neurons and regularization in DenseNet169

## Technologies Used

- Python, NumPy, Pandas, OpenCV, scikit-learn
- TensorFlow / Keras
- Matplotlib, Seaborn

## Authors
Gianluca Milani, Luca Evangelisti, Alessandro Manucci

## References

- [MURA Dataset](https://stanfordmlgroup.github.io/competitions/mura/)
- [Official MURA Model Leaderboard](https://stanfordmlgroup.github.io/competitions/mura/)
