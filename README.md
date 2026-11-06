# TAL-GRN: Topology-Aware Learning Graph Representation Network for Long-Tailed Multi-Class Medical Image Classification

**TAL-GRN** is a hybrid CNN–GNN framework designed to improve medical image classification under class imbalance. It combines pretrained CNN feature extraction with adaptive graph construction and graph-based message passing.

Unlike conventional graph methods that use a fixed number of neighbors, TAL-GRN dynamically determines how many neighboring samples each image should connect to and learns how important each connection is. This allows the model to use relationships between images to improve classification, particularly for underrepresented classes.

## Key Features

- **Hybrid CNN–GNN architecture:** Combines ResNet-50 feature extraction with graph-based learning.
- **Adaptive neighborhood construction:** Predicts a different neighborhood size for each image instead of using a fixed \(k\).
- **Learnable edge weights:** Learns the importance of connections between image representations.
- **Graph message passing:** Uses relationships between samples to improve their representations.
- **Long-tailed classification:** Evaluates performance across both frequent and underrepresented classes.
- **Extensive ablation studies:** Examines the contribution of adaptive neighborhoods, edge weights, and graph reasoning.
- **Cross-dataset evaluation:** Tests the framework on different medical imaging datasets.

## Model Architecture

TAL-GRN consists of three main components:

### 1. CNN Feature Extraction

A pretrained **ResNet-50** is used as the feature extractor. Its weights remain frozen during training, and each image is converted into a 2048-dimensional feature vector.

\[
h_i = \operatorname{ResNet50}(x_i)
\in \mathbb{R}^{2048}
\]

These feature vectors represent the images and serve as the input to the graph-learning module.

### 2. Adaptive Graph Construction

This is the central component of TAL-GRN. Instead of connecting every image to a fixed number of neighbors, the model estimates a suitable neighborhood size for each image.

First, cosine similarity is calculated between image feature vectors:

\[
S_{ij} =
\frac{h_i}{\|h_i\|}
\cdot
\frac{h_j}{\|h_j\|}
\]

Next, an Adaptive Neighborhood module predicts the number of neighbors:

\[
k_i =
\left\lfloor
K_{\text{base}}+
K_{\alpha}\cdot\sigma(a_i)
\right\rfloor
\]

where:

- \(K_{\text{base}}=5\)
- \(K_{\alpha}=10\)
- \(\sigma\) is the sigmoid function.
- \(k_i\) determines how many similar images are selected as neighbors.

The resulting neighborhood size is approximately between 5 and 15.

The model then uses an Edge Weight Predictor to estimate the importance of each selected connection:

\[
e_{ij} =
\operatorname{softplus}
\left(
\operatorname{MLP}_E
([h_i;h_j;S_{ij}])
\right)
\]

An edge-pruning threshold of \(\tau=0.3\) is used to remove weak connections.

### 3. Graph Representation Learning

The constructed graph is processed by a **3-layer Graph Neural Network (GNN)** with a hidden dimension of 512.

Through message passing, each image representation receives information from its connected neighbors. The updated representations are then passed to the classification layer to predict the medical image class.

## Dataset

The primary benchmark is a brain MRI dataset containing:

| Property | Description |
|---|---|
| Total images | 34,192 |
| Number of classes | 40 |
| Minimum class size | 488 |
| Maximum class size | 1,464 |
| Class imbalance ratio | 3:1 |

The dataset is divided into stratified training, validation, and test sets to preserve the class distribution.

### Additional Datasets

To evaluate generalization across different medical imaging domains, TAL-GRN was also tested on:

- **BloodMNIST:** Peripheral blood cell classification.
- **HAM10000:** Dermoscopic skin lesion classification.
- **Pediatric Chest X-ray:** Normal and pneumonia classification.

## Data Preprocessing

The primary brain MRI dataset uses the following preprocessing pipeline:

- ImageNet normalization.
- Resize images to \(256\times256\).
- Random crop to \(224\times224\) during training.
- Random horizontal flipping.
- Color jitter with brightness, contrast, and saturation of 0.3, and hue of 0.1.
- Validation and test images are resized to \(224\times224\) without augmentation.

## Training Configuration

| Parameter | Value |
|---|---|
| CNN backbone | Frozen ResNet-50 |
| Feature dimension | 2048 |
| GNN layers | 3 |
| GNN hidden dimension | 512 |
| Base neighborhood size | 5 |
| Adaptive neighborhood range | Approximately 5–15 |
| Edge-pruning threshold | 0.3 |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Batch size | 64 |
| Epochs | 50 |
| Learning-rate scheduler | Cosine Annealing |
| Gradient clipping | 1.0 |
| Training precision | Automatic Mixed Precision (AMP) |

The model uses standard cross-entropy loss without oversampling, synthetic data generation, or class-based loss reweighting.

## Experimental Results

### Main Results on Brain MRI

TAL-GRN achieved the following results on the primary 40-class brain MRI benchmark:

| Metric | TAL-GRN |
|---|---:|
| Accuracy | **86.12%** |
| Macro Precision | 0.8506 |
| Macro Recall | 0.8520 |
| Macro F1-score | **0.8472** |
| Weighted F1-score | 0.8558 |
| ROC-AUC (OvR) | 0.9968 |
| MCC | 0.8575 |

The model also showed stable performance across five random seeds:

- Mean accuracy: **85.89% ± 0.39%**
- Mean macro F1-score: **0.8443 ± 0.0049**

### Comparison with Graph-Based Methods

TAL-GRN was compared with several graph-based learning methods using a shared experimental setup.

| Method | Accuracy |
|---|---:|
| TAL-GRN | **86.12%** |
| GCN | 83.20% |
| GraphSMOTE | 83.08% |
| GraphSAGE | 82.79% |
| TAM | 82.60% |
| ReNode | 82.07% |

These comparisons show that adaptive graph construction and learnable edge weighting improve performance over the evaluated fixed-graph approaches.

### Ablation Studies

Extensive ablation experiments were conducted to understand the contribution of individual components.

The experiments examined:

- **Adaptive neighborhood size:** Compared adaptive neighborhood construction with fixed-\(k\) graph construction.
- **Learnable edge weights:** Investigated the contribution of learning connection importance.
- **Graph module:** Examined the effect of removing graph construction and message passing.
- **Backbone comparison:** Compared different CNN feature extractors.
- **Class imbalance methods:** Evaluated traditional loss-based approaches.
- **Graph-based alternatives:** Compared TAL-GRN with GCN, GraphSMOTE, GraphSAGE, TAM, and ReNode.

Selected ablation results:

| Configuration | Accuracy | Macro F1 |
|---|---:|---:|
| Full TAL-GRN | **86.12%** | **0.8472** |
| Fixed neighborhood | 85.22% | — |
| Without learnable edge weights | 83.14% | — |
| Without graph module | 78.20% | 0.7668 |

The results highlight the importance of adaptive graph construction, learnable edge weights, and relational message passing.

## Generalization Results

TAL-GRN was also evaluated on three additional medical imaging datasets.

| Dataset | Baseline Accuracy | TAL-GRN Accuracy |
|---|---:|---:|
| BloodMNIST | 75.53% | **80.62%** |
| HAM10000 (Skin Cancer) | 66.34% | **71.55%** |
| Pediatric Chest X-ray | 91.47% | **93.52%** |

These experiments demonstrate improvements over the frozen ResNet-50 baseline across different medical imaging domains.

## Interpretability

Grad-CAM was applied to correctly classified representative images from each class to visualize the image regions contributing to the CNN's predictions. This provides an additional way to inspect whether the model focuses on relevant image regions.

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Research Objective

The main objective of TAL-GRN is to investigate whether adaptive graph construction and learnable edge weighting can improve long-tailed medical image classification without relying on traditional data-level or loss-level imbalance handling.

The experimental results suggest that learning relationships between image samples, rather than treating every image independently or relying on a fixed graph, can improve classification performance across multiple medical imaging datasets.

## Future Work

- Evaluate additional CNN backbones and graph architectures.
- Investigate more adaptive graph construction strategies.
- Extend evaluation to additional medical imaging datasets.
- Further study graph connectivity and class-wise performance.

---

**Note:** TAL-GRN is a research framework. The reported results correspond to the experimental configurations and datasets described above.
