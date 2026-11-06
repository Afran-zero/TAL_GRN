

# TAL-GRN: Topology-Aware Learnable Graph Reweighting Network for Imbalanced Medical Image Classification

**Technologies:** Python, PyTorch, Torchvision, NumPy, Pandas, Scikit-learn

## Overview

TAL-GRN is a graph-based deep learning framework designed for **imbalanced medical image classification**. It addresses the problem of unequal class distribution, where some medical conditions have many training images while others have significantly fewer.

Traditional CNN models classify images independently, which can make them biased toward majority classes. TAL-GRN addresses this problem by allowing images within a mini-batch to exchange information through an adaptively constructed graph.

Unlike conventional graph-based methods that use a fixed number of neighbors and predefined connections, TAL-GRN learns:

- How many neighboring images each sample should connect to.
- How important each connection is for classification.
- How information should be propagated between related images.

The framework uses a frozen ResNet-50 backbone for feature extraction and a topology-aware Graph Neural Network (GNN) for relational learning.

## Key Contributions

- Developed a **Topology-Aware Learnable Graph Reweighting Network (TAL-GRN)** for imbalanced medical image classification.
- Designed an adaptive neighborhood module that determines the number of neighbors for each image.
- Implemented a learnable edge-weight predictor to estimate the importance of graph connections.
- Integrated a three-layer topology-aware GNN with residual connections for graph-based feature learning.
- Conducted extensive comparisons with CNN backbones, traditional class-imbalance methods, and existing graph-based approaches.
- Performed ablation studies to investigate the contribution of adaptive neighborhoods, learnable edge weights, and graph message passing.
- Evaluated generalization on additional medical imaging datasets.

## Architecture

TAL-GRN consists of four main stages:

1. **Feature Extraction:** A pretrained and frozen ResNet-50 extracts a 2048-dimensional feature vector from each image.
2. **Adaptive Graph Construction:** The model determines the number of neighbors for each image and predicts the importance of each candidate connection.
3. **Graph Message Passing:** A three-layer topology-aware GNN propagates information between connected images.
4. **Classification:** A linear classification head predicts the medical image category.

### Architecture Configuration

| Component | Configuration |
|---|---|
| Feature extractor | Frozen ResNet-50 |
| Feature dimension | 2048 |
| Adaptive neighborhood module | Two-layer MLP |
| Edge-weight predictor | Three-layer MLP |
| GNN layers | 3 |
| GNN hidden dimension | 512 |
| Base neighborhood size | 5 |
| Adaptive neighborhood range | Approximately 5–15 |
| Edge-pruning threshold | 0.3 |
| Classification head | Linear layer |

## Methodology

### 1. Feature Extraction

A pretrained ResNet-50 is used as the feature extraction backbone. Its final fully connected layer is removed, and its weights are frozen.

Each input image is transformed into a feature vector:

$$
h_i = \operatorname{Backbone}(x_i)
$$

where:

- $x_i$ represents an input image.
- $h_i$ represents its extracted feature vector.
- $h_i \in \mathbb{R}^{2048}$.

### 2. Adaptive Graph Construction

Instead of connecting every image to a fixed number of neighbors, TAL-GRN dynamically determines the neighborhood size for each image.

First, cosine similarity is calculated between image feature vectors:

$$
S_{ij} =
\frac{h_i}{\|h_i\|_2}
\cdot
\frac{h_j}{\|h_j\|_2}
$$

The adaptive neighborhood module then predicts the number of neighbors:

$$
k_i =
\left\lfloor
K_{\text{base}} +
K_{\alpha}\sigma(a_i)
\right\rfloor
$$

where:

$$
a_i = \operatorname{MLP}_N(h_i)
$$

Here:

- $K_{\text{base}} = 5$
- $K_{\alpha} = 10$
- $\sigma$ is the sigmoid function.
- $k_i$ is the selected neighborhood size.

This allows different images to use different amounts of relational information.

### 3. Learnable Edge-Weight Prediction

After selecting candidate neighbors, TAL-GRN predicts how important each connection is.

The Edge Weight Predictor (EWP) uses both endpoint features and their cosine similarity:

$$
e_{ij} =
\operatorname{softplus}
\left(
\operatorname{MLP}_E
([h_i;h_j;S_{ij}])
\right)
$$

Connections with weights below the threshold are removed:

$$
E =
\left\{
(i,j):
j \in \operatorname{top-}k_i(S_i),
\ e_{ij} > \tau
\right\}
$$

where $\tau = 0.3$.

Self-loops are also added to preserve each image's own features during message passing.

The resulting weighted adjacency matrix is normalized:

$$
\hat{A} = D^{-1/2}AD^{-1/2}
$$

where $A$ is the weighted adjacency matrix and $D$ is its degree matrix.

### 4. Graph Message Passing

TAL-GRN uses three graph neural network layers to combine information from neighboring images.

The message aggregation step is:

$$
m^{(\ell)} =
\hat{A}W_m^{(\ell)}h^{(\ell)}
$$

The node update is:

$$
h_i^{(\ell+1)} =
\operatorname{LayerNorm}
\left(
\operatorname{GELU}
\left(
W_u^{(\ell)}
[h_i^{(\ell)};m_i^{(\ell)}]
\right)
\right)
+
R^{(\ell)}(h_i^{(\ell)})
$$

The architecture combines graph-based message passing, nonlinear feature transformation, normalization, and residual connections.

### 5. Classification

The final graph representation is passed through a linear classification layer:

$$
\operatorname{logits}_i =
W_ch_i^{(L)} + b_c
$$

The model is trained using standard categorical cross-entropy:

$$
\mathcal{L} =
-\frac{1}{B}
\sum_i
\log
\left(
\frac{
\exp(\operatorname{logits}_{i,y_i})
}{
\sum_c \exp(\operatorname{logits}_{i,c})
}
\right)
$$

TAL-GRN does not use class-weighted loss or oversampling. Its imbalance-handling strategy is based on adaptive graph construction and learnable edge weights.

## Dataset

The primary experiment uses a **40-class brain MRI dataset** containing 34,192 images.

| Dataset property | Description |
|---|---|
| Total images | 34,192 |
| Number of classes | 40 |
| Minimum class size | 488 |
| Maximum class size | 1,464 |
| Imbalance ratio | 3:1 |
| Data splitting | Stratified train/validation/test |

The dataset is divided into training, validation, and test sets while preserving the class distribution.

### Data Preprocessing

- Images are resized to $256 \times 256$.
- Random cropping is applied to obtain $224 \times 224$ images.
- Random horizontal flipping and color jitter are used during training.
- ImageNet mean and standard deviation are used for normalization.
- Validation and test images are resized to $224 \times 224$ without random augmentation.

## Experimental Setup

All experiments use a shared training configuration:

| Parameter | Value |
|---|---|
| Epochs | 50 |
| Batch size | 64 |
| Learning rate | $1 \times 10^{-3}$ |
| Weight decay | $1 \times 10^{-4}$ |
| Optimizer | AdamW |
| Learning-rate scheduler | Cosine Annealing |
| Gradient clipping | 1.0 |
| Training precision | Automatic Mixed Precision (AMP) |
| GPU environment | 2 × NVIDIA T4 |

The model checkpoint with the highest validation macro-F1 score is selected for final test evaluation.

## Comparative Experiments

Experiments are organized into three main groups.

**1. CNN Baselines**

Five pretrained CNN architectures are evaluated using ordinary cross-entropy loss:

- ResNet-50
- EfficientNet-B2
- DenseNet-121
- MobileViT-XXS
- ConvNeXt-T

**2. Traditional Class-Imbalance Methods**

Different loss-based approaches are compared using the same ResNet-50 backbone:

- Cross-Entropy Loss
- Focal Loss
- Class-Balanced Loss
- Logit Adjustment
- LDAM-DRW

**3. Graph-Based Methods**

TAL-GRN is compared with:

- GCN
- GraphSAGE
- GraphSMOTE
- TAM
- ReNode

The graph-based baselines are adapted to the same mini-batch training setting to make the comparisons more consistent.

## Ablation Studies

Ablation experiments investigate how individual components contribute to the overall performance.

| Experiment | Purpose |
|---|---|
| Full TAL-GRN | Complete proposed architecture |
| Fixed-K Graph | Replaces adaptive neighborhood sizing with a fixed $k=20$ |
| Without Edge Weights | Removes learnable edge weighting and uses uniform weights |
| Without Graph Module | Removes graph construction and message passing, replacing them with an MLP |

These experiments help determine the contribution of adaptive neighborhood selection, learnable edge weights, and graph-based relational learning.

## Results

### Main Brain MRI Dataset

The proposed TAL-GRN achieved the following results in the reported main experiment:

| Evaluation Metric | Result |
|---|---:|
| Test Accuracy | **86.12%** |
| Balanced Accuracy | 85.20% |
| Macro Precision | 85.06% |
| Macro Recall | 85.20% |
| Macro F1-score | **0.8472** |
| Weighted F1-score | 0.8558 |
| MCC | 0.8575 |
| ROC-AUC | 0.9968 |

### Comparison with Graph-Based Methods

| Method | Test Accuracy |
|---|---:|
| TAL-GRN | **86.12%** |
| GCN | 83.20% |
| GraphSMOTE | 83.08% |
| GraphSAGE | 82.79% |
| TAM | 82.60% |
| ReNode | 82.07% |

TAL-GRN achieved the highest accuracy among the evaluated graph-based methods.

### Ablation Results

The complete TAL-GRN model achieved 86.12% accuracy and a 0.8472 macro F1-score.

Removing the graph module resulted in a decrease to:

- Accuracy: 78.20%
- Macro F1-score: 0.7668

This highlights the contribution of relational graph learning to the classification performance.

### Robustness Across Random Seeds

The model was also evaluated using five random seeds.

| Metric | Mean Result |
|---|---:|
| Accuracy | 85.89% |
| Accuracy Standard Deviation | 0.39% |
| Macro F1-score | 0.8443 |
| Macro F1 Standard Deviation | 0.0049 |
| ROC-AUC | 0.9968 |
| MCC | Approximately 0.8552 |

The relatively small standard deviations indicate consistent performance across the tested random seeds.

## Generalization on Additional Datasets

TAL-GRN was additionally evaluated on three medical imaging benchmarks.

| Dataset | TAL-GRN Accuracy | Baseline Accuracy |
|---|---:|---:|
| BloodMNIST | 80.62% | 75.53% |
| Skin Cancer (HAM10000) | 71.55% | 66.34% |
| Chest X-ray | 93.52% | 91.47% |

These experiments investigate whether adaptive graph learning can also improve classification across different medical imaging modalities and class distributions.

## Evaluation Metrics

The following metrics are used to evaluate classification performance:

- **Accuracy:** Overall proportion of correctly classified images.
- **Balanced Accuracy:** Average recall across classes, giving equal importance to each class.
- **Macro Precision:** Average precision across all classes.
- **Macro Recall:** Average recall across all classes.
- **Macro F1-score:** Average F1-score across classes, treating every class equally.
- **Weighted F1-score:** F1-score averaged according to class frequency.
- **MCC:** Correlation-based metric that considers the agreement between actual and predicted labels.
- **ROC-AUC:** Measures the model's ability to distinguish between classes.

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Scikit-learn
- CUDA
- Automatic Mixed Precision (AMP)

## Research Objective

The main objective of TAL-GRN is to investigate whether **learning the graph structure itself** can improve imbalanced medical image classification.

Instead of relying only on loss reweighting or a fixed neighborhood structure, TAL-GRN learns both the number of neighbors and the importance of their connections. Through adaptive graph construction and topology-aware message passing, the framework aims to use relationships between medical images to improve classification, particularly when class distributions are unequal.

---

**Note:** The reported results represent the experiments described in this project. They should not be interpreted as clinical validation or as evidence of deployment readiness.

