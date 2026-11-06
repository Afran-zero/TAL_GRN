# TAL-GRN: Topology-Adaptive Learning Graph Refinement Network

## Overview

In this project, I developed TAL-GRN, a graph-based deep learning framework for multi-class image classification. The main idea is to improve graph learning by introducing adaptive neighborhood selection and learned edge weights.

Traditional graph-based methods often construct a graph using a fixed number of nearest neighbors. However, a fixed neighborhood size may not always capture the relationships between samples effectively. To address this limitation, TAL-GRN uses an adaptive graph structure that can be refined during training.

The framework combines a pretrained ResNet-50 model for feature extraction with graph-based learning for capturing relationships between image samples. My main focus is to investigate how adaptive graph construction and edge refinement affect classification performance.

## Key Contributions

- Used a pretrained ResNet-50 backbone to extract image features.
- Represented each image using a 2048-dimensional feature vector.
- Constructed an initial graph using cosine similarity.
- Introduced adaptive neighborhood selection instead of relying only on a fixed k-NN structure.
- Incorporated learned edge weights to represent the importance of connections between samples.
- Applied periodic graph refinement during training.
- Evaluated the framework using multiple classification metrics.

## Dataset

The experiments were conducted on the TAL-GRN preprocessed dataset, which contains 34,192 images distributed across 40 classes.

| Dataset split | Number of images |
| ------------- | ---------------: |
| Training      |           23,234 |
| Validation    |            4,085 |
| Testing       |            6,873 |
| **Total**     |       **34,192** |

**Dataset details:**

- Number of classes: 40
- Input image size: 224 × 224
- Feature dimension: 2048

The dataset contains the following CSV files:

```text
shuffled_csvs/
├── train.csv
├── val.csv
├── test.csv
└── label_map.csv
```

The CSV files contain the image information and corresponding class labels. The dataset is stored separately and is not included in this repository.

## Proposed Architecture

The overall architecture of TAL-GRN consists of image feature extraction, graph construction, adaptive neighborhood selection, edge weight learning, graph refinement, and final classification.

### Overall Workflow

```text
Input Images
     |
     v
Image Preprocessing
     |
     v
Pretrained ResNet-50
     |
     v
2048-D Feature Vectors
     |
     v
Cosine Similarity
     |
     v
Initial k-NN Graph
     |
     v
Adaptive Neighborhood Selection
     |
     v
Learned Edge Weights
     |
     v
Graph Refinement
     |
     v
Graph Neural Network
     |
     v
40-Class Classification
```

### 1. Feature Extraction

I used a pretrained ResNet-50 as the feature extraction backbone. Each input image is passed through the network to obtain a 2048-dimensional feature representation.

The feature extraction process is expressed as:

$$
h_i = \text{Backbone}(x_i)
$$

Here, $x_i$ represents the input image, and $h_i$ represents its extracted feature vector.

The ResNet-50 backbone remains frozen during graph learning, so the extracted features are used as input to the graph-based model.

### 2. Initial Graph Construction

After extracting the image features, cosine similarity is calculated between the feature vectors to measure the similarity between different samples.

$$
s\_{ij} =
\frac{h_i^\top h_j}
{|h_i|\_2 |h_j|\_2}
$$

Here, $s\_{ij}$ represents the similarity between samples $i$ and $j$. A higher value indicates that their feature representations are more similar.

Based on these similarity scores, an initial k-nearest-neighbor graph is constructed. Each node represents an image sample, while the edges represent relationships between similar samples.

### 3. Adaptive Neighborhood Selection

One of the main components of TAL-GRN is adaptive neighborhood selection.

In a conventional k-NN graph, every node is connected to a fixed number of nearest neighbors. However, the same neighborhood size may not be equally suitable for all samples.

To address this issue, TAL-GRN introduces an adaptive neighborhood strategy that allows the graph structure to change during training.

Conceptually, the selected neighborhood can be represented as:

$$
\mathcal{N}*k(i) = \text{TopK}(s*{ij})
$$

Here, $\mathcal{N}\_k(i)$ represents the selected neighbors of node $i$, based on their similarity scores.

This adaptive strategy allows the model to learn more flexible relationships between samples instead of depending entirely on a fixed graph structure.

### 4. Learned Edge Weights

Another important component is learned edge weighting.

In the initial graph, connections between samples are determined using feature similarity. However, all connections may not contribute equally to the classification process.

TAL-GRN incorporates learned edge weights to represent the importance of different connections.

The weighted adjacency matrix can be expressed as:

$$
A\_{ij} =
\begin{cases}
w\_{ij}, & j \in \mathcal{N}(i) \\
0, & \text{otherwise}
\end{cases}
$$

Here, $A\_{ij}$ represents the connection between two nodes, and $w\_{ij}$ represents the corresponding edge weight.

This allows the graph model to learn which neighboring relationships are more useful during classification.

### 5. Graph Refinement

The graph structure is not kept completely fixed throughout training. TAL-GRN periodically refines the graph by updating its neighborhood relationships and edge weights.

This refinement process helps the model adjust the graph structure as learning progresses. The purpose is to improve the quality of connections between samples and provide more useful information to the graph neural network.

### 6. Graph Neural Network

After graph construction and refinement, the graph neural network learns from both image features and relationships between connected samples.

The graph convolution operation can be represented as:

$$
H^{(l+1)} =
\sigma\left(
\tilde{D}^{-\frac{1}{2}}
\tilde{A}
\tilde{D}^{-\frac{1}{2}}
H^{(l)}W^{(l)}
\right)
$$

Here:

- $H^{(l)}$ represents the node features at layer $l$.
- $\tilde{A}$ represents the adjacency matrix with self-connections.
- $\tilde{D}$ represents the degree matrix.
- $W^{(l)}$ represents trainable weights.
- $\sigma$ represents the activation function.

Through graph convolution, information from neighboring samples is combined to learn improved feature representations.

### 7. Classification

The final graph representations are passed to a classification layer to predict one of the 40 classes.

To train the classification layer, I used the cross-entropy loss function. It measures the difference between the actual class labels and the probabilities predicted by the model.

The cross-entropy loss is calculated as:

$$
\mathcal{L}_{CE} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C} y_{ic}\log(p_{ic})
$$

Here:

- $N$ represents the total number of samples.
- $C$ represents the number of classes.
- $y_{ic}$ represents the true label of sample $i$ for class $c$.
- $p_{ic}$ represents the predicted probability of sample $i$ belonging to class $c$.

The goal is to minimize the cross-entropy loss so that the predicted probabilities become closer to the actual class labels.

## Experimental Configuration

The main experimental configuration is presented below.

| Parameter                 |          Value |
| ------------------------- | -------------: |
| Backbone                  |      ResNet-50 |
| Feature dimension         |           2048 |
| Image size                |      224 × 224 |
| Number of classes         |             40 |
| GNN hidden dimension      |            512 |
| GNN layers                |              3 |
| Base neighborhood size    |              5 |
| Neighborhood parameter    |             10 |
| Edge threshold            |            0.3 |
| Graph refinement interval | Every 2 epochs |
| Batch size                |             64 |
| Epochs                    |             30 |
| Learning rate             |         0.0001 |
| Weight decay              |         0.0001 |
| Random seed               |             42 |

## Experimental Results

The full TAL-GRN experiment achieved an accuracy of **86.12%**, showing an improvement over the GCN baseline.

| Evaluation Metric |     Result |
| ----------------- | ---------: |
| Accuracy          | **86.12%** |
| Macro Precision   |     0.8506 |
| Macro Recall      |     0.8520 |
| Macro F1-score    |     0.8472 |
| Weighted F1-score |   0.855... |

Compared with the GCN baseline, TAL-GRN achieved a **2.92 percentage-point improvement in accuracy**.

This improvement suggests that adaptive neighborhood selection and learned edge weights can contribute to better graph representations and classification performance.

### Confusion Matrix Analysis

The classification performance was further analyzed using a 40 × 40 confusion matrix.

Some important observations from the results are:

- 31 out of 40 classes achieved a recall of at least 0.80.
- Adenomyosis, Sjögren's Syndrome, Balloon Cell Cortical Dysplasia, and Optic Glioma achieved recall above 0.95.
- One of the major confusion patterns was observed between Acute Unilateral Cerebellitis in HIV and Acute Cerebellitis in HIV.

Overall, the results show that TAL-GRN performs well across many classes, although distinguishing between some visually similar classes remains challenging.

## Technologies Used

- Python
- PyTorch
- Torchvision
- PyTorch Geometric
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Limitations and Future Work

Although TAL-GRN shows promising classification performance, there are still some areas that can be explored further.

Future work may include evaluating the framework on additional datasets, comparing it with other graph learning methods, investigating different neighborhood selection strategies, and conducting more extensive multi-seed experiments.

Further investigation of computational complexity and scalability would also help understand the framework's performance on larger datasets.

## Conclusion

In this project, I explored adaptive graph learning for multi-class image classification through TAL-GRN. By combining ResNet-50 feature extraction with adaptive neighborhood selection, learned edge weights, and periodic graph refinement, the framework aims to learn more meaningful relationships between image samples.

The experimental results show a 2.92 percentage-point accuracy improvement over the GCN baseline, with an overall accuracy of 86.12%. These findings support further investigation of adaptive graph structures for image classification.

##
