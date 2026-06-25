# CCNet: Criss-Cross Attention for Efficient Semantic Segmentation
## A Comprehensive Architecture Study and Implementation Report

---

## Abstract

Semantic segmentation requires modeling long-range spatial dependencies, but standard non-local attention mechanisms suffer from quadratic computational complexity with respect to image resolution. This report presents a comprehensive study of CCNet (Criss-Cross Attention Network), which addresses this limitation through a novel attention mechanism that reduces complexity from $O(N^2)$ to $O(N)$ while maintaining competitive accuracy. We provide a detailed analysis of the architecture, its mathematical formulation, implementation details, and a comparison with existing methods including Non-local Networks and SENet. UML class diagrams and complete PyTorch implementations are presented, along with experimental results on CIFAR-10 and Cityscapes datasets. The report concludes with identified challenges and future research directions.

---

## 1. Introduction

### 1.1 Background

Semantic segmentation is a fundamental task in computer vision that assigns a semantic label to every pixel in an image. Accurate segmentation requires understanding both local texture patterns and global contextual relationships. For example, to segment a "car" correctly, the network must recognize that it is typically located on a "road" and has a distinctive shape—information that spans across the entire image.

Traditional Convolutional Neural Networks (CNNs) are inherently limited to local receptive fields, making it difficult to capture long-range dependencies. Stacking deeper layers partially addresses this but introduces vanishing gradient problems and still fails to model explicit global relationships.

**Non-local Networks** (Wang et al., 2018) introduced self-attention mechanisms to CNNs, enabling direct modeling of relationships between distant pixels. However, the standard non-local block computes pairwise affinities between all pixel positions, resulting in $O(N^2)$ complexity where $N = H \times W$ is the number of spatial positions. For a 1024×2048 Cityscapes image, $N \approx 2$ million, making non-local attention computationally prohibitive—requiring over 4 trillion pairwise computations.

### 1.2 Motivation

CCNet (Huang et al., 2019) was proposed to address this scalability challenge. The key insight is that **full pairwise attention is not necessary** to capture global context. Instead, CCNet introduces a **Criss-Cross Attention** mechanism that only computes attention along horizontal and vertical directions within a single pass. By stacking two attention layers, information can propagate across the entire feature map, achieving effectively similar global context modeling to non-local, but with $O(N)$ complexity.

This report provides a complete examination of CCNet's architecture, implementation, and comparison with alternatives, fulfilling the requirement for a comprehensive understanding of this important advancement.

### 1.3 Research Questions

This study is guided by the following questions:

1. How does CCNet's Criss-Cross Attention achieve $O(N)$ complexity while maintaining Non-local-like accuracy?
2. What are the architectural trade-offs between CCNet, Non-local, and SENet?
3. How can CCNet be implemented in a modular, extensible architecture?

---

## 2. CCNet Architecture

### 2.1 Core Concept: Criss-Cross Attention

The fundamental innovation of CCNet lies in its **Criss-Cross Attention** module. Instead of computing attention between every pair of positions (which is $O(N^2)$), the Criss-Cross Attention computes attention **only in the horizontal and vertical directions**.

This is based on the observation that global context can be captured effectively through a two-step propagation process:

1. **First criss-cross pass**: Each pixel aggregates information from all pixels in the same row and column.
2. **Second criss-cross pass**: This enables information to flow from any pixel to any other pixel through intermediate paths.

The result is global receptive field coverage with only $O(N)$ operations per pass.

### 2.2 Mathematical Formulation

#### 2.2.1 Standard Non-local Attention (Baseline)

For a feature map $X \in \mathbb{R}^{C \times H \times W}$, the standard non-local block computes:

$$
Y_i = \frac{1}{\mathcal{C}(X)} \sum_j f(X_i, X_j) g(X_j)
$$

where:
- $f(X_i, X_j) = e^{\theta(X_i)^T \phi(X_j)}$ is the pairwise affinity function
- $\theta$ and $\phi$ are 1x1 convolution embeddings
- $g$ is a value embedding
- $\mathcal{C}(X)$ is a normalization factor

**Computational cost**: $O(N^2)$ where $N = H \times W$

#### 2.2.2 Criss-Cross Attention

The Criss-Cross Attention decomposes the full pairwise attention into **horizontal** and **vertical** components:

**Horizontal attention:**
For each position $(h, w)$, compute attention with all positions in the same row:

$$
Y^H_{h,w} = \sum_{w'=1}^{W} \text{Softmax}_w \left( \theta(X_{h,w})^T \phi(X_{h,w'}) \right) \cdot g(X_{h,w'})
$$

**Vertical attention:**
Similarly, compute attention with all positions in the same column:

$$
Y^V_{h,w} = \sum_{h'=1}^{H} \text{Softmax}_h \left( \theta(X_{h,w})^T \phi(X_{h',w}) \right) \cdot g(X_{h',w})
$$

**Combined output:**
$Y = Y^H + Y^V$

**Computational cost**: $O(HW) = O(N)$

### 2.3 Two-Pass Information Propagation

A single criss-cross pass only allows each pixel to aggregate information from its row and column. However, after two passes, information from any pixel can reach any other pixel:

- **Pass 1**: Pixel $A$ receives information from its row and column.
- **Pass 2**: Pixel $A$ can now receive information from *those* pixels' rows and columns, effectively covering the entire feature map.

This is analogous to two-hop message passing in a graph, achieving global context with only local computations.

### 2.4 Architecture Diagram

![Architecture Diagram]
