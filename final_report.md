# CCNet: Criss-Cross Attention Architecture Study

## 1. Introduction

### 1.1 Research Questions

This study focuses on two core questions:

1. **Can CCNet's criss-cross attention maintain competitive accuracy with Non-local while reducing complexity from O(N²) to O(N)?**

2. **Does CCNet demonstrate clear advantages over SENet (pure channel attention) in global context modeling?**

### 1.2 Why Linear-Complexity Attention Matters

**Non-local Limitations:**
- Attention matrix complexity: O(N²), where N = H × W (pixel count)
- Memory explosion at high resolution (e.g., Cityscapes: 1024×2048 → N ≈ 2 million)
- Not suitable for real-time deployment

**CCNet Solution:**
- Criss-Cross Attention computes attention **only in horizontal and vertical directions**
- Complexity: O(N)
- Two recursive passes capture full global context


## 2. CCNet Architecture

### 2.1 Core Concept: Criss-Cross Attention

The fundamental innovation of CCNet is its **Criss-Cross Attention** module, which decomposes full pairwise attention into horizontal and vertical components:

Full Non-local: Every pixel attends to every other pixel → O(N²)

CCNet: Each pixel attends to pixels in same row and column → O(N)
Two passes achieve equivalent global coverage


### 2.2 Mathematical Formulation

**Standard Non-local Attention:**

$$
Y_i = \frac{1}{\mathcal{C}(X)} \sum_j f(X_i, X_j) g(X_j)
$$

**Criss-Cross Attention:**

**Horizontal Attention:**
For each position (h, w), compute attention with all positions in the same row:

$$
Y^H_{h,w} = \sum_{w'=1}^{W} \text{Softmax}_w \left( \theta(X_{h,w})^T \phi(X_{h,w'}) \right) \cdot g(X_{h,w'})
$$

**Vertical Attention:**
Similarly, compute attention with all positions in the same column:

$$
Y^V_{h,w} = \sum_{h'=1}^{H} \text{Softmax}_h \left( \theta(X_{h,w})^T \phi(X_{h',w}) \right) \cdot g(X_{h',w})
$$

**Combined Output:**

$$
Y = Y^H + Y^V
$$

### 2.3 Two-Pass Information Propagation

A single criss-cross pass only allows each pixel to aggregate information from its row and column. However, after **two passes**, information can flow from any pixel to any other pixel:

- **Pass 1**: Pixel A receives information from its row and column
- **Pass 2**: Pixel A can now receive information from *those* pixels' rows and columns, effectively covering the entire feature map
- 
Pass 1: A → (row, column) neighbors
Pass 2: A → (row, column) neighbors of neighbors → full coverage

### 2.4 Architecture Diagram

![Architecture Diagram](../main/Procedure.png)



### 2.5 Comparison Summary

| Aspect | Non-local | CCNet |
|--------|-----------|-------|
| Attention Computation | Full pairwise | Horizontal + vertical only |
| Time Complexity | O(N²) | O(N) |
| Memory Complexity | O(N²) | O(N) |
| Global Context | One pass | Two passes needed |
| Suitability for High Resolution | Poor | Good |


## 3. UML Architecture Design

### 3.1 UML Class Diagram



