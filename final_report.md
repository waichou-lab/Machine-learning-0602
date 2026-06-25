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

![UML Class Diagram](../main/Procedure2.png)


### 3.2 Module Descriptions

| Module | Description |
|--------|-------------|
| **AttentionModule** | Abstract base class defining the interface for all attention mechanisms. Subclasses must implement `forward(x)`. |
| **SENet** | Channel attention via squeeze-and-excitation. Applies global pooling then learns channel weights through a two-layer MLP. |
| **CBAM** | Channel + Spatial attention. Applies channel attention, then spatial attention sequentially. |
| **NonLocal** | Full spatial self-attention. Computes pairwise affinities between all positions. High O(N²) complexity. |
| **CCNet** | Criss-Cross Attention. Computes attention only along horizontal and vertical directions. Low O(N) complexity. |
| **CNNBackbone** | Feature extractor. Returns multi-scale feature maps. |
| **SegmentationNet** | Complete segmentation model. Combines backbone, attention module, and segmentation head. |

### 3.3 Design Principles

The architecture follows these software engineering principles:

1. **Open-Closed Principle**: New attention mechanisms can be added by subclassing `AttentionModule` without modifying existing code.

2. **Modularity**: Each component has a single, well-defined responsibility.

3. **Reusability**: The same attention modules can be used for both classification and segmentation tasks.

4. **Testability**: The modular design enables independent testing of each component.


## 4. Implementation

### 4.1 CCNet Module (PyTorch)

```python
class CrissCrossAttention(nn.Module):
    """Criss-Cross Attention Module for CCNet.

    Complexity: O(N) where N = H * W.
    """
    def __init__(self, in_channels, reduction=2):
        super().__init__()
        key_channels = in_channels // reduction
        value_channels = in_channels // reduction

        self.query_conv = nn.Conv2d(in_channels, key_channels, 1)
        self.key_conv = nn.Conv2d(in_channels, key_channels, 1)
        self.value_conv = nn.Conv2d(in_channels, value_channels, 1)
        self.out_conv = nn.Conv2d(value_channels, in_channels, 1)
        nn.init.constant_(self.out_conv.weight, 0)
        nn.init.constant_(self.out_conv.bias, 0)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n, c, h, w = x.size()

        # Embeddings
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)

        # Horizontal attention: (N*H, W, W)
        q_h = query.permute(0, 2, 1, 3).contiguous().view(n * h, -1, w)
        k_h = key.permute(0, 2, 1, 3).contiguous().view(n * h, -1, w)
        v_h = value.permute(0, 2, 1, 3).contiguous().view(n * h, -1, w)
        energy_h = torch.bmm(q_h.permute(0, 2, 1), k_h)
        attn_h = self.softmax(energy_h)
        out_h = torch.bmm(v_h, attn_h.permute(0, 2, 1))
        out_h = out_h.view(n, h, -1, w).permute(0, 2, 1, 3)

        # Vertical attention: (N*W, H, H)
        q_v = query.permute(0, 3, 1, 2).contiguous().view(n * w, -1, h)
        k_v = key.permute(0, 3, 1, 2).contiguous().view(n * w, -1, h)
        v_v = value.permute(0, 3, 1, 2).contiguous().view(n * w, -1, h)
        energy_v = torch.bmm(q_v.permute(0, 2, 1), k_v)
        attn_v = self.softmax(energy_v)
        out_v = torch.bmm(v_v, attn_v.permute(0, 2, 1))
        out_v = out_v.view(n, w, -1, h).permute(0, 2, 3, 1)

        # Combine + output projection + residual
        out = out_h + out_v
        out = self.out_conv(out)
        return x + out
```

### 4.2 Integration Example
```
class SegmentationNet(nn.Module):
    def __init__(self, attention_type='ccnet'):
        super().__init__()
        self.backbone = CNNBackbone()
        self.attention = CrissCrossAttention(backbone_channels) if attention_type == 'ccnet' else None
        self.head = SegmentationHead(backbone_channels, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        if self.attention is not None:
            x = self.attention(x)
        x = self.head(x)
        return x
```
###4.3 Component Dependencies
SegmentationNet
    ├── CNNBackbone
    │   └── layers: Sequential (Conv2d, BatchNorm, ReLU, MaxPool)
    ├── AttentionModule (可选)
    │   └── CCNet / SENet / CBAM / NonLocal
    └── SegmentationHead
        ├── Conv2d (256 → num_classes)
        └── Upsample (scale_factor=8, mode='bilinear')
