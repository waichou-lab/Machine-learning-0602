# ============================================================
# UML Architecture Validation on CIFAR-10
# Purpose: Verify class design (CNNBackbone, AttentionModule)
# Output: training curves (uml_training_results.png)
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import random
import os

# ---------- 固定隨機種子 ----------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# 1. UML Class Definitions
# ============================================================

# ---------- 1.1 AttentionModule (abstract) ----------
class AttentionModule(nn.Module):
    def forward(self, x):
        raise NotImplementedError

# ---------- 1.2 SENet ----------
class SENet(AttentionModule):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# ---------- 1.3 ChannelAttention (for CBAM) ----------
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return self.sigmoid(out).view(b, c, 1, 1)

# ---------- 1.4 SpatialAttention (for CBAM) ----------
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(concat))

# ---------- 1.5 CBAM ----------
class CBAM(AttentionModule):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = x * self.channel_attn(x)
        x = x * self.spatial_attn(x)
        return x

# ---------- 1.6 NonLocal ----------
class NonLocal(AttentionModule):
    def __init__(self, channels, reduction=2, use_pool=True):
        super().__init__()
        inter_channels = channels // reduction
        self.theta = nn.Conv2d(channels, inter_channels, 1)
        self.phi = nn.Conv2d(channels, inter_channels, 1)
        self.g = nn.Conv2d(channels, inter_channels, 1)
        self.W = nn.Conv2d(inter_channels, channels, 1)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)
        self.pool = nn.MaxPool2d(2) if use_pool else None

    def forward(self, x):
        n, c, h, w = x.size()
        if self.pool is not None:
            x_pool = self.pool(x)
            hp, wp = x_pool.shape[2], x_pool.shape[3]
        else:
            x_pool = x
            hp, wp = h, w

        g_x = self.g(x_pool).view(n, -1, hp * wp).permute(0, 2, 1)
        theta_x = self.theta(x_pool).view(n, -1, hp * wp)
        phi_x = self.phi(x_pool).view(n, -1, hp * wp)

        f = torch.matmul(theta_x.permute(0, 2, 1), phi_x)
        f = torch.softmax(f, dim=-1)

        y = torch.matmul(f, g_x)
        y = y.permute(0, 2, 1).contiguous().view(n, -1, hp, wp)
        y = self.W(y)

        if self.pool is not None:
            y = nn.functional.interpolate(y, size=(h, w), mode='bilinear', align_corners=False)
        return x + y

# ---------- 1.7 CNNBackbone ----------
class CNNBackbone(nn.Module):
    def __init__(self, input_channels=3, base_channels=32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels*2, 3, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels*2, base_channels*4, 3, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels*4, base_channels*8, 3, padding=1),
            nn.BatchNorm2d(base_channels*8),
            nn.ReLU(inplace=True),
        )
        self.output_channels = {
            'layer1': base_channels,
            'layer2': base_channels*2,
            'layer3': base_channels*4,
            'layer4': base_channels*8,
        }

    def forward(self, x):
        x1 = self.layers[0:3](x)
        x2 = self.layers[3:6](x1)
        x3 = self.layers[6:9](x2)
        x4 = self.layers[9:](x3)
        return x1, x2, x3, x4

# ---------- 1.8 AttentionCNN ----------
class AttentionCNN(nn.Module):
    def __init__(self, attention_module=None, attention_kwargs=None, num_classes=10):
        super().__init__()
        self.backbone = CNNBackbone()
        self.attention_modules = nn.ModuleDict()
        self.attn_type = 'baseline'
        if attention_module is not None:
            self.attn_type = attention_module.__name__
            for name, ch in self.backbone.output_channels.items():
                kwargs = attention_kwargs.copy() if attention_kwargs else {}
                kwargs['channels'] = ch
                self.attention_modules[name] = attention_module(**kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        l1, l2, l3, l4 = self.backbone(x)
        if 'layer1' in self.attention_modules:
            l1 = self.attention_modules['layer1'](l1)
        if 'layer2' in self.attention_modules:
            l2 = self.attention_modules['layer2'](l2)
        if 'layer3' in self.attention_modules:
            l3 = self.attention_modules['layer3'](l3)
        if 'layer4' in self.attention_modules:
            l4 = self.attention_modules['layer4'](l4)
        x = self.avgpool(l4)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ============================================================
# 2. Data Loading
# ============================================================
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

# ============================================================
# 3. Training Utilities
# ============================================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, pred = outputs.max(1)
        correct += pred.eq(targets).sum().item()
        total += targets.size(0)
    return total_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * images.size(0)
            _, pred = outputs.max(1)
            correct += pred.eq(targets).sum().item()
            total += targets.size(0)
    return total_loss / total, correct / total

# ============================================================
# 4. Main Experiment
# ============================================================
def run_experiment(attention_module=None, attention_kwargs=None, epochs=10):
    model_name = attention_module.__name__ if attention_module else 'baseline'
    print(f"\n{'='*50}\nTraining: {model_name}\n{'='*50}")
    model = AttentionCNN(attention_module, attention_kwargs, num_classes=10).to(device)
    params = count_parameters(model)
    print(f"Parameters: {params:.2f} M")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, epochs+1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print(f"Epoch {epoch:2d}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    best_acc = max(history['val_acc'])
    return model_name, params, best_acc, history

# ============================================================
# 5. Run All Models and Plot
# ============================================================
if __name__ == "__main__":
    models_config = [
        ('Baseline', None, None),
        ('SENet', SENet, {'reduction': 16}),
        ('CBAM', CBAM, {'reduction': 16, 'spatial_kernel': 7}),
        ('NonLocal', NonLocal, {'reduction': 2, 'use_pool': True}),
    ]

    results = {}
    for name, attn_module, kwargs in models_config:
        m_name, params, best_acc, hist = run_experiment(attn_module, kwargs, epochs=10)
        results[name] = {'params': params, 'best_acc': best_acc, 'history': hist}

    # ----- 輸出比較表 -----
    print("\n" + "="*60)
    print("Performance Summary (10 epochs)")
    print("="*60)
    print(f"{'Model':<12} {'Params (M)':<12} {'Best Val Acc (%)':<18}")
    for name, r in results.items():
        print(f"{name:<12} {r['params']:<12.2f} {r['best_acc']*100:<18.2f}")

    # ----- 繪製曲線 -----
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    for name, r in results.items():
        plt.plot(r['history']['val_acc'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    for name, r in results.items():
        plt.plot(r['history']['val_loss'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('uml_training_results.png', dpi=150)
    plt.show()

    print("\n✅ Done! Image saved as 'uml_training_results.png'")