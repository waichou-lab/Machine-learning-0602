# ============================================================
# Cityscapes 語意分割訓練：UML 架構驗證（快速版）
# 目的：產出訓練曲線圖供簡報使用
# 訓練時間：約 30 分鐘（1 epoch）
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time
import random
from PIL import Image
import glob
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用裝置: {device}")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

DATA_ROOT = r"C:\cityscapes"
NUM_CLASSES = 19
BATCH_SIZE = 2
EPOCHS = 1  # 只跑 1 個 epoch，快速產出圖片
INPUT_SIZE = (256, 512)  # 降低解析度加速

print(f"資料集路徑: {DATA_ROOT}")

# ---------- 資料集 ----------
class CityscapesDataset(torch.utils.data.Dataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        img_dir = os.path.join(root, 'leftImg8bit', split)
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, '*', '*.png')))

        self.label_paths = []
        for img_path in self.img_paths:
            basename = os.path.basename(img_path)
            city = os.path.basename(os.path.dirname(img_path))
            label_basename = basename.replace('_leftImg8bit', '_gtFine_labelIds')
            label_path = os.path.join(root, 'gtFine', split, city, label_basename)
            self.label_paths.append(label_path)

        valid_pairs = []
        for img, lbl in zip(self.img_paths, self.label_paths):
            if os.path.exists(lbl):
                valid_pairs.append((img, lbl))

        self.img_paths = [v[0] for v in valid_pairs]
        self.label_paths = [v[1] for v in valid_pairs]
        print(f"載入 {split}: {len(self.img_paths)} 張圖片")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert('RGB')
        label = Image.open(self.label_paths[idx])

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

# ---------- 資料預處理 ----------
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

val_transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

target_transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE, interpolation=Image.NEAREST),
    transforms.ToTensor(),
])

# ---------- 載入資料 ----------
print("\n載入訓練資料集...")
train_dataset = CityscapesDataset(
    root=DATA_ROOT,
    split='train',
    transform=train_transform,
    target_transform=target_transform
)

print("\n載入驗證資料集...")
val_dataset = CityscapesDataset(
    root=DATA_ROOT,
    split='val',
    transform=val_transform,
    target_transform=target_transform
)

if len(train_dataset) == 0:
    print("❌ 資料集為空！")
    exit()

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

print(f"\n✅ 訓練: {len(train_dataset)} 張, 驗證: {len(val_dataset)} 張")

# ---------- 注意力模組 ----------
class CrissCrossAttention(nn.Module):
    def __init__(self, in_channels, reduction=2):
        super().__init__()
        key_channels = in_channels // reduction
        value_channels = in_channels // reduction
        self.query_conv = nn.Conv2d(in_channels, key_channels, 1)
        self.key_conv = nn.Conv2d(in_channels, key_channels, 1)
        self.value_conv = nn.Conv2d(in_channels, value_channels, 1)
        self.out_conv = nn.Conv2d(value_channels, in_channels, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n, c, h, w = x.size()
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)

        q_h = query.permute(0, 2, 1, 3).contiguous().view(n * h, -1, w)
        k_h = key.permute(0, 2, 1, 3).contiguous().view(n * h, -1, w)
        v_h = value.permute(0, 2, 1, 3).contiguous().view(n * h, -1, w)
        energy_h = torch.bmm(q_h.permute(0, 2, 1), k_h)
        attn_h = self.softmax(energy_h)
        out_h = torch.bmm(v_h, attn_h.permute(0, 2, 1))
        out_h = out_h.view(n, h, -1, w).permute(0, 2, 1, 3)

        q_v = query.permute(0, 3, 1, 2).contiguous().view(n * w, -1, h)
        k_v = key.permute(0, 3, 1, 2).contiguous().view(n * w, -1, h)
        v_v = value.permute(0, 3, 1, 2).contiguous().view(n * w, -1, h)
        energy_v = torch.bmm(q_v.permute(0, 2, 1), k_v)
        attn_v = self.softmax(energy_v)
        out_v = torch.bmm(v_v, attn_v.permute(0, 2, 1))
        out_v = out_v.view(n, w, -1, h).permute(0, 2, 3, 1)

        out = out_h + out_v
        out = self.out_conv(out)
        return x + out

class SEModule(nn.Module):
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

# ---------- 分割網路（深層版） ----------
class SegmentationNet(nn.Module):
    def __init__(self, num_classes=19, attention_type='baseline'):
        super().__init__()
        # 較深的骨幹
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.backbone_channels = 512

        if attention_type == 'ccnet':
            self.attention = CrissCrossAttention(self.backbone_channels)
        elif attention_type == 'senet':
            self.attention = SEModule(self.backbone_channels)
        else:
            self.attention = nn.Identity()

        self.head = nn.Sequential(
            nn.Conv2d(self.backbone_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.attention(x)
        x = self.head(x)
        return x

# ---------- mIoU 計算 ----------
def compute_miou(pred, target, num_classes=19):
    if pred.dim() == 4:
        pred = pred.argmax(dim=1)
    if target.dim() == 4:
        target = target.squeeze(1)

    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append((intersection / union).item())
    return np.nanmean(ious)

# ---------- 訓練函數 ----------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, labels in tqdm(loader, desc='訓練', leave=False):
        images, labels = images.to(device), labels.to(device)
        labels = labels.squeeze(1).long()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    total_iou = 0
    total_loss = 0
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='評估', leave=False):
            images, labels = images.to(device), labels.to(device)
            labels = labels.squeeze(1).long()
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            for i in range(images.size(0)):
                iou = compute_miou(outputs[i:i+1], labels[i:i+1])
                if not np.isnan(iou):
                    total_iou += iou
    return total_loss / len(loader), total_iou / len(loader.dataset)

# ---------- 主實驗 ----------
def run_experiment(attention_type, epochs=EPOCHS):
    print(f"\n{'='*50}")
    print(f"訓練: {attention_type.upper()}")
    print(f"{'='*50}")

    model = SegmentationNet(attention_type=attention_type).to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"參數量: {params:.2f} M")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    history = {'train_loss': [], 'val_loss': [], 'val_miou': []}
    best_miou = 0

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_miou = evaluate(model, val_loader, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_miou'].append(val_miou)

        if val_miou > best_miou:
            best_miou = val_miou

        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | mIoU: {val_miou:.4f} | Best: {best_miou:.4f}")

    # 推論速度測試
    model.eval()
    dummy = torch.randn(1, 3, INPUT_SIZE[0], INPUT_SIZE[1]).to(device)
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy)
    torch.cuda.synchronize()
    inference_time = (time.time() - start) / 50 * 1000

    print(f"\n{attention_type.upper()} 完成!")
    print(f"  最佳 mIoU: {best_miou*100:.2f}%")
    print(f"  參數量: {params:.2f} M")
    print(f"  推論時間: {inference_time:.2f} ms/張")

    return {
        'name': attention_type,
        'params': params,
        'best_miou': best_miou,
        'inference_time': inference_time,
        'history': history
    }

# ============================================================
# 執行（1 Epoch 快速版）
# ============================================================
if __name__ == "__main__":
    print("="*60)
    print("Cityscapes UML 架構驗證（1 Epoch 快速版）")
    print(f"資料集: {DATA_ROOT}")
    print("="*60)

    experiments = ['baseline', 'senet', 'ccnet']
    results = {}

    for exp in experiments:
        results[exp] = run_experiment(exp, epochs=1)
        torch.cuda.empty_cache()
        gc.collect()

    # 輸出比較表
    print("\n" + "="*60)
    print("效能比較總表 (1 Epoch 快速驗證)")
    print("="*60)
    print(f"{'Model':<12} {'Params (M)':<12} {'mIoU (%)':<12} {'Time (ms)':<12}")
    print("-"*60)
    for name, r in results.items():
        print(f"{name:<12} {r['params']:<12.2f} {r['best_miou']*100:<12.2f} {r['inference_time']:<12.2f}")

    # 繪圖
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for name, r in results.items():
        plt.plot(r['history']['train_loss'], label=f'{name} (train)')
        plt.plot(r['history']['val_loss'], linestyle='--', label=f'{name} (val)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.title('Training / Validation Loss (1 Epoch)')

    plt.subplot(1, 2, 2)
    for name, r in results.items():
        plt.plot(r['history']['val_miou'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.legend()
    plt.grid(True)
    plt.title('Validation mIoU (1 Epoch)')

    plt.tight_layout()
    plt.savefig('cityscapes_quick_validation.png', dpi=150)
    plt.show()

    print("\n✅ 快速驗證完成！")
    print("   圖片已儲存: cityscapes_quick_validation.png")
    print("   請將此圖片放入簡報第 13-15 頁。")