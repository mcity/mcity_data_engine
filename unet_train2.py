import os
import glob
from PIL import Image
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math

class PatchEmbedding(nn.Module):
    """Patch embedding layer for transformer"""
    def __init__(self, img_size=224, patch_size=4, in_channels=11, embed_dim=64):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, embed_dim)
        x = self.norm(x)
        return x

class EfficientSelfAttention(nn.Module):
    """Efficient self-attention with reduced complexity"""
    def __init__(self, dim, num_heads=8, sr_ratio=1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class MixFFN(nn.Module):
    """Mix-FFN module"""
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, H, W):
        x = self.fc1(x)
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.fc2(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer block with efficient attention"""
    def __init__(self, dim, num_heads, mlp_ratio=4, sr_ratio=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads, sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim, int(dim * mlp_ratio), dim)

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x), H, W)
        return x

class SegFormer(nn.Module):
    """SegFormer model for semantic segmentation"""
    def __init__(self, in_channels=11, num_classes=1, img_size=224):
        super().__init__()

        # Hierarchical encoder with 4 stages
        self.patch_embed1 = PatchEmbedding(img_size, 4, in_channels, 64)
        self.patch_embed2 = PatchEmbedding(img_size//4, 2, 64, 128)
        self.patch_embed3 = PatchEmbedding(img_size//8, 2, 128, 256)
        self.patch_embed4 = PatchEmbedding(img_size//16, 2, 256, 512)

        # Transformer blocks for each stage
        self.block1 = nn.ModuleList([TransformerBlock(64, 1, 8, 8) for _ in range(2)])
        self.block2 = nn.ModuleList([TransformerBlock(128, 2, 8, 4) for _ in range(2)])
        self.block3 = nn.ModuleList([TransformerBlock(256, 4, 4, 2) for _ in range(2)])
        self.block4 = nn.ModuleList([TransformerBlock(512, 8, 4, 1) for _ in range(2)])

        # Decoder
        self.decode_head = nn.ModuleList([
            nn.Conv2d(64, 256, 1),
            nn.Conv2d(128, 256, 1),
            nn.Conv2d(256, 256, 1),
            nn.Conv2d(512, 256, 1)
        ])

        self.fusion = nn.Conv2d(256 * 4, 256, 1)
        self.classifier = nn.Conv2d(256, num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        B = x.shape[0]
        features = []

        # Stage 1
        x, H, W = self.patch_embed1(x), 56, 56
        for blk in self.block1:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        features.append(x)

        # Stage 2
        x, H, W = self.patch_embed2(x), 28, 28
        for blk in self.block2:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        features.append(x)

        # Stage 3
        x, H, W = self.patch_embed3(x), 14, 14
        for blk in self.block3:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        features.append(x)

        # Stage 4
        x, H, W = self.patch_embed4(x), 7, 7
        for blk in self.block4:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        features.append(x)

        # Decode
        decoded_features = []
        for i, feat in enumerate(features):
            feat = self.decode_head[i](feat)
            feat = F.interpolate(feat, size=(56, 56), mode='bilinear', align_corners=False)
            decoded_features.append(feat)

        x = torch.cat(decoded_features, dim=1)
        x = self.fusion(x)
        x = self.dropout(x)
        x = self.classifier(x)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        return torch.sigmoid(x)

# Alternative: DeepLabV3+ with ResNet backbone
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ implementation"""
    def __init__(self, in_channels=11, num_classes=1):
        super().__init__()

        # Backbone (simplified ResNet-like)
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),

            # Layer 1
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # Layer 2
            nn.Conv2d(256, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 512, 3, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # Layer 3 (with dilation)
            nn.Conv2d(512, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 1024, 3, 1, 2, dilation=2, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )

        self.aspp = ASPP(1024, [12, 24, 36])

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x):
        input_shape = x.shape[-2:]

        # Encode
        features = self.backbone(x)
        features = self.aspp(features)

        # Decode
        x = F.interpolate(features, size=input_shape, mode='bilinear', align_corners=False)
        x = self.decoder(x)

        return torch.sigmoid(x)

# Usage example:
#if __name__ == "__main__":
    # Choose your model
#    model = SegFormer(in_channels=11, num_classes=1)  # SegFormer
    # model = DeepLabV3Plus(in_channels=11, num_classes=1)  # DeepLabV3+

class UNet(nn.Module):
    def __init__(self, in_channels=11, out_channels=1):
        super(UNet, self).__init__()
        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.2),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        self.enc1 = CBR(in_channels, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.middle = CBR(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec4 = CBR(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = CBR(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = CBR(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = CBR(128, 64)
        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2)); e4 = self.enc4(self.pool(e3))
        m = self.middle(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(m), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.final(d1))

# ------------------ Dataset ------------------
class HeatmapDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.classes = [os.path.basename(p).split("_")[0] for p in self.image_paths]
        self.class_to_idx = {cls: i for i, cls in enumerate(sorted(set(self.classes)))}
        # ADD THIS LINE: Create reverse mapping from index to class name
        self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}
        self.class_indices = [self.class_to_idx[c] for c in self.classes]

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        img = self.transform(img) if self.transform else transforms.ToTensor()(img)
        mask = mask.resize((224, 224), Image.NEAREST)
        mask = torch.from_numpy(np.array(mask)).float().unsqueeze(0) / 255.0
        mask = (mask > 0.1).float()
        class_idx = self.class_indices[idx]
        one_hot = torch.zeros(len(self.class_to_idx), *img.shape[1:])
        one_hot[class_idx] = 1.0
        img = torch.cat([img, one_hot], dim=0)
        return img, mask, class_idx

# ------------------ Load Paths ------------------
def load_split_paths(base_dir, split):
    image_paths, mask_paths = [], []
    class_folders = sorted(os.listdir(os.path.join(base_dir, split, "images")))
    for cls in class_folders:
        img_dir = os.path.join(base_dir, split, "images", cls)
        msk_dir = os.path.join(base_dir, split, "masks", cls)
        imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        msks = sorted(glob.glob(os.path.join(msk_dir, "*.png")))
        image_paths.extend(imgs)
        mask_paths.extend(msks)
    return image_paths, mask_paths

# ------------------ Config ------------------
BASE_DIR = "resampled_balanced_split"
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
train_img, train_mask = load_split_paths(BASE_DIR, "train")
val_img, val_mask = load_split_paths(BASE_DIR, "val")

train_dataset = HeatmapDataset(train_img, train_mask, transform)
val_dataset = HeatmapDataset(val_img, val_mask, transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = UNet().to(device)
model = SegFormer(in_channels=11, num_classes=1).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

print(f"🚀 Training on {device} with {len(train_dataset)} train and {len(val_dataset)} val samples.")
print(f"📋 Classes found: {sorted(val_dataset.class_to_idx.keys())}")

# ------------------ Training ------------------
num_epochs = 14
for epoch in range(num_epochs):
    model.train(); train_loss = 0
    for images, masks, _ in tqdm(train_loader, desc=f"[Train Epoch {epoch+1}]"):
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        train_loss += loss.item()

    model.eval(); val_loss = 0; val_metrics = defaultdict(lambda: {"y_true": [], "y_pred": []})
    with torch.no_grad():
        for images, masks, class_indices in tqdm(val_loader, desc=f"[Val Epoch {epoch+1}]"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            preds = (outputs > 0.2).float()
            for i in range(images.size(0)):
                cls_idx = class_indices[i].item()
                # FIXED LINE: Use the reverse mapping to get class name
                cls_name = val_dataset.idx_to_class[cls_idx]
                val_metrics[cls_name]["y_true"] += masks[i].cpu().numpy().flatten().tolist()
                val_metrics[cls_name]["y_pred"] += preds[i].cpu().numpy().flatten().tolist()

    print(f"\n📉 Epoch {epoch+1}: Train Loss = {train_loss/len(train_loader):.4f}, Val Loss = {val_loss/len(val_loader):.4f}")
    print("📦 Seen classes in val:", sorted(list(val_metrics.keys())))
    for cls in sorted(val_metrics.keys()):
        values = val_metrics[cls]
        prec = precision_score(values["y_true"], values["y_pred"], zero_division=0)
        rec = recall_score(values["y_true"], values["y_pred"], zero_division=0)
        f1 = f1_score(values["y_true"], values["y_pred"], zero_division=0)
        print(f"📊 {cls}: Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}")

# Save model
torch.save(model.state_dict(), "segformer_model12.pth")