import os
import glob
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from collections import Counter

# ---- U-Net architecture with class-aware channels ----
class UNet(nn.Module):
    def __init__(self, num_classes=8):
        super(UNet, self).__init__()
        in_channels = 3 + num_classes

        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
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

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = CBR(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = CBR(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = CBR(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = CBR(128, 64)

        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        m = self.middle(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(m), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.final(d1))

# ---- Dataset with class-aware input ----
class HeatmapDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        self.transform = transform

        self.classes = [os.path.basename(p).split("_")[0] for p in self.image_paths]
        self.class_to_idx = {cls: i for i, cls in enumerate(sorted(set(self.classes)))}
        self.class_indices = [self.class_to_idx[c] for c in self.classes]
        self.num_classes = len(self.class_to_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        mask = (mask > 127).float()

        class_index = self.class_indices[idx]
        class_tensor = torch.zeros(self.num_classes, 224, 224)
        class_tensor[class_index, :, :] = 1.0

        img_aug = torch.cat([img, class_tensor], dim=0)
        return img_aug, mask

# ---- Class-aware sampling ----
def get_sampler(class_indices):
    class_counts = np.bincount(class_indices)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[i] for i in class_indices]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# ---- Paths and transforms ----
image_dir = "cnn_heatmap_dataset/inputs"
mask_dir = "cnn_heatmap_dataset/masks"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---- Prepare dataset ----
dataset = HeatmapDataset(image_dir, mask_dir, transform=transform)
sampler = get_sampler(dataset.class_indices)

# ---- Print sampled class distribution ----
sampled_indices = list(sampler)
sampled_classes = [dataset.classes[idx] for idx in sampled_indices]
class_counts = Counter(sampled_classes)
print("\n📊 Actual Samples Used for Training (after class-aware sampling):")
for cls, count in sorted(class_counts.items()):
    print(f"  {cls}: {count} images")

dataloader = DataLoader(dataset, batch_size=16, sampler=sampler)

# ---- Model and training ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(num_classes=dataset.num_classes).to(device)

print("Using device:", device)
print("Model on device:", next(model.parameters()).device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ---- Train the model ----
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for images, masks in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1} - Loss: {epoch_loss / len(dataloader):.4f}")

# ---- Save model ----
torch.save(model.state_dict(), "unet_heatmap_model_classaware.pth")
