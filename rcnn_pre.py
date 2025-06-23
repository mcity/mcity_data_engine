import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torchvision.ops import box_convert
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt

# Load dataframe
df = pd.read_csv("resnet_keypoints.csv")
class_names = sorted(df['class'].unique())
class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

# Dataset class for Keypoint R-CNN
class KeypointRCNNDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['image_path']).convert("RGB")
        image = F.to_tensor(image)
        if self.transform:
            image = self.transform(image)

        w, h = image.shape[2], image.shape[1]
        keypoint = torch.tensor([[row['x'] * 224.0, row['y'] * 224.0, 2.0]], dtype=torch.float32)  # (
        bbox = torch.tensor([0.0, 0.0, float(w), float(h)], dtype=torch.float32)
        label = torch.tensor(class_to_idx[row['class']], dtype=torch.int64)

        target = {
            'boxes': bbox.unsqueeze(0),
            'labels': label.unsqueeze(0),
            'keypoints': keypoint.unsqueeze(0),
            'image_id': torch.tensor([idx])
        }

        return image, target, row['class']

# Split data
df_train = df[df['split'] == 'train']
df_val = df[df['split'] == 'val']

train_dataset = KeypointRCNNDataset(df_train)
val_dataset = KeypointRCNNDataset(df_val)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = keypointrcnn_resnet50_fpn(num_keypoints=1, num_classes=len(class_names) + 1)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=1e-4)

# Training and validation loop
EPOCHS = 10
best_rmse = float("inf")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for images, targets, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Train"):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        train_loss += losses.item()

    # Validation
    model.eval()
    val_preds = {cls: [] for cls in class_names}
    val_trues = {cls: [] for cls in class_names}
    with torch.no_grad():
        for images, targets, classes in tqdm(val_loader, desc="Validation"):
            images = list(img.to(device) for img in images)
            outputs = model(images)

            for i, cls in enumerate(classes):
                true_xy = targets[i]['keypoints'][0, 0, :2].cpu().numpy()
                if outputs[i]['keypoints'].shape[0] == 0:
                    continue  # skip if no detection
                pred_xy = outputs[i]['keypoints'][0, 0, :2].cpu().numpy()
                val_trues[cls].append(true_xy)
                val_preds[cls].append(pred_xy)

    print(f"Epoch {epoch+1}: Train Loss = {train_loss / len(train_loader):.4f}")
    for cls in class_names:
        if val_preds[cls]:
            mse = mean_squared_error(np.array(val_trues[cls]), np.array(val_preds[cls]))
            rmse = np.sqrt(mse)
            print(f"  [{cls}] RMSE: {rmse:.4f}")
            if cls == 'pedestrian' and rmse < best_rmse:
                best_rmse = rmse
                torch.save(model.state_dict(), "best_keypointrcnn_model.pth")

print("✅ Training complete.")
