import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# 1. Dataset class
class KeypointDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(df['class'].unique()))}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        class_id = self.class_to_idx[row['class']]
        class_onehot = torch.nn.functional.one_hot(torch.tensor(class_id), num_classes=len(self.class_to_idx)).float()
        input_tensor = torch.cat([image, class_onehot.view(-1, 1, 1).expand(-1, image.shape[1], image.shape[2])], dim=0)
        keypoint = torch.tensor([row['x'], row['y']], dtype=torch.float32)
        return input_tensor, keypoint, row['class']


# 2. Load CSV and preprocess
df = pd.read_csv("resnet_keypoints.csv")

# 3. Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 4. Split
df_train = df[df['split'] == 'train']
df_val = df[df['split'] == 'val']

train_dataset = KeypointDataset(df_train, transform=transform)
val_dataset = KeypointDataset(df_val, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# 5. Model
def get_model(num_classes=8):
    resnet = models.resnet18(pretrained=True)
    resnet.conv1 = nn.Conv2d(3 + num_classes, 64, kernel_size=7, stride=2, padding=3, bias=False)
    resnet.fc = nn.Linear(resnet.fc.in_features, 2)  # output x, y
    return resnet


# 6. Training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

best_val_loss = float('inf')

for epoch in range(20):
    model.train()
    train_loss = 0
    for inputs, keypoints, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/20 - Training"):
        inputs, keypoints = inputs.to(device), keypoints.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, keypoints)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)

    train_loss /= len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0
    val_results = {}
    with torch.no_grad():
        for inputs, keypoints, classes in tqdm(val_loader, desc="Validation"):
            inputs, keypoints = inputs.to(device), keypoints.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, keypoints)
            val_loss += loss.item() * inputs.size(0)

            # Per-class logging
            for c, l in zip(classes, (outputs - keypoints).pow(2).sum(dim=1).sqrt().cpu().numpy()):
                val_results.setdefault(c, []).append(l)

    val_loss /= len(val_loader.dataset)
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    # Log per-class RMSE
    for cls in sorted(val_results):
        rmse = np.mean(val_results[cls])
        print(f"  [{cls}] RMSE: {rmse:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_resnet_keypoint.pth")
        print("✅ Saved new best model.")
