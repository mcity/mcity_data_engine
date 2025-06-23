import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

# 1. Dataset class
class KeypointDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(df['class'].unique()))}
        self.num_classes = len(self.class_to_idx)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        class_id = self.class_to_idx[row['class']]
        class_onehot = torch.nn.functional.one_hot(torch.tensor(class_id), num_classes=self.num_classes).float()
        input_tensor = torch.cat([image, class_onehot.view(-1, 1, 1).expand(-1, image.shape[1], image.shape[2])], dim=0)
        keypoint = torch.tensor([row['x'], row['y']], dtype=torch.float32)* 260.0
        return input_tensor, keypoint, row['class']


# 2. Load CSV and preprocess
df = pd.read_csv("resnet_keypoints.csv")

# 3. Transformations (resize to 260x260 for EfficientNet-B2)
transform = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.ToTensor()
])

# 4. Data splits
df_train = df[df['split'] == 'train']
df_val = df[df['split'] == 'val']

train_dataset = KeypointDataset(df_train, transform=transform)
val_dataset = KeypointDataset(df_val, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

'''
# 5. EfficientNet-B2 model modification
def get_model(num_classes=8):
    base = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
    base.features[0][0] = nn.Conv2d(3 + num_classes, 32, kernel_size=3, stride=2, padding=1, bias=False)
    base.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(base.classifier[1].in_features, 2)  # Output (x, y)
    )
    return base
'''
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

class EnhancedEfficientNetKeypointModel(nn.Module):
    def __init__(self, num_classes=8, dropout_rate=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)

        # Adjust first convolution layer for extra channels
        original_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(3 + num_classes, 32, kernel_size=3, stride=2, padding=1, bias=False)

        with torch.no_grad():
            self.backbone.features[0][0].weight[:, :3] = original_conv.weight
            nn.init.normal_(self.backbone.features[0][0].weight[:, 3:], mean=0, std=0.01)

        # Enhanced classifier
        feature_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.BatchNorm1d(feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 2)
        )

        self.register_buffer("coord_bounds", torch.tensor([260.0, 260.0]))

    def forward(self, x):
        out = self.backbone(x)
        return torch.sigmoid(out) * self.coord_bounds

# 6. Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EnhancedEfficientNetKeypointModel(num_classes=8).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

best_val_loss = float('inf')

# 7. Training loop
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

            for c, l in zip(classes, (outputs - keypoints).pow(2).sum(dim=1).sqrt().cpu().numpy()):
                val_results.setdefault(c, []).append(l)

    val_loss /= len(val_loader.dataset)
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    for cls in sorted(val_results):
        rmse = np.mean(val_results[cls])
        print(f"  [{cls}] RMSE: {rmse:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best2_enhanced_efficientnetb2_keypoint.pth")
        print("✅ Saved new best model.")
