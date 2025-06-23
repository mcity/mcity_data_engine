import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 1. Define the model architecture
class EnhancedEfficientNetKeypointModel(nn.Module):
    def __init__(self, num_classes=8, dropout_rate=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)

        # Modify first conv layer to accept 3 + num_classes channels
        original_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(3 + num_classes, 32, kernel_size=3, stride=2, padding=1, bias=False)
        with torch.no_grad():
            self.backbone.features[0][0].weight[:, :3] = original_conv.weight
            nn.init.normal_(self.backbone.features[0][0].weight[:, 3:], mean=0, std=0.01)

        # Modify classifier
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


# 2. Inference utility function
def predict_keypoint(model, image_path, object_class, class_to_idx, transform, device):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)  # [3, 260, 260]

    # Class one-hot encoding
    num_classes = len(class_to_idx)
    class_id = class_to_idx[object_class]
    class_onehot = F.one_hot(torch.tensor(class_id), num_classes=num_classes).float()
    onehot_expanded = class_onehot.view(-1, 1, 1).expand(-1, image_tensor.shape[1], image_tensor.shape[2])

    # Prepare input
    input_tensor = torch.cat([image_tensor, onehot_expanded], dim=0).unsqueeze(0).to(device)  # [1, 3+C, H, W]

    with torch.no_grad():
        output = model(input_tensor)
        predicted_xy = output.squeeze().cpu().numpy()

    return predicted_xy


# 3. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EnhancedEfficientNetKeypointModel(num_classes=8).to(device)
model.load_state_dict(torch.load("best2_enhanced_efficientnetb2_keypoint.pth", map_location=device))

# 4. Define classes and transform
class_names = ['bus', 'car', 'motorbike', 'pedestrian', 'pickup', 'trailer', 'truck', 'van']
class_to_idx = {cls: idx for idx, cls in enumerate(sorted(class_names))}

transform = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.ToTensor()
])

# 5. Predict
image_path = "/home/dataengine/Downloads/mcity_data_engine-1/resampled_balanced_split/val/images/pickup/pickup_29_0ce811.jpg"  # 🔁 Replace with your image path
object_class = "pickup"  # 🔁 Replace with the object class

predicted_point = predict_keypoint(
    model=model,
    image_path=image_path,
    object_class=object_class,
    class_to_idx=class_to_idx,
    transform=transform,
    device=device
)

print(f"Predicted keypoint (x, y): {predicted_point}")

import pandas as pd
import os
# 6. Visualize predicted and ground truth keypoints
df = pd.read_csv("resnet_keypoints.csv")



# Load ground truth (optional: match by image filename)
img = Image.open(image_path).resize((260, 260))
plt.imshow(img)

# Plot predicted keypoint (in red)
plt.scatter(predicted_point[0], predicted_point[1], c='orange', s=50, label='Predicted')

# Load ground truth (convert from normalized to pixel coordinates)
image_filename = os.path.basename(image_path)
gt_row = df[(df['image_path'].str.contains(image_filename)) & (df['class'] == object_class)]

if not gt_row.empty:
    gt_x = gt_row.iloc[0]['x'] * 260
    gt_y = gt_row.iloc[0]['y'] * 260
    plt.scatter(gt_x, gt_y, c='brown', s=50, label='Ground Truth')
else:
    print("Ground truth not found for this image.")

plt.title(f"Keypoint Prediction for '{object_class}'")
plt.legend()
plt.axis("off")
plt.show()
