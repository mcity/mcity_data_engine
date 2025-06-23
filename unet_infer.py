import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ---- U-Net definition (must match training architecture) ----
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        self.enc1 = CBR(11, 64)  # 3 RGB + 8 class one-hot channels
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

# ---- Inference function ----
def infer_contact_point(image_path, class_name, model_path="unet_heatmap_model3.pth"):
    class_to_idx = {
        "bus": 0, "car": 1, "motorbike": 2, "pedestrian": 3,
        "pickup": 4, "trailer": 5, "truck": 6, "van": 7
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load image
    original_img = Image.open(image_path).convert("RGB")
    original_size = original_img.size

    # Resize and transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img_tensor = transform(original_img)

    # Create one-hot class channels
    class_index = class_to_idx[class_name]
    one_hot = torch.zeros(8, 224, 224)
    one_hot[class_index, :, :] = 1.0

    # Combine input
    input_tensor = torch.cat([img_tensor, one_hot], dim=0).unsqueeze(0).to(device)

    # Load model
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        pred_mask = model(input_tensor)[0, 0].cpu().numpy()

    # Threshold and find bounding box
    mask = (pred_mask > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result_img = np.array(original_img)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        scale_x = original_size[0] / 224
        scale_y = original_size[1] / 224
        x1, y1 = int(x * scale_x), int(y * scale_y)
        x2, y2 = int((x + w) * scale_x), int((y + h) * scale_y)
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show result
    plt.imshow(result_img)
    plt.title(f"Predicted Contact ROI - Class: {class_name}")
    plt.axis("off")
    plt.show()

# Example usage
# infer_contact_point("cnn_heatmap_dataset/inputs/pedestrian_000123.jpg", "pedestrian")
