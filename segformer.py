import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score

# --- Config ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 12
threshold = 0.3  # adjust if needed

# Choose your model
model = SegFormer(in_channels=11, num_classes=1).to(device)
# model = DeepLabV3Plus(in_channels=11, num_classes=1).to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

print(f"🚀 Training on {device} with {len(train_dataset)} train and {len(val_dataset)} val samples.")

# --- Training Loop ---
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
            preds = (outputs > threshold).float()

            for i in range(images.size(0)):
                cls_idx = class_indices[i].item()
                cls_name = val_dataset.idx_to_class[cls_idx]
                val_metrics[cls_name]["y_true"] += masks[i].cpu().numpy().flatten().tolist()
                val_metrics[cls_name]["y_pred"] += preds[i].cpu().numpy().flatten().tolist()

    print(f"\n📉 Epoch {epoch+1}: Train Loss = {train_loss/len(train_loader):.4f}, Val Loss = {val_loss/len(val_loader):.4f}")
    print("📦 Seen classes in val:", sorted(val_metrics.keys()))
    for cls in sorted(val_metrics.keys()):
        values = val_metrics[cls]
        prec = precision_score(values["y_true"], values["y_pred"], zero_division=0)
        rec = recall_score(values["y_true"], values["y_pred"], zero_division=0)
        f1 = f1_score(values["y_true"], values["y_pred"], zero_division=0)
        print(f"📊 {cls}: Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}")

# Save model
torch.save(model.state_dict(), "segformer_trained.pth")
