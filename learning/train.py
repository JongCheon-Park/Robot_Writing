# === Stroke Classification & Order Regression Training Code ===

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import json
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# === CONFIGURATION ===
IMAGE_SIZE = (550, 500)
MAX_STROKES = 15
BATCH_SIZE = 64
EPOCHS = 50000
ALPHA = 1.0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


from datetime import datetime

# === Unique run directory 생성 ===
def create_unique_output_dir(base='output/weights'):
    os.makedirs(base, exist_ok=True)
    existing = sorted([d for d in os.listdir(base) if d.isdigit()])
    next_id = f"{int(existing[-1])+1:04d}" if existing else "0000"
    path = os.path.join(base, next_id)
    os.makedirs(path)
    return path

RUN_DIR = create_unique_output_dir()



# === DATASET ===
class StrokeDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.root = os.path.dirname(json_path)
        self.samples = []

        for char, info in data["characters"].items():
            img_path = os.path.join(self.root, info["image_path"].replace("\\", "/"))
            if not (os.path.exists(img_path) and info.get("strokes") and len(info.get("points", [])) > 0):
                continue
            self.samples.append({
                "char": char,
                "image_path": img_path,
                "points": np.array(info["points"]),
                "stroke_labels": np.array(info["stroke_labels"]),
                "stroke_orders": self._get_orders(info["strokes"], len(info["points"]))
            })

        self.transform = T.Compose([
            T.Resize(IMAGE_SIZE),
            T.ToTensor()
        ])

    def _get_orders(self, strokes, num_points):
        orders = np.zeros(num_points, dtype=np.float32)
        for stroke in strokes:
            for i, idx in enumerate(stroke):
                if idx < num_points:
                    orders[idx] = i / max(1, len(stroke) - 1)
        return orders

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s["image_path"]).convert("RGB")
        img = self.transform(img)
        pts = torch.tensor(s["points"], dtype=torch.float32) / torch.tensor(IMAGE_SIZE[::-1])
        labels = torch.tensor(s["stroke_labels"], dtype=torch.long)
        orders = torch.tensor(s["stroke_orders"], dtype=torch.float32)
        return img, pts, labels, orders, s["char"], s["points"], s["image_path"]


# === MODEL ===
class StrokeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_enc = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(32, 128)
        )
        self.pt_enc = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.fusion = nn.Sequential(
            nn.Linear(128 + 64, 128), nn.ReLU()
        )
        self.stroke_head = nn.Linear(128, MAX_STROKES)
        self.order_head = nn.Linear(128, 1)

    def forward(self, img, points):
        B, N, _ = points.shape
        img_feat = self.img_enc(img).unsqueeze(1).expand(B, N, -1)
        pt_feat = self.pt_enc(points)
        x = torch.cat([pt_feat, img_feat], dim=-1)
        fused = self.fusion(x)
        stroke_logits = self.stroke_head(fused)
        order_preds = self.order_head(fused).squeeze(-1)
        return stroke_logits, order_preds


# === COLLATE ===
def collate_fn(batch):
    imgs, pts, lbls, orders, chars, raw_pts, img_paths = zip(*batch)
    max_len = max(len(p) for p in pts)
    B = len(imgs)
    padded_pts = torch.zeros(B, max_len, 2)
    padded_lbls = torch.full((B, max_len), -1)
    padded_orders = torch.zeros(B, max_len)
    mask = torch.zeros(B, max_len, dtype=torch.bool)

    for i in range(B):
        L = len(pts[i])
        padded_pts[i, :L] = pts[i]
        padded_lbls[i, :L] = lbls[i]
        padded_orders[i, :L] = orders[i]
        mask[i, :L] = 1

    imgs = torch.stack(imgs)
    return imgs, padded_pts, padded_lbls, padded_orders, mask, chars, raw_pts, img_paths


# === TRAIN ===
def train():
    dataset = StrokeDataset("../progress.json")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    model = StrokeNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
    mse_loss = nn.MSELoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        all_cls, all_order = [], []

        for img, pts, lbls, orders, mask, chars, raw_pts, img_paths in loader:
            img, pts, lbls, orders, mask = img.to(DEVICE), pts.to(DEVICE), lbls.to(DEVICE), orders.to(DEVICE), mask.to(DEVICE)
            logits, preds = model(img, pts)
            loss_cls = ce_loss(logits[mask], lbls[mask])
            loss_order = mse_loss(preds[mask], orders[mask])
            loss = loss_cls + ALPHA * loss_order

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            all_cls.append(loss_cls.item())
            all_order.append(loss_order.item())

        print(f"[Epoch {epoch}] Total Loss: {total_loss:.4f} | Cls: {np.mean(all_cls):.4f} | Order: {np.mean(all_order):.4f}")

        # 시각화 샘플 저장 (첫 배치 하나)
        if epoch % 1000 == 0:
            with torch.no_grad():
                # 저장 경로 변경
                ckpt_path = os.path.join(RUN_DIR, f"strokenet_epoch{epoch:03d}.pt")
                torch.save(model.state_dict(), ckpt_path)
                print(f"✅ Saved model checkpoint: {ckpt_path}")


                sample_img = img[0].cpu().permute(1, 2, 0).numpy()
                sample_pts = pts[0][mask[0]].cpu().numpy() * np.array(IMAGE_SIZE[::-1])
                sample_preds = preds[0][mask[0]].cpu().numpy()
                sample_labels = lbls[0][mask[0]].cpu().numpy()

                fig, ax = plt.subplots()
                ax.imshow(sample_img)
                ax.set_title(f"Epoch {epoch} predicted stroke order")

                for s in range(sample_labels.max() + 1):
                    indices = np.where(sample_labels == s)[0]
                    if len(indices) == 0:
                        continue
                    points = sample_pts[indices]
                    order_values = sample_preds[indices]

                    ax.plot(points[:, 0], points[:, 1], color='gray', linewidth=1, alpha=0.4)
                    ax.scatter(points[:, 0], points[:, 1], c=order_values, cmap='viridis', s=30, edgecolors='k')
                    ax.scatter(points[0, 0], points[0, 1], color='red', s=40, label='start' if s == 0 else None)
                    ax.scatter(points[-1, 0], points[-1, 1], color='blue', s=40, label='end' if s == 0 else None)
                    ax.text(points[0, 0] + 4, points[0, 1] - 6, str(s + 1), fontsize=11, color='black', weight='bold',
                            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

                fig.savefig(os.path.join(OUTPUT_DIR, f"epoch_{epoch}_sample.png"))
                plt.close()


if __name__ == "__main__":
    train()