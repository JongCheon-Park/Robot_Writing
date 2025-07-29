import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
from PIL import Image
import os
from torchvision import transforms
import argparse
import json

# 한글 폰트 설정
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

IMAGE_SIZE = (550, 500)
MAX_STROKES = 15
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# 모델 정의
class StrokeNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.img_enc = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 5, stride=2), torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3, stride=2), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)), torch.nn.Flatten(),
            torch.nn.Linear(32, 128)
        )
        self.pt_enc = torch.nn.Sequential(
            torch.nn.Linear(2, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 64)
        )
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(128 + 64, 128), torch.nn.ReLU()
        )
        self.stroke_head = torch.nn.Linear(128, MAX_STROKES)
        self.order_head = torch.nn.Linear(128, 1)

    def forward(self, img, points):
        B, N, _ = points.shape
        img_feat = self.img_enc(img).unsqueeze(1).expand(B, N, -1)
        pt_feat = self.pt_enc(points)
        x = torch.cat([pt_feat, img_feat], dim=-1)
        fused = self.fusion(x)
        stroke_logits = self.stroke_head(fused)
        order_preds = self.order_head(fused).squeeze(-1)
        return stroke_logits, order_preds


# 예측 시각화
def visualize_prediction(image_path, raw_points, stroke_preds, order_preds):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    raw_points = np.array(raw_points)
    stroke_preds = np.array(stroke_preds)
    order_preds = np.array(order_preds)

    plt.figure(figsize=(6, 6.5))
    plt.imshow(img)
    plt.axis('off')

    for s in np.unique(stroke_preds):
        idxs = np.where(stroke_preds == s)[0]
        if len(idxs) == 0:
            continue
        pts = raw_points[idxs]
        orders = order_preds[idxs]

        plt.plot(pts[:, 0], pts[:, 1], color='gray', linewidth=1, alpha=0.4)
        plt.scatter(pts[:, 0], pts[:, 1], c=orders, cmap='viridis', s=30, edgecolors='k')
        plt.scatter(pts[0, 0], pts[0, 1], color='red', s=40)
        plt.scatter(pts[-1, 0], pts[-1, 1], color='blue', s=40)
        plt.text(pts[0, 0]+4, pts[0, 1]-6, str(s+1), fontsize=11,
                 color='black', weight='bold',
                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    plt.title("예측된 획 분할 및 순서")
    plt.show()


# 실행 함수
def run_inference(model_path, image_path, points):
    # 모델 준비
    model = StrokeNet().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # 이미지 및 포인트 전처리
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])
    pil_img = Image.open(image_path).convert("RGB")
    img_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

    norm_pts = torch.tensor(points, dtype=torch.float32) / torch.tensor(IMAGE_SIZE[::-1])
    norm_pts = norm_pts.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        stroke_logits, order_preds = model(img_tensor, norm_pts)
        stroke_preds = stroke_logits.argmax(dim=-1).squeeze().cpu().numpy()
        order_preds = order_preds.squeeze().cpu().numpy()

    visualize_prediction(image_path, points, stroke_preds, order_preds)


# 예시 실행
if __name__ == "__main__":
    # 예시: 테스트 샘플 지정
    model_file = "output/weights/0000/strokenet_epoch1000.pt"
    image_file = "../img/U+AC07.png"
    with open("../progress.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        pts = raw_data["characters"]["갇"]["points"]

    run_inference(model_file, image_file, pts)
