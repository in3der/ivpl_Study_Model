import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from darknet import YOLOv1
from utils import cellboxes_to_boxes, non_max_suppression

# 모델 설정
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
model = YOLOv1().to(DEVICE)

# 체크포인트 파일 경로
checkpoint = torch.load("/home/ivpl-d29/myProject/Study_Model/YOLOv1/logs/model/detect/final_model.pt")
state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
model.load_state_dict(state_dict)
model.eval()

# 클래스 사전 설정
class_dict = {
    0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle',
    5: 'bus', 6: 'car', 7: 'cat', 8: 'chair', 9: 'cow',
    10: 'diningtable', 11: 'dog', 12: 'horse', 13: 'motorbike', 14: 'person',
    15: 'pottedplant', 16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tvmonitor',
}


# 이미지 로드 및 전처리
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0).to(DEVICE)


# 바운딩 박스 시각화 함수
def plot_image(image_path, boxes):
    image = np.array(Image.open(image_path).convert("RGB"))
    height, width, _ = image.shape
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for box in boxes:
        class_num = int(box[0])  # 클래스 번호
        class_name = class_dict.get(class_num, "Unknown")  # 클래스 이름
        confidence = box[1]
        box = box[2:]

        # 바운딩 박스 좌표 계산
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)

        # 클래스명과 confidence 표시
        annotation = f"{class_name}: {confidence:.2f}"
        ax.text(
            upper_left_x * width, upper_left_y * height,
            annotation,
            verticalalignment='top',
            color="white",
            fontsize=7,
            bbox=dict(facecolor="red", alpha=0.5, pad=0.5)
        )

    plt.show()


# 단일 이미지 예측 및 시각화
def predict_and_plot(image_path, model, iou_threshold=0.5, confidence_threshold=0.2):
    image_tensor = load_image(image_path)
    with torch.no_grad():
        predictions = model(image_tensor)
    bboxes = cellboxes_to_boxes(predictions)
    nms_boxes = non_max_suppression(
        bboxes[0], iou_threshold=iou_threshold, threshold=confidence_threshold
    )
    plot_image(image_path, nms_boxes)


# 이미지 파일 경로 설정
image_path = "/home/ivpl-d29/myProject/Study_Model/YOLOv1/newyork.jpg"
predict_and_plot(image_path, model)
image_path = "/home/ivpl-d29/myProject/Study_Model/YOLOv1/cowdog.jpg"
predict_and_plot(image_path, model)

