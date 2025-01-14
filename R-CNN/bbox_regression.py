import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import AlexNet
import matplotlib.pyplot as plt
import numpy as np
# custom_bbox_regression_dataset.py에서 바운딩 박스 회귀 데이터셋 클래스 임포트
from utils.data.custom_bbox_regression_dataset import BBoxRegressionDataset
import sys
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

sys.path.append("/home/ivpl-d29/myProject/Study_Model/R-CNN")
import utils as util
from utils.util import save_model, plot_loss


def load_data(data_root_dir):
    """
    데이터 로딩 및 전처리를 수행하는 함수
    Args:
        data_root_dir: 데이터셋의 루트 디렉토리 경로
    Returns:
        DataLoader 객체
    """
    # 이미지 전처리 파이프라인 정의
    transform = transforms.Compose([
        transforms.ToPILImage(),  # 텐서를 PIL 이미지로 변환
        transforms.Resize((227, 227)),  # AlexNet 입력 크기에 맞게 리사이즈
        transforms.RandomHorizontalFlip(),  # 데이터 증강: 좌우 반전
        transforms.ToTensor(),  # PIL 이미지를 텐서로 변환
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 정규화
    ])

    # 커스텀 데이터셋 객체 생성
    data_set = BBoxRegressionDataset(data_root_dir, transform=transform)
    # DataLoader 설정 (배치 사이즈=128, 셔플=True, 멀티프로세싱 워커=8)
    data_loader = DataLoader(data_set, batch_size=32, shuffle=True, num_workers=8)

    return data_loader


def calculate_iou(pred_bbox, target_bbox):
    """
    예측된 바운딩 박스와 실제 바운딩 박스 간의 IoU(Intersection over Union) 계산
    """
    # pred_bbox와 target_bbox는 [x, y, w, h] 형식
    pred_x1 = pred_bbox[:, 0] - pred_bbox[:, 2] / 2
    pred_y1 = pred_bbox[:, 1] - pred_bbox[:, 3] / 2
    pred_x2 = pred_bbox[:, 0] + pred_bbox[:, 2] / 2
    pred_y2 = pred_bbox[:, 1] + pred_bbox[:, 3] / 2

    target_x1 = target_bbox[:, 0] - target_bbox[:, 2] / 2
    target_y1 = target_bbox[:, 1] - target_bbox[:, 3] / 2
    target_x2 = target_bbox[:, 0] + target_bbox[:, 2] / 2
    target_y2 = target_bbox[:, 1] + target_bbox[:, 3] / 2

    # 교집합 영역 계산
    x1 = torch.max(pred_x1, target_x1)
    y1 = torch.max(pred_y1, target_y1)
    x2 = torch.min(pred_x2, target_x2)
    y2 = torch.min(pred_y2, target_y2)

    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    # 합집합 영역 계산
    pred_area = pred_bbox[:, 2] * pred_bbox[:, 3]
    target_area = target_bbox[:, 2] * target_bbox[:, 3]
    union = pred_area + target_area - intersection

    return (intersection / (union + 1e-6)).mean()   # union이 0이 되는 경우를 방지하기 위해 1e-6을 추가로 더함

def plot_training_progress(losses, ious):
    plt.figure(figsize=(12, 5))

    # Loss 플롯
    plt.subplot(1, 2, 1)
    plt.plot(range(len(losses)), losses, 'b-', label='Training Loss')
    plt.title('Training Loss Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # x축 범위 명시적 설정
    plt.xlim(0, len(losses))

    # IoU 플롯
    plt.subplot(1, 2, 2)
    plt.plot(range(len(ious)), ious, 'r-', label='IoU Score')
    plt.title('IoU Score Progress')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.grid(True)
    plt.legend()

    # x축 범위 명시적 설정
    plt.xlim(0, len(ious))

    plt.tight_layout()
    plt.savefig('./logs/BBox_training_progress.png')
    plt.close()


def get_model(device=None):
    """
    사전 학습된 AlexNet 모델을 로드하는 함수
    """
    # AlexNet 모델 생성 (2개 클래스: 배경, 자동차)
    model = AlexNet(num_classes=2)
    # 사전 학습된 가중치 로드
    model.load_state_dict(
        torch.load('/home/ivpl-d29/myProject/Study_Model/R-CNN/logs/model/linear_svm_alexnet_car_best.pth'))
    model.eval()  # 평가 모드로 설정

    # 특징 추출기로만 사용할 것이므로 그래디언트 계산 비활성화
    for param in model.parameters():
        param.requires_grad = False

    if device:
        model = model.to(device)

    return model


def load_specific_image(image_path, transform):
    """Load and transform a specific image"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]  # (height, width)

    # Transform for model input
    if transform:
        image_tensor = transform(image)

    return image, image_tensor, original_size


def load_ground_truth(bbox_path):
    """Load ground truth bounding boxes"""
    bbox_data = pd.read_csv(bbox_path, header=None, sep=' ')
    # Assuming the format is [x_min, y_min, x_max, y_max]
    return bbox_data.values[0, :4]  # Taking first bounding box for simplicity


def plot_boxes(image, gt_box, pred_box, epoch, save_path):
    """Plot original image with ground truth and predicted boxes"""
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)

    # Ground truth box in green
    rect_gt = patches.Rectangle(
        (gt_box[0], gt_box[1]),
        gt_box[2] - gt_box[0],
        gt_box[3] - gt_box[1],
        linewidth=2,
        edgecolor='g',
        facecolor='none',
        label='Ground Truth'
    )
    ax.add_patch(rect_gt)

    # Predicted box in red
    rect_pred = patches.Rectangle(
        (pred_box[0], pred_box[1]),
        pred_box[2] - pred_box[0],
        pred_box[3] - pred_box[1],
        linewidth=2,
        edgecolor='r',
        facecolor='none',
        label='Prediction'
    )
    ax.add_patch(rect_pred)

    plt.title(f'Epoch {epoch}')
    plt.legend()
    plt.savefig(os.path.join(save_path, f'bbox_prediction_epoch_{epoch}.png'))
    plt.close()


def visualize_prediction(model, feature_model, image_tensor, original_size, device):
    """Get model prediction and convert to original image coordinates"""
    model.eval()
    feature_model.eval()

    with torch.no_grad():
        # Get features and prediction
        image_tensor = image_tensor.unsqueeze(0).to(device)
        features = feature_model.features(image_tensor)
        features = torch.flatten(features, 1)
        pred_bbox = model(features).cpu().numpy()[0]

        # Convert normalized coordinates back to image coordinates
        height, width = original_size
        x_center = pred_bbox[0] * width
        y_center = pred_bbox[1] * height
        w = pred_bbox[2] * width
        h = pred_bbox[3] * height

        # Convert to [x_min, y_min, x_max, y_max] format
        x_min = x_center - w / 2
        y_min = y_center - h / 2
        x_max = x_center + w / 2
        y_max = y_center + h / 2

        return [x_min, y_min, x_max, y_max]


# Modify the train_model function to include visualization
def train_model_with_viz(data_loader, feature_model, model, criterion, optimizer,
                         lr_scheduler, viz_image_path, viz_bbox_path, num_epochs=25, device=None):
    """Modified training function with visualization for a specific image"""
    # Create directory for saving visualizations
    os.makedirs('./logs/bbox_viz', exist_ok=True)

    since = time.time()
    best_loss = float('inf')

    # 손실값과 IoU 기록을 위한 리스트
    loss_list = []
    iou_list = []

    # Load and prepare visualization image
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    viz_image, viz_image_tensor, original_size = load_specific_image(viz_image_path, transform)
    gt_bbox = load_ground_truth(viz_bbox_path)

    model.train()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 30)

        running_loss = 0.0
        running_iou = 0.0
        batch_count = 0

        # Regular training code...
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.float().to(device)
            batch_count += 1

            features = feature_model.features(inputs)
            features = torch.flatten(features, 1)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # IoU 계산
            with torch.no_grad():
                batch_iou = calculate_iou(outputs, targets)
                running_iou += batch_iou.item()

            # 통계 기록
            running_loss += loss.item() * inputs.size(0)

            # 배치마다 중간 결과 출력
            if batch_count % 100 == 0:
                print(f'Batch {batch_count}: Loss: {loss.item():.4f}, IoU: {batch_iou.item():.4f}')

        # Visualize predictions for the specific image
        pred_bbox = visualize_prediction(model, feature_model, viz_image_tensor, original_size, device)
        plot_boxes(viz_image, gt_bbox, pred_bbox, epoch, './logs/bbox_viz')

        lr_scheduler.step()

        # 에폭당 평균 손실 및 IoU 계산
        epoch_loss = running_loss / len(data_loader.dataset)
        epoch_iou = running_iou / len(data_loader)

        loss_list.append(epoch_loss)
        iou_list.append(epoch_iou)

        print(f'Epoch {epoch} Summary:')
        print(f'Loss: {epoch_loss:.4f}')
        print(f'IoU: {epoch_iou:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.8f}')

        # 최고 성능 모델 저장
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_model(model, './logs/model/bbox_regression_best.pth')

        # 현재 에폭 모델 저장
        # save_model(model, f'./logs/model/bbox_regression_{epoch}.pth')

        # 학습 곡선 실시간 플로팅
        plot_training_progress(loss_list, iou_list)

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Loss: {best_loss:.4f}')
    plot_training_progress(loss_list, iou_list)

    return model



if __name__ == '__main__':
    # 데이터 로더 생성
    data_loader = load_data('/home/ivpl-d29/dataset/VOC/voc_car/bbox_regression')

    # GPU 사용 가능 시 GPU 설정
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 특징 추출을 위한 AlexNet 모델 로드
    feature_model = get_model(device)

    # 바운딩 박스 회귀를 위한 선형 레이어 생성
    # AlexNet의 feature map 크기: 256 채널 * 6 * 6
    in_features = 256 * 6 * 6
    out_features = 4  # 바운딩 박스 좌표 (x, y, w, h)
    model = nn.Linear(in_features, out_features)
    model.to(device)

    # 손실 함수, 옵티마이저, 학습률 스케줄러 설정
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # # 모델 학습 실행
    # loss_list, iou_list = train_model(data_loader, feature_model, model, criterion, optimizer, lr_scheduler, device=device, num_epochs=200)

    # Update the training call with visualization parameters
    train_model_with_viz(
        data_loader,
        feature_model,
        model,
        criterion,
        optimizer,
        lr_scheduler,
        viz_image_path='/home/ivpl-d29/dataset/VOC/voc_car/bbox_regression/JPEGImages/000012.jpg',
        viz_bbox_path='/home/ivpl-d29/dataset/VOC/voc_car/bbox_regression/bndboxs/000012.csv',
        device=device,
        num_epochs=200
    )

