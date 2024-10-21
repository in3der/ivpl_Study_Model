import torch
import torch.nn as nn
import os
import torchvision.transforms as transforms
import torch.optim as optim
from torchsummary import summary
import torchvision
from tqdm import tqdm
from torch.utils.data import DataLoader
from darknet import YOLOv1
from dataset import VOCDataset
from loss import YoloLoss
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from utils import (
    intersection_over_union,
    non_max_suppression,
    mean_average_precision,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    #load_checkpoint
)

print("실행")
seed = 123
torch.manual_seed(seed)

# 하이퍼파라미터
writer = SummaryWriter(log_dir="/home/ivpl-d29/myProject/Study_Model/YOLOv1/logs")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LOAD_MODEL = True
LOAD_MODEL_FILE = "/home/ivpl-d29/myProject/Study_Model/YOLOv1/logs/model/darknet_pretrain/checkpoint_epoch78_acc64.pth"
IMG_DIR = "/home/ivpl-d29/dataset/VOC/Aladdin/images"
LABEL_DIR = "/home/ivpl-d29/dataset/VOC/Aladdin/labels"
SAVE_MODEL_PATH = "/home/ivpl-d29/myProject/Study_Model/YOLOv1/logs/model/detect"
# 매 에폭마다 추적할 이미지 인덱스 설정
TRACK_IMAGE_IDX = 0  # 첫 번째 배치의 첫 번째 이미지

# from darknet_pretrain import YOLOv1 as YOLOv1_Pretrain
# from darknet import YOLOv1 as YOLOv1_New
#
# import torch.nn.parallel
# torch.serialization.add_safe_globals([torch.nn.DataParallel, set, YOLOv1])
# # 1. pretrained model 불러오기
# pretrained_model_path = '/home/ivpl-d29/myProject/Study_Model/YOLOv1/logs/model/darknet_pretrain/model_epoch78_acc64.pth'
# model_pretrained = YOLOv1_Pretrain()
#
# # DataParallel로 래핑
# model = torch.nn.DataParallel(model_pretrained, device_ids=[0]).to(DEVICE)
#
# # pretrained 모델의 state_dict 로드 (weights_only=True)
# state_dict = torch.load(pretrained_model_path)
#
# # DataParallel 모델에서 module 접근하여 state_dict 로드
# model.module.load_state_dict(state_dict)
#
# # 2. YOLOv1_New로 새로운 모델 초기화
# new_model = YOLOv1_New()
#
# # 3. 모델 하단 레이어 수정
# # 예를 들어, fc 레이어를 darknet.py와 동일하게 설정
# model.fc = nn.Sequential(
#     nn.Linear(1024, 7 * 7 * ((1 + 4) * 2 + 20)),
# )
#
# # 4. pretrained weights 복사
# # 필요한 부분만 복사
# with torch.no_grad():
#     model.conv.load_state_dict(model_pretrained.conv.state_dict())
#     # fc 레이어는 모델 구조가 다르므로 무시
#
# # 5. 학습을 위한 준비
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)



# 모델, 체크포인트 불러오기
model = YOLOv1().to(DEVICE)
summary(model, input_size=(3, 448, 448))
# model = torch.nn.DataParallel(model, device_ids=[0]).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")

    # 체크포인트에 'state_dict'가 없으면 다른 로드 방식을 시도
    if "state_dict" in checkpoint:
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(checkpoint["state_dict"])
            print("=> ckpt - state_dict loaded")
        else:
            model.load_state_dict(checkpoint["state_dict"])
            print("=> ckpt - state_dict loaded 2")
        optimizer.load_state_dict(checkpoint["optimizer"])

    else:
        # 전체 모델이 저장된 경우
        model = torch.load(LOAD_MODEL_FILE)
        print("=> ckpt - model loaded")

# if LOAD_MODEL:
#     load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)
#     print("체크포인트 모델 불러옴")

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img, bboxes

transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(), ])
print("data transform 완료")
train_dataset = VOCDataset("/home/ivpl-d29/dataset/VOC/Aladdin/train.csv", transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR)
val_dataset = VOCDataset("/home/ivpl-d29/dataset/VOC/Aladdin/test.csv", transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR)
test_dataset = VOCDataset("/home/ivpl-d29/dataset/VOC/Aladdin/test.csv", transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR)
train_loader = DataLoader(dataset=train_dataset, batch_size=8, num_workers=4, pin_memory=True, shuffle=True, drop_last=True)
val_loader = DataLoader(dataset=train_dataset, batch_size=8, num_workers=4, pin_memory=True, shuffle=False, drop_last=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=8, num_workers=4, pin_memory=True, shuffle=False, drop_last=True)
print("data 준비 완료")

# 학습률 0.001 [1 epoch] –>warmup으로 점차 증가-> 0.01 [75 epochs] –> 0.001 [30 epochs] –> 0.0001 [30 epochs]
EPOCHS = 135
#EPOCHS = 2
INITIAL_LR = 0.001  # 초기 학습률
WARMUP_EPOCHS = 75  # warm-up을 적용할 에폭 수
MAX_LR = 0.01  # warm-up 이후의 최대 학습률
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[75, 105], gamma=0.1)

def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        """Warm-up 동안 학습률을 점진적으로 증가"""
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

# Warm-up Scheduler 설정
warmup_scheduler = warmup_lr_scheduler(optimizer, WARMUP_EPOCHS, INITIAL_LR / MAX_LR)



loss_fn = YoloLoss()
def train_fn(train_loader, model, optimizer, loss_fn, scheduler, warmup_scheduler, epoch):
    loop = tqdm(train_loader, leave=True, desc=f"Training Epoch[{epoch}]")
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the progress bar
        loop.set_postfix(loss=loss.item())

    # Epoch에 맞춰 스케줄러 갱신 (학습률 조정)
    if epoch < WARMUP_EPOCHS:
        warmup_scheduler.step()
    else:
        scheduler.step()

    # Mean loss, lr 기록
    avg_loss = sum(mean_loss) / len(mean_loss)
    current_lr = optimizer.param_groups[0]["lr"]
    writer.add_scalar("Train/MeanLoss", avg_loss, epoch)
    writer.add_scalar("Train/Learning_Rate", current_lr, epoch)
    print(f"Mean loss was {sum(mean_loss) / len(mean_loss)}, Learning Rate was {current_lr}")
    return avg_loss  # 최종 평균 손실 반환


def validate_fn(val_loader, model, loss_fn, epoch):
    model.eval()
    loop = tqdm(val_loader, leave=True, desc="Validating")
    mean_loss, pred_boxes, target_boxes = [], [], []

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loop):
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = loss_fn(out, y)
            mean_loss.append(loss.item())

            # NMS로 예측 박스 얻기
            bboxes = cellboxes_to_boxes(out)
            pred_boxes += bboxes
            target_boxes += y

            # Update the progress bar
            loop.set_postfix(val_loss=loss.item())

    # Loss 계산
    avg_loss = sum(mean_loss) / len(mean_loss)
    writer.add_scalar("Validation/Loss", avg_loss, epoch)
    print(f"Epoch [{epoch}] - Val Loss: {avg_loss}\n")

    model.train()

def test_fn(test_loader, model, loss_fn):
    model.eval()
    loop = tqdm(test_loader, leave=True, desc="Testing")
    mean_loss = []

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loop):
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = loss_fn(out, y)
            mean_loss.append(loss.item())

            # Update the progress bar
            loop.set_postfix(test_loss=loss.item())

    avg_loss = sum(mean_loss) / len(mean_loss)

    # TensorBoard에 테스트 손실 기록
    writer.add_scalar("Test/Loss", avg_loss)

    print(f"Test Loss: {avg_loss}")


for epoch in range(EPOCHS):
    pred_boxes, target_boxes, num_pred_boxes, num_true_boxes = get_bboxes(train_loader, model, iou_threshold=0.4, threshold=0.2)
    print(f"예측 박스 개수: {num_pred_boxes}, 실제 박스 개수: {num_true_boxes}")
    mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.4, box_format="midpoint")
    print(f"Train mAP in {epoch}: {mean_avg_prec}")
    writer.add_scalar('Bounding Boxes/Predicted', num_pred_boxes, epoch)
    writer.add_scalar('Bounding Boxes/True', num_true_boxes, epoch)
    writer.add_scalar("mAP", mean_avg_prec, epoch)

    train_fn(train_loader, model, optimizer, loss_fn, scheduler, warmup_scheduler, epoch)
    validate_fn(val_loader, model, loss_fn, epoch)
    # 주기적으로 체크포인트 저장
    if epoch % 10 == 0 or epoch == EPOCHS - 1:
        # 모델을 pt와 pth 파일로 저장
        checkpoint = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(checkpoint, os.path.join(SAVE_MODEL_PATH, f"model_epoch{epoch}.pth"))  # 체크포인트 저장
        torch.save(model.state_dict(), os.path.join(SAVE_MODEL_PATH, f"model_epoch{epoch}.pt"))  # 가중치만 저장

# 최종 모델 저장
torch.save(model.state_dict(), os.path.join(SAVE_MODEL_PATH, "final_model.pt"))
torch.save({
    "epoch": EPOCHS,
    "state_dict": model.state_dict(),
    "optimizer": optimizer.state_dict()
}, os.path.join(SAVE_MODEL_PATH, "final_model.pth"))

# TensorBoard 닫기
writer.close()

# Test
test_fn(test_loader, model, loss_fn)

test_pred_boxes, test_target_boxes, test_num_pred_boxes, test_num_true_boxes = get_bboxes(test_loader, model, iou_threshold=0.4, threshold=0.2)
print(f"Test 예측 박스 개수: {test_num_pred_boxes}, Test 실제 박스 개수: {test_num_true_boxes}")
test_mean_avg_prec = mean_average_precision(test_pred_boxes, test_target_boxes, iou_threshold=0.4, box_format="midpoint")
print(f"Test mAP : {test_mean_avg_prec}")

