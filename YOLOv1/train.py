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

EPOCHS = 135
epoch = 135

# 모델, 체크포인트 불러오기
model = YOLOv1().to(DEVICE)
model = torch.nn.DataParallel(model, device_ids=[0]).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=2e-5, weight_decay=0)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> 체크포인트 조회")
    # state_dict 로드
    load_info = model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    # 반영된 레이어와 누락된 레이어 확인
    print("ckpt에 있어서 불러온 레이어:", set(load_info.missing_keys))
    print("ckpt에 있지만 정의된 모델에 없는 레이어:", set(load_info.unexpected_keys))
    print("=> 체크포인트 불러오기 성공")

if LOAD_MODEL:
    load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)
summary(model, input_size=(3, 448, 448))

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
train_loader = DataLoader(dataset=train_dataset, batch_size=16, num_workers=4, pin_memory=True, shuffle=True, drop_last=True)
val_loader = DataLoader(dataset=train_dataset, batch_size=16, num_workers=4, pin_memory=True, shuffle=False, drop_last=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=16, num_workers=4, pin_memory=True, shuffle=False, drop_last=True)
print("data 준비 완료")

loss_fn = YoloLoss()
def train_fn(train_loader, model, optimizer, loss_fn, epoch):
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

        current_lr = 2e-5
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

    # Mean loss, lr 기록
    avg_loss = sum(mean_loss) / len(mean_loss)
    current_lr = optimizer.param_groups[0]["lr"]
    format_lr = format(current_lr, '.5f')
    writer.add_scalar("Train/MeanLoss", avg_loss, epoch)
    writer.add_scalar("Train/Learning_Rate", current_lr, epoch)
    print(f"Mean loss was {sum(mean_loss) / len(mean_loss)}, Learning Rate was {format_lr}")
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
    pred_boxes, target_boxes, num_pred_boxes, num_true_boxes = get_bboxes(train_loader, model, iou_threshold=0.5, threshold=0.4, epoch=epoch)
    print(f"예측 박스 개수: {num_pred_boxes}, 실제 박스 개수: {num_true_boxes}")
    mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint")
    print(f"Train mAP in {epoch}: {mean_avg_prec}")
    writer.add_scalar('Bounding Boxes/Predicted', num_pred_boxes, epoch)
    writer.add_scalar('Bounding Boxes/True', num_true_boxes, epoch)
    writer.add_scalar("mAP", mean_avg_prec, epoch)

    train_fn(train_loader, model, optimizer, loss_fn, epoch)
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

test_pred_boxes, test_target_boxes, test_num_pred_boxes, test_num_true_boxes = get_bboxes(test_loader, model, iou_threshold=0.5, threshold=0.4, epoch=epoch)
print(f"Test 예측 박스 개수: {test_num_pred_boxes}, Test 실제 박스 개수: {test_num_true_boxes}")
test_mean_avg_prec = mean_average_precision(test_pred_boxes, test_target_boxes, iou_threshold=0.5, box_format="midpoint")
print(f"Test mAP : {test_mean_avg_prec}")

