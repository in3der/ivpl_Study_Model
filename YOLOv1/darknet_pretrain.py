import os
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F
from torch.nn import AvgPool2d, MaxPool2d, ReLU
from torchsummary import summary
from tqdm import tqdm
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device : {device}')

# 시드 설정- 재현가능하도록
seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 로그를 저장할 logs 폴더 경로
logs_dir = '/home/ivpl-d29/myProject/Study_Model/YOLOv1/logs'

# SummaryWriter 생성
writer = SummaryWriter(logs_dir)

def make_conv(in_channels, out_channels, kernel_size, stride, padding, use_maxpool=False):
    layers = [
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                  padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.1, inplace=True)
    ]
    if use_maxpool:
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

    return nn.Sequential(*layers)


# 24 Conv layers + 2 FC layers
class YOLOv1(nn.Module):
    def __init__(self, in_channels=3, S=7, B=2, C=20, **kwargs):
        super(YOLOv1, self).__init__()
        self.conv = nn.Sequential(
            make_conv(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, use_maxpool=True),

            make_conv(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1, use_maxpool=True),

            make_conv(in_channels=192, out_channels=128, kernel_size=1, stride=1, padding=0),
            make_conv(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            make_conv(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
            make_conv(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, use_maxpool=True),

            make_conv(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
            make_conv(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            make_conv(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
            make_conv(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            make_conv(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
            make_conv(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            make_conv(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
            make_conv(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            make_conv(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0),
            make_conv(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, use_maxpool=True),

            make_conv(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),
            make_conv(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            make_conv(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),
            make_conv(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            # make_conv(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            # make_conv(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1),
            # make_conv(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            # make_conv(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
        )
        # self.fc1 = nn.Sequential(
        #     nn.Linear(S * S * 1024, 4096),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Dropout(p=0.5),
        # )
        # self.fc2 = nn.Sequential(
        #     nn.Linear(4096, S * S * ((1 + 4) * B + C)),
        # )
        self.init_weights()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Linear(1024, 1000),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p=0.5),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)
        return out

    def init_weights(self):
        # Initialize all Conv2D and Linear layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLOv1().to(device)
summary(model, input_size=(3, 224, 224))

model.to(device)


# --- 데이터셋 준비
data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
testdata_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# 데이터셋과 데이터로더 준비
def prepare_data(data_dir):
    data_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=data_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=data_transforms)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=data_transforms)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, drop_last=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, drop_last=True, pin_memory=True)

    return train_loader, val_loader, test_loader
# ---- 학습 준비 https://github.com/motokimura/yolo_v1_pytorch/blob/master/train_darknet.py#L197
epochs = 90

# 손실 함수 및 optimizer 설정
criterion = nn.CrossEntropyLoss()
from torch.optim.lr_scheduler import StepLR
optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4, momentum=0.9, nesterov=True)
lr_scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# 모델 훈련 함수
def train_model(model, criterion, optimizer, lr_scheduler, train_loader, val_loader, num_epochs=epochs, topk=5):
    best_accuracy = 0.0  # 최상의 정확도를 추적할 변수
    train_losses, val_losses, train_accuracies, val_accuracies, topk_accuracies, learning_rates = ([] for _ in range(6))

    for epoch in range(num_epochs):
        model.train()
        running_loss = correct = total = topk_correct = 0.0

        with tqdm(train_loader, unit="batch", ncols=100) as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{num_epochs}")
            for inputs, labels in tepoch:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Top-k 정확도 계산
                _, predicted_topk = outputs.topk(topk, dim=1, largest=True, sorted=True)
                topk_correct += sum(labels[i] in predicted_topk[i] for i in range(labels.size(0)))

                tepoch.set_postfix(loss=loss.item(), accuracy=correct / total)

            train_loss = running_loss / len(train_loader.dataset)
            train_losses.append(train_loss)
            train_accuracy = correct / total
            train_accuracies.append(train_accuracy)

            # Validation
            val_accuracy, val_loss, val_topk_accuracy = evaluate_model(model, criterion, val_loader, device, topk=topk)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            topk_accuracies.append(val_topk_accuracy)

            current_lr = optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)

            print(
                f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Top-{topk} Accuracy: {val_topk_accuracy:.4f}"
            )

            writer.add_scalar(f'Top{topk}/train', topk_correct / total, epoch)
            writer.add_scalar(f'Top{topk}/val', val_topk_accuracy, epoch)
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Accuracy/train', train_accuracy, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/val', val_accuracy, epoch)
            writer.add_scalar('Learning_Rate', current_lr, epoch)

            # 현재 epoch이 최고의 성능일 때만 저장
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                model_save_dir = os.path.join(logs_dir, 'model')
                torch.save(model, os.path.join(model_save_dir, f'model_epoch{epoch + 1}_acc{int(val_accuracy * 100):02d}.pth'))
                torch.save(model, os.path.join(model_save_dir, f'model_epoch{epoch + 1}_acc{int(val_accuracy * 100):02d}.pt'))
                torch.save(model.state_dict(), os.path.join(model_save_dir, f'model_weights_epoch{epoch + 1}_acc{int(val_accuracy * 100):02d}.pth'))
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                }
                torch.save(checkpoint, os.path.join(model_save_dir, f'checkpoint_epoch{epoch + 1}_acc{int(val_accuracy * 100):02d}.pth'))

        lr_scheduler.step()
        print(f"Epoch {epoch + 1}: learning rate = {current_lr}")

    # 마지막 에폭에서도 모델과 체크포인트 저장
    model_save_dir = os.path.join(logs_dir, 'model')
    torch.save(model, os.path.join(model_save_dir, f'model_final_epoch{num_epochs}.pth'))
    torch.save(model.state_dict(), os.path.join(model_save_dir, f'model_weights_final_epoch{num_epochs}.pth'))
    checkpoint = {
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': val_loss,
    }
    torch.save(checkpoint, os.path.join(model_save_dir, f'checkpoint_final_epoch{num_epochs}.pth'))

    print('Training complete')
    return train_losses, val_losses, train_accuracies, val_accuracies, topk_accuracies, learning_rates


# 모델 평가 함수
def evaluate_model(model, criterion, data_loader, device, topk=5):
    model.eval()
    val_loss = correct = total = topk_correct = 0.0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            _, predicted_topk = outputs.topk(topk, dim=1, largest=True, sorted=True)
            topk_correct += sum(labels[i] in predicted_topk[i] for i in range(labels.size(0)))

    val_loss = val_loss / len(data_loader.dataset)
    val_accuracy = correct / total
    topk_accuracy = topk_correct / total
    return val_accuracy, val_loss, topk_accuracy

# 테스트 함수
def test_model(model, criterion, test_loader, device, topk=5):
    model.eval()
    test_loss = correct = total = topk_correct = 0.0

    with tqdm(total=len(test_loader), unit="batch", ncols=100, desc="Testing") as pbar:
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                _, predicted_topk = outputs.topk(topk, dim=1, largest=True, sorted=True)
                topk_correct += sum(labels[i].item() in predicted_topk[i] for i in range(labels.size(0)))
                pbar.update(1)

    test_loss = test_loss / len(test_loader.dataset)
    test_accuracy = correct / total
    test_topk_accuracy = topk_correct / total

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Top-{topk} Test Accuracy: {test_topk_accuracy:.4f}")
    return test_loss, test_accuracy, test_topk_accuracy

# 모델 훈련 및 평가 함수
def main():
    model = YOLOv1().to(device)
    summary(model, input_size=(3, 224, 224))

    # 데이터 준비
    data_dir = '/home/ivpl-d29/dataset/imagenet'
    train_loader, val_loader, test_loader = prepare_data(data_dir)

    # 손실 함수 및 optimizer 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4, momentum=0.9, nesterov=True)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # 모델 훈련
    train_losses, val_losses, train_accuracies, val_accuracies, topk_accuracies, learning_rates = train_model(model, criterion, optimizer, lr_scheduler, train_loader, val_loader)

    # 모델 테스트
    test_loss, test_accuracy, test_topk_accuracy = test_model(model, criterion, test_loader, device)

    # 그래프 그리기
    plot_graphs(train_losses, val_losses, train_accuracies, val_accuracies, learning_rates, topk_accuracies)


def plot_graphs(train_losses, val_losses, train_accuracies, val_accuracies, learning_rates, topk_accuracies):
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(train_accuracies, label='Training accuracy')
    plt.plot(val_accuracies, label='Validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(learning_rates, label='Learning rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(topk_accuracies, label='Top-5 validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Top-5 Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('output_image.png')

if __name__ == "__main__":
    main()