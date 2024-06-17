# torchaudio                2.0.2+cu118              pypi_0    pypi
# torchsummary              1.5.1                    pypi_0    pypi
# torchvision               0.15.2+cu118
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
import torch.nn.init as init
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

# 로그를 저장할 logs 폴더 경로
logs_dir = '/home/ivpl-d29/myProject/Study_Model/ResNet/logs'

# SummaryWriter 생성
writer = SummaryWriter(logs_dir)

# GPU가 사용 가능한지 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 데이터 전처리 및 augmentation
data_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# 데이터셋 로드
data_dir = '/home/ivpl-d29/dataset/imagenet'
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=data_transforms)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=data_transforms)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=data_transforms)

# 데이터 로더
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, drop_last=True, pin_memory=True, prefetch_factor=2)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=4, drop_last=True, pin_memory=True, prefetch_factor=2)S
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=4, drop_last=True, pin_memory=True, prefetch_factor=2)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

        # He initialization 적용
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)  # 3x3 stride = 1
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        # He initialization 적용
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:     # Identitiy mapping의 크기를 맞춰 주기 위하여 1X1 conv layer 추가
           identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, ):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=True),    # 7X7 64, stride2
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),  # 3X3 max pool, stride 2
        )
        self.layer2 = self.make_layer(block, 64, layers[0], stride = 1)
        self.layer3 = self.make_layer(block, 128, layers[1], stride = 2)   # downsampling 수행 시 pooling사용 X, stride2인 conv filter 사용
        self.layer4 = self.make_layer(block, 256, layers[2], stride = 2)
        self.layer5 = self.make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # AdaptiveAvgPool2D에 1을 넣어 GlobalAveragePooling2D처럼 활용
        self.fc = nn.Linear(512 * 4, num_classes)  # ResNet34: expansion=1, ResNet50: expansion=4

    def make_layer(self, block, red_channels, blocks, stride=1, expansion=4):  # map size 맞추어 리스트 안 element 수에 맞게 layer 생성
        downsample = None
        if stride != 1 or self.in_channels != red_channels * expansion: # stride 2인 경우 downsampling
            downsample = nn.Sequential(     # identity mapping 크기 맞추기 위한 conv 1X1 layer
                nn.Conv2d(self.in_channels, red_channels * expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(red_channels * expansion),
            )
        layers = []
        layers.append(block(self.in_channels, red_channels, stride, downsample))
        self.in_channels = red_channels * expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, red_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)   # flatten
        x = self.fc(x)

        return x

def resnet34():
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    return model

def resnet50():
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    return model

# 모델 인스턴스 생성
model = resnet50()

# 모델을 GPU로 이동
model = model.to(device)


from torchsummary import summary
import matplotlib.pyplot as plt
# 모델 요약 정보 출력
summary(model, input_size=(3, 224, 224))
print(model)


# 손실 함수 및 optimizer 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True, min_lr=0.0001)

mean = torch.tensor([0.485, 0.456, 0.406], device=torch.device('cuda'))
std = torch.tensor([0.229, 0.224, 0.225], device=torch.device('cuda'))
# 훈련 함수
epochs = 28
def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=epochs, topk=(5,)):
    train_losses = []
    val_losses = []
    train_accuracies = []  # train 정확도를 기록할 리스트
    val_accuracies = []  # validation 정확도를 기록할 리스트
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        topk_correct = {k: 0 for k in topk}
        topk_total = {k: 0 for k in topk}

        with tqdm(train_loader, unit="batch", ncols=150) as tepoch:  # tqdm 사용
            tepoch.set_description(f"Epoch {epoch + 1}/{num_epochs}")
            for inputs, labels in tepoch:
                inputs, labels = inputs.to(device), labels.to(device)   # GPU로 이미지, 라벨 넘김

                # 입력 이미지 Normalize 수행
                inputs = transforms.Normalize(mean, std)(inputs)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # # Top-k 정확도 계산
                # _, predicted_topk = torch.topk(outputs, max(topk), dim=1)
                # for k in topk:
                #     topk_correct[k] += torch.sum(
                #         torch.tensor([l in p for l, p in zip(labels, predicted_topk[:, :k])])).item()
                #     topk_total[k] += labels.size(0)

                tepoch.set_postfix(loss=loss.item(), accuracy=1. * correct / total)
            train_loss = running_loss / len(train_loader.dataset)
            train_losses.append(train_loss)

            # 정확도 계산
            train_accuracy = correct / total
            # val accuracy, loss 계산
            val_accuracy, val_loss = evaluate_model(model, criterion, val_loader, device)
            # topk_accuracy = {k: topk_corr / topk_tot for k, (topk_corr, topk_tot) in zip(topk_correct.values(), topk_total.values())}
            # 리스트에 추가
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
            val_losses.append(val_loss)

            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            # for k, acc in topk_accuracy.items():
            #     writer.add_scalar(f'Top{k}/train', acc, epoch)
            # TensorBoard에 로그 기록
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Accuracy/train', train_accuracy, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/val', val_accuracy, epoch)

        # Learning rate 조정
        lr_scheduler.step(val_loss)

        # 에포크가 끝날 때마다 스케줄러를 업데이트 # StepLR 쓰는 경우
        #lr_scheduler.step()
        lr_scheduler.step(val_loss) # ReduceLROnPlateau를 쓰는 경우

        # 현재 학습률을 출력
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}: learning rate = {current_lr}")

    print('Training complete')
    return train_losses, val_losses, train_accuracies, val_accuracies

def evaluate_model(model, criterion, data_loader, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 입력 이미지를 Float32로 변환하기 전에 Normalize 수행
            inputs = transforms.Normalize(mean, std)(inputs)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # # Top-k 정확도 계산
            # _, predicted_topk = torch.topk(outputs, max(topk), dim=1)
            # for k in topk:
            #     topk_correct[k] += torch.sum(
            #         torch.tensor([l in p for l, p in zip(labels, predicted_topk[:, :k])])).item()
            #     topk_total[k] += labels.size(0)
    val_loss = val_loss / len(data_loader.dataset)
    val_accuracy = correct / total
    return val_accuracy, val_loss


# 모델 훈련
train_losses, val_losses, train_accuracies, val_accuracies = train_model(model, criterion, optimizer, lr_scheduler)

# 테스트
model.eval()
test_loss = 0.0
correct = 0
total = 0
with tqdm(total=len(test_loader), unit="batch", ncols=150, desc="Testing") as pbar:
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 입력 이미지를 Float32로 변환하기 전에 Normalize 수행
            inputs = transforms.Normalize(mean, std)(inputs)


            outputs = model(inputs.float())
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 진행 상황 갱신
            pbar.update(1)

test_loss = test_loss / len(test_loader.dataset)
test_accuracy = correct / total
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# 모델 저장
torch.save(model, os.path.join(logs_dir+'/model', 'model.pth'.format(epochs)))
# 가중치 저장
torch.save(model.state_dict(), os.path.join(logs_dir+'/model', 'model_weights.pth'.format(epochs)))
# 체크포인트 저장
checkpoint = {
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, os.path.join(logs_dir+'/model', 'checkpoint.pth'))

# 그래프 그리기
plt.figure(figsize=(10, 5))

# Loss 그래프
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy 그래프
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training accuracy')
plt.plot(val_accuracies, label='Validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
