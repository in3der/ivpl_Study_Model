import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn import AvgPool2d, MaxPool2d, ReLU

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device : {device}')
from torchsummary import summary
from tqdm import tqdm
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os
from torchvision.ops import StochasticDepth

# 로그를 저장할 logs 폴더 경로
logs_dir = '/home/ivpl-d29/myProject/Study_Model/EfficientNet/logs'

# SummaryWriter 생성
writer = SummaryWriter(logs_dir)


class SiLU(nn.Module):  # ReLU 대신 사용하는 activation 함수
    def __init__(self):
        super(SiLU, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


class SEBlock(nn.Module):
    def __init__(self, channel, ratio):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.SiLU(),  # ReLU 대신 SiLU 사용
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        batch, channel, _, _ = input.size()
        x = self.squeeze(input).view(batch, channel)
        x = self.excitation(x).view(batch, channel, 1, 1)
        x = input * x.expand_as(input)  # scaling
        return x


class MBConv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding, ratio=4):
        super(MBConv1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel, stride=stride, padding=padding,
                                   bias=False, groups=in_channels)
        self.bn = nn.BatchNorm2d(in_channels, momentum=0.99)
        self.SiLU = SiLU()
        self.SEBlock = SEBlock(in_channels, ratio=4)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.99)
        self.stochastic_depth = StochasticDepth(p=0.2, mode='batch')

    def forward(self, x):
        identity = x
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.SiLU(x)
        x = self.SEBlock(x)
        x = self.conv(x)
        x = self.bn2(x)
        if self.in_channels == self.out_channels and self.stride == 1:
            x = self.stochastic_depth(x)
            x += identity
        return x


class MBConv6(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding, ratio=6):
        super(MBConv6, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.conv1 = nn.Conv2d(in_channels, in_channels * ratio, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(in_channels * ratio, momentum=0.99)
        self.SiLU = SiLU()
        self.depthwise = nn.Conv2d(in_channels * ratio, in_channels * ratio, kernel_size=kernel, stride=stride,
                                   padding=padding, groups=in_channels)
        self.SEBlock = SEBlock(in_channels * ratio, ratio=4)
        self.conv2 = nn.Conv2d(in_channels * ratio, out_channels, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.99)
        self.stochastic_depth = StochasticDepth(p=0.2, mode='batch')

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn(x)
        x = self.SiLU(x)
        x = self.depthwise(x)
        x = self.SiLU(x)
        x = self.SEBlock(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.in_channels == self.out_channels and self.stride == 1:
            x = self.stochastic_depth(x)
            x += identity
        return x


class EfficientNetB0(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(EfficientNetB0, self).__init__()
        # Conv3x3, 224x224, 32, 1
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, momentum=0.99),
            SiLU()
        )
        # MBConv1, k3x3, 112x112, 16, 1
        self.stage2 = MBConv1(in_channels=32, out_channels=16, kernel=3, stride=2, padding=1)
        # MBConv6, k3x3, 112x112, 24, 1
        self.stage3 = MBConv6(in_channels=16, out_channels=24, kernel=3, stride=1, padding=1)
        # MBConv6, k5x5, 56x56, 40, 2
        self.stage4 = nn.Sequential(
            MBConv6(in_channels=24, out_channels=40, kernel=5, stride=2, padding=2),
            MBConv6(in_channels=40, out_channels=40, kernel=5, stride=1, padding=2)
        )
        # MBConv6, k3x3, 28x28, 80, 3
        self.stage5 = nn.Sequential(
            MBConv6(in_channels=40, out_channels=80, kernel=3, stride=2, padding=1),
            MBConv6(in_channels=80, out_channels=80, kernel=3, stride=1, padding=1),
            MBConv6(in_channels=80, out_channels=80, kernel=3, stride=1, padding=1)
        )
        # MBConv6, k5x5, 14x14, 112, 3
        self.stage6 = nn.Sequential(
            MBConv6(in_channels=80, out_channels=112, kernel=5, stride=2, padding=2),
            MBConv6(in_channels=112, out_channels=112, kernel=5, stride=1, padding=2),
            MBConv6(in_channels=112, out_channels=112, kernel=5, stride=1, padding=2)
        )
        # MBConv6, k5x5, 14x14, 192, 4
        self.stage7 = nn.Sequential(
            MBConv6(in_channels=112, out_channels=192, kernel=5, stride=1, padding=2),
            MBConv6(in_channels=192, out_channels=192, kernel=5, stride=1, padding=2),
            MBConv6(in_channels=192, out_channels=192, kernel=5, stride=1, padding=2),
            MBConv6(in_channels=192, out_channels=192, kernel=5, stride=1, padding=2),
        )
        # MBConv6, k3x3, 7x7, 320, 1
        self.stage8 = MBConv6(in_channels=192, out_channels=320, kernel=3, stride=2, padding=1)
        # Conv1x1 & Pooling & FC, 7x7, 1280, 1
        self.stage9 = nn.Sequential(
            # nn.Conv2d(in_channels=320, out_channels=1280, kernel_size=7, stride=1),
            nn.Conv2d(in_channels=320, out_channels=1280, kernel_size=1, stride=1), # kernel size를 7->1로 수정
            nn.BatchNorm2d(1280, momentum=0.99),
            SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.stage8(x)
        x = self.stage9(x)
        return x


model = EfficientNetB0().to(device)
model = nn.DataParallel(model)
model.to(device)
summary(model, (3, 224, 224))

# --- 데이터셋 준비, 학습 실행
transform = transforms.Compose([
    transforms.Resize((331, 331)),  # 임의로 resize 331 지정
    transforms.RandomCrop(224),
    transforms.AutoAugment(policy=transforms.autoaugment.AutoAugmentPolicy.IMAGENET,
                           interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

epochs = 100

# 손실 함수 및 optimizer 설정
criterion = nn.CrossEntropyLoss()
from torch.optim.lr_scheduler import ExponentialLR
optimizer = optim.RMSprop(model.parameters(), lr=0.256, weight_decay=1e-5, eps=1.0, momentum=0.9, alpha=0.9)
step_size = 2.4
decay_rate = 0.97
gamma = decay_rate ** (1 / step_size)
lr_scheduler = ExponentialLR(optimizer, gamma=gamma)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib.pyplot as plt
from tqdm import tqdm

# 모델 훈련 함수
def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=epochs, topk=(5,)):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    topk_accuracies = {k: [] for k in topk}
    learning_rates = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        topk_correct = {k: 0 for k in topk}

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
                for k in topk:
                    _, predicted_topk = outputs.topk(k, dim=1, largest=True, sorted=True)
                    topk_correct[k] += sum(labels[i] in predicted_topk[i] for i in range(labels.size(0)))

                tepoch.set_postfix(loss=loss.item(), accuracy=correct / total)

            train_loss = running_loss / len(train_loader.dataset)
            train_losses.append(train_loss)
            train_accuracy = correct / total
            train_accuracies.append(train_accuracy)

            # Validation
            val_accuracy, val_loss, val_topk_accuracies = evaluate_model(model, criterion, test_loader, device, topk=topk)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            for k in topk:
                topk_accuracies[k].append(val_topk_accuracies[k])

            current_lr = optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)

            print(
                f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, "
                f"Top-k Accuracy: {val_topk_accuracies} ")

            for k in topk:
                writer.add_scalar(f'Top{k}/train', topk_correct[k] / total, epoch)
                writer.add_scalar(f'Top{k}/val', val_topk_accuracies[k], epoch)

            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Accuracy/train', train_accuracy, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/val', val_accuracy, epoch)
            writer.add_scalar('Learning_Rate', current_lr, epoch)

        lr_scheduler.step()
        print(f"Epoch {epoch + 1}: learning rate = {current_lr}")

    print('Training complete')
    return train_losses, val_losses, train_accuracies, val_accuracies, topk_accuracies, learning_rates

# 모델 평가 함수
def evaluate_model(model, criterion, data_loader, device, topk=(5,)):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    topk_correct = {k: 0 for k in topk}

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            for k in topk:
                _, predicted_topk = outputs.topk(k, dim=1, largest=True, sorted=True)
                topk_correct[k] += sum(labels[i] in predicted_topk[i] for i in range(labels.size(0)))

    val_loss = val_loss / len(data_loader.dataset)
    val_accuracy = correct / total

    topk_accuracies = {k: topk_correct[k] / total for k in topk}
    return val_accuracy, val_loss, topk_accuracies

# 모델 훈련
train_losses, val_losses, train_accuracies, val_accuracies, topk_accuracies, learning_rates = train_model(
    model, criterion, optimizer, lr_scheduler)

# 테스트
model.eval()
test_loss = 0.0
correct = 0
total = 0
topk_correct = {k: 0 for k in (5,)}

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
            for k in topk_correct.keys():
                _, predicted_topk = outputs.topk(k, dim=1, largest=True, sorted=True)
                topk_correct[k] += sum(labels[i] in predicted_topk[i] for i in range(labels.size(0)))

            pbar.update(1)

test_loss = test_loss / len(test_loader.dataset)
test_accuracy = correct / total
topk_accuracies = {k: topk_correct[k] / total for k in topk_correct}

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
for k, acc in topk_accuracies.items():
    print(f"Top-{k} Test Accuracy: {acc:.4f}")

# 그래프 그리기
plt.figure(figsize=(15, 10))

# Loss 그래프
plt.subplot(2, 2, 1)
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy 그래프
plt.subplot(2, 2, 2)
plt.plot(train_accuracies, label='Training accuracy')
plt.plot(val_accuracies, label='Validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Learning Rate 그래프
plt.subplot(2, 2, 3)
plt.plot(learning_rates, label='Learning rate')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.legend()

# Top-k Accuracy 그래프
plt.subplot(2, 2, 4)
for k in topk_accuracies:
    plt.plot(topk_accuracies[k], label=f'Top-{k} validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Top-k Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


# 모델 저장
torch.save(model, os.path.join(logs_dir + '/model', 'model.pth'.format(epochs)))
# 가중치 저장
torch.save(model.state_dict(), os.path.join(logs_dir + '/model', 'model_weights.pth'.format(epochs)))
# 체크포인트 저장
checkpoint = {
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, os.path.join(logs_dir + '/model', 'checkpoint.pth'))

