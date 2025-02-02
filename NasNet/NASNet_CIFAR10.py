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

# 로그를 저장할 logs 폴더 경로
logs_dir = '/home/ivpl-d29/myProject/Study_Model/NasNet/logs'

# SummaryWriter 생성
writer = SummaryWriter(logs_dir)


class SepConv2d(nn.Module):  # Separable Convolution 2D, 논문 Appendix.A.4. 참고
    def __init__(self, in_channels, out_channels, kernel, stride=1, padding=1, bias=False):
        super(SepConv2d, self).__init__()
        # depthwise convolution : 각 입력 채널에 대해 독립적으로 필터링 수행 (입력 채널 수 = 출력 채널 수)
        # groups=in_channels 설정하여 입력 채널 수로 그룹을 나누어 각 그룹에 대해 독립적 필터링 수행
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel, stride=stride, padding=padding, bias=bias,
                                   groups=in_channels)
        self.depthwise_s1 = nn.Conv2d(in_channels, in_channels, kernel, stride=1, padding=padding, bias=bias,
                                      groups=in_channels)
        # pointwise convolution : 1x1 convolution 수행하여 채널 수를 변환.
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=bias)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1)

    def forward(self, x):
        # ver.1 only conv
        # x = self.depthwise(x)
        # x = self.pointwise(x)

        # ver.2 bn-conv-relu    # acc 가장 높음
        x = self.bn(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.relu(x)

        # ver.3 sep 2번      # 학습 불안정
        # x = self.relu(x)
        # x = self.depthwise(x)
        # x = self.pointwise(x)
        # x = self.bn(x)
        # x = self.relu(x)
        # x = self.depthwise_s1(x)
        # x = self.pointwise(x)
        # x = self.bn(x)
        return x


class NormalCell(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NormalCell, self).__init__()
        self.prev1X1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.SepConv3x3 = SepConv2d(out_channels, out_channels, kernel=3, stride=1, padding=1)
        self.SepConv5x5 = SepConv2d(out_channels, out_channels, kernel=5, stride=1, padding=2)
        self.Avg3x3 = AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.bn_in = nn.BatchNorm2d(in_channels, eps=0.001, momentum=0.1)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1)
        self.relu = nn.ReLU()
    def forward(self, cur, prev):
        # h_j = cur(현재 hidden state)
        # h_{j-1} = prev(revious)(이전 hidden state)
        cur = self.relu(self.prev1X1(self.bn_in(cur)))
        prev = self.relu(self.prev1X1(self.bn_in(prev)))
        b0 = self.SepConv3x3(cur) + cur
        b1 = self.SepConv3x3(prev) + self.SepConv5x5(prev)
        b2 = self.Avg3x3(cur) + prev
        b3 = self.Avg3x3(prev) + self.Avg3x3(prev)
        b4 = self.SepConv5x5(prev) + self.SepConv3x3(prev)
        out = torch.cat([prev, b0, b1, b2, b3, b4], dim=1)
        return out


class ReductionCell(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ReductionCell, self).__init__()
        self.prev1X1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.SepConv3x3 = SepConv2d(in_channels // 3, out_channels, kernel=3, stride=1, padding=1)
        self.SepConv5x5 = SepConv2d(in_channels // 3, out_channels, kernel=5, stride=2, padding=2)
        self.SepConv7x7 = SepConv2d(in_channels // 3, out_channels, kernel=7, stride=2, padding=3)
        self.Avg3x3 = AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.Avg3x3_1 = AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.Max3x3 = MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.Conv1x1 = nn.Conv2d(in_channels // 3, out_channels, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(in_channels // 3, eps=0.001, momentum=0.1)
        self.bn_in = nn.BatchNorm2d(in_channels, eps=0.001, momentum=0.1)
        self.relu = nn.ReLU()
    def forward(self, cur, prev):
        cur = self.relu(self.prev1X1(self.bn_in(cur)))
        prev = self.relu(self.prev1X1(self.bn_in(prev)))
        b0 = self.SepConv7x7(prev) + self.SepConv5x5(cur)
        b1 = self.Max3x3(cur) + self.SepConv7x7(prev)
        b2 = self.Avg3x3(cur) + self.SepConv5x5(prev)
        b3 = self.Max3x3(cur) + self.SepConv3x3(b0)
        b4 = self.Avg3x3_1(b0) + b1
        out = torch.cat([b1, b2, b3, b4], dim=1)
        return out


class NASNet(nn.Module):  # NASNet-A 6@768 for CIFAR-10 (3->32(192)->64(384)->128(768))
    def __init__(self, in_channels=3, num_classes=10, N=6, afterStem=96):
        super(NASNet, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, afterStem, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(afterStem),
            nn.ReLU()
        )
        self.cells = nn.ModuleList()
        self.cells.append(NormalCell(96, 32))
        for _ in range(1, N - 1):
            self.cells.append(NormalCell(192, 32))

        self.cells.append(ReductionCell(192, 64))

        self.cells.append(NormalCell(256, 64))
        for _ in range(1, N - 1):
            self.cells.append(NormalCell(384, 64))

        self.cells.append(ReductionCell(384, 128))

        self.cells.append(NormalCell(512, 128))
        for _ in range(1, N - 1):
            self.cells.append(NormalCell(768, 128))

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(768, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.bn = nn.BatchNorm2d(in_channels, eps=0.001, momentum=0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.stem(x)
        prev_x = x
        for cell in self.cells:
            x = cell(x, prev_x)
            prev_x = x
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return x


model = NASNet().to(device)
summary(model, (3, 32, 32))

# --- 데이터셋 준비, 학습 실행
transform = transforms.Compose([
    transforms.Resize((40, 40)),
    transforms.RandomCrop((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

epochs = 100

# 손실 함수 및 optimizer 설정
criterion = nn.CrossEntropyLoss()
from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
# optimizer = optim.RMSprop(model.parameters(), lr=0.1, weight_decay=0.9, eps=1.0)  # 학습불안정, 안됨
lr_scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


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

        with tqdm(train_loader, unit="batch", ncols=100) as tepoch:  # tqdm 사용
            tepoch.set_description(f"Epoch {epoch + 1}/{num_epochs}")
            for inputs, labels in tepoch:
                inputs, labels = inputs.to(device), labels.to(device)  # GPU로 이미지, 라벨 넘김

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
            val_accuracy, val_loss = evaluate_model(model, criterion, test_loader, device)
            # topk_accuracy = {k: topk_corr / topk_tot for k, (topk_corr, topk_tot) in zip(topk_correct.values(), topk_total.values())}
            # 리스트에 추가
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
            val_losses.append(val_loss)

            print(
                f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            # for k, acc in topk_accuracy.items():
            #     writer.add_scalar(f'Top{k}/train', acc, epoch)
            # TensorBoard에 로그 기록
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Accuracy/train', train_accuracy, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/val', val_accuracy, epoch)

        # 에포크가 끝날 때마다 스케줄러를 업데이트
        lr_scheduler.step()

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
with tqdm(total=len(test_loader), unit="batch", ncols=100, desc="Testing") as pbar:
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

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