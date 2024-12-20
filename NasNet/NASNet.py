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

class SepConv2d1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, padding=1, bias=False):
        super(SepConv2d1, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, out_channels, kernel, stride=stride, padding=padding, bias=bias)
        self.depthwise_s1 = nn.Conv2d(out_channels, out_channels, kernel, stride=1, padding=padding, bias=bias)
        self.pointwise = nn.Conv2d(out_channels, out_channels, 1, stride=1, bias=bias)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1)
    def forward(self, x):
        x = self.relu(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.depthwise_s1(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x

class SepConv2d(nn.Module):  # Separable Convolution 2D, 논문 Appendix.A.4. 참고
    def __init__(self, in_channels, out_channels, kernel, stride=1, padding=1, bias=False):
        super(SepConv2d, self).__init__()
        # depthwise convolution : 각 입력 채널에 대해 독립적으로 필터링 수행 (입력 채널 수 = 출력 채널 수)
        # groups=in_channels 설정하여 입력 채널 수로 그룹을 나누어 각 그룹에 대해 독립적 필터링 수행
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel, stride=stride, padding=padding, bias=bias, groups=in_channels)
        self.depthwise_s1 = nn.Conv2d(in_channels, in_channels, kernel, stride=1, padding=padding, bias=bias, groups=in_channels)
        # pointwise convolution : 1x1 convolution 수행하여 채널 수를 변환.
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=bias)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1)
    def forward(self, x):
        x = self.relu(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.depthwise_s1(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x

class ReductionCell1(nn.Module):
    def __init__(self, in_channels=96, out_channels=42):     # 96, 42
        super(ReductionCell1, self).__init__()
        self.prev1X1_cur = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.SepConv3x3 = SepConv2d(out_channels, out_channels, kernel=3, stride=1, padding=1)
        self.SepConv5x5 = SepConv2d(out_channels, out_channels, kernel=5, stride=2, padding=2)
        self.SepConv5x5_out = SepConv2d1(in_channels, out_channels, kernel=5, stride=2, padding=2)
        self.SepConv7x7 = SepConv2d1(in_channels, out_channels, kernel=7, stride=2, padding=3)
        self.Avg3x3 = AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.Avg3x3_1 = AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.Max3x3 = MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1)
        self.relu = nn.ReLU()
    def forward(self, prev):
        cur = self.bn(self.prev1X1_cur(self.relu(prev)))
        b0 = self.SepConv7x7(prev) + self.SepConv5x5(cur)
        b1 = self.Max3x3(cur) + self.SepConv7x7(prev)
        b2 = self.Avg3x3(cur) + self.SepConv5x5_out(prev)
        b3 = self.Max3x3(cur) + self.SepConv3x3(b0)
        b4 = self.Avg3x3_1(b0) + b1
        out = torch.cat([b1, b2, b3, b4], dim=1)
        '''
        print(f'After b0: {b0.shape}')
        print(f'After b1: {b1.shape}')
        print(f'After b2: {b2.shape}')
        print(f'After b3: {b3.shape}')
        print(f'After b4: {b4.shape}')
        print(f'out Reduction shape :  {out.shape}')
        '''
        return out

class ReductionCell2(nn.Module):
    def __init__(self, inPrev=96, outPrev=42, inCur=168, outCur=84): # 96, 42, 168, 84
        super(ReductionCell2, self).__init__()
        self.prev1X1_cur = nn.Conv2d(inCur, outCur, kernel_size=1)
        self.prev1X1_prev = nn.Conv2d(inPrev, outPrev*2, kernel_size=1, stride=2)
        self.SepConv3x3Prev = SepConv2d(inPrev, outPrev, kernel=3, stride=1, padding=1)
        self.SepConv3x3 = SepConv2d(84, 84, kernel=3, stride=1, padding=1)
        self.SepConv5x5 = SepConv2d(84, 84, kernel=5, stride=2, padding=2)
        self.SepConv7x7 = SepConv2d(84, 84, kernel=7, stride=2, padding=3)
        self.SepConv7x7Cur = SepConv2d(inCur, outCur, kernel=7, stride=2, padding=3)
        self.Avg3x3 = AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.Avg3x3_1 = AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.Max3x3 = MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(84, eps=0.001, momentum=0.1)
        self.relu = nn.ReLU()
    def forward(self, prev, cur):
        cur = self.bn(self.prev1X1_cur(self.relu(cur)))
        prev = self.bn(self.prev1X1_prev(self.relu(prev)))
        b0 = self.SepConv7x7(prev) + self.SepConv5x5(cur)
        b1 = self.Max3x3(cur) + self.SepConv7x7(prev)
        b2 = self.Avg3x3(cur) + self.SepConv5x5(prev)
        b3 = self.Max3x3(cur) + self.SepConv3x3(b0)
        b4 = self.Avg3x3_1(b0) + b1
        out = torch.cat([b1, b2, b3, b4], dim=1)
        return out


class FirstNormalCell(nn.Module):
    def __init__(self, inPrev, outPrev, inCur, outCur):
        super(FirstNormalCell, self).__init__()
        self.avg = nn.AvgPool2d(1, 2)
        self.prev1X1 = nn.Conv2d(inPrev, outCur, kernel_size=1)
        self.cur1X1 = nn.Conv2d(inCur, outCur, kernel_size=1)
        self.SepConv3x3 = SepConv2d(outCur, outCur, kernel=3, stride=1, padding=1)
        self.SepConv3x3Cur = SepConv2d(outCur, outCur, kernel=3, stride=1, padding=1)
        self.SepConv5x5 = SepConv2d(outCur, outCur, kernel=5, stride=1, padding=2)
        self.SepConv5x5Cur = SepConv2d(outCur, outCur, kernel=5, stride=1, padding=2)
        self.Avg3x3 = AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.bnPrev = nn.BatchNorm2d(outCur, eps=0.001, momentum=0.1)
        self.bnCur = nn.BatchNorm2d(outCur, eps=0.001, momentum=0.1)
        self.relu = nn.ReLU()
    def forward(self, prev, cur):

        prev = self.bnPrev(self.prev1X1(self.avg(prev)))
        cur = self.bnCur(self.cur1X1(self.relu(cur)))
        b0 = self.SepConv3x3(cur) + cur
        b1 = self.SepConv3x3(prev) + self.SepConv5x5(prev)
        b2 = self.Avg3x3(cur) + prev
        b3 = self.Avg3x3(prev) + self.Avg3x3(prev)
        b4 = self.SepConv5x5(prev) + self.SepConv3x3(prev)
        out = torch.cat([prev, b0, b1, b2, b3, b4], dim=1)
        return out


class NormalCell(nn.Module):
    def __init__(self, inPrev, outPrev, inCur, outCur):
        super(NormalCell, self).__init__()
        self.prev1X1 = nn.Conv2d(inPrev, outCur, kernel_size=1)
        self.cur1X1 = nn.Conv2d(inCur, outCur, kernel_size=1)
        self.SepConv3x3Prev = SepConv2d(outPrev, outPrev, kernel=3, stride=1, padding=1)
        self.SepConv3x3Cur = SepConv2d(outCur, outCur, kernel=3, stride=1, padding=1)
        self.SepConv5x5Prev = SepConv2d(outPrev, outPrev, kernel=5, stride=1, padding=2)
        self.SepConv5x5Cur = SepConv2d(outCur, outCur, kernel=5, stride=1, padding=2)
        self.Avg3x3 = AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.bnPrev = nn.BatchNorm2d(outPrev, eps=0.001, momentum=0.1)
        self.bnCur = nn.BatchNorm2d(outCur, eps=0.001, momentum=0.1)
        self.relu = nn.ReLU()

    def forward(self, prev, cur):
        prev = self.bnPrev(self.prev1X1(self.relu(prev)))
        cur = self.bnCur(self.cur1X1(self.relu(cur)))
        b0 = self.SepConv3x3Cur(cur) + cur
        b1 = self.SepConv3x3Prev(prev) + self.SepConv5x5Prev(prev)
        b2 = self.Avg3x3(cur) + prev
        b3 = self.Avg3x3(prev) + self.Avg3x3(prev)
        b4 = self.SepConv5x5Prev(prev) + self.SepConv3x3Prev(prev)
        out = torch.cat([prev, b0, b1, b2, b3, b4], dim=1)
        return out

class ReductionCell(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ReductionCell, self).__init__()
        self.prev1X1_cur = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.prev1X1_prev = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.SepConv3x3 = SepConv2d(out_channels, out_channels, kernel=3, stride=1, padding=1)
        self.SepConv5x5 = SepConv2d(out_channels, out_channels, kernel=5, stride=2, padding=2)
        self.SepConv7x7 = SepConv2d(out_channels, out_channels, kernel=7, stride=2, padding=3)
        self.Avg3x3 = AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.Avg3x3_1 = AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.Max3x3 = MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.Conv1x1 = nn.Conv2d(in_channels//3, out_channels, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1)
        self.relu = nn.ReLU()

    def forward(self, prev, cur):
        cur = self.bn(self.prev1X1_cur(self.relu(cur)))
        prev = self.bn(self.prev1X1_prev(self.relu(prev)))
        b0 = self.SepConv7x7(prev) + self.SepConv5x5(cur)
        b1 = self.Max3x3(cur) + self.SepConv7x7(prev)
        b2 = self.Avg3x3(cur) + self.SepConv5x5(prev)
        b3 = self.Max3x3(cur) + self.SepConv3x3(b0)
        b4 = self.Avg3x3_1(b0) + b1
        out = torch.cat([b1, b2, b3, b4], dim=1)
        return out


class NASNet(nn.Module):  # NASNet-A 6@4032 for ImageNet (3->96->42->84->168(1008)->336->336(2016)->672->672(4032))
    def __init__(self, in_channels=3, num_classes=1000, N=6, filters=4032):
        super(NASNet, self).__init__()
        # 3x3 conv, stride 2
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )
        self.reduction_cell1 = ReductionCell1(96, 42)   # 96, 96, 96, 42
        self.reduction_cell2 = ReductionCell2(96, 42, 168, 84)   # 96, 42, 168, 84
        self.cells = nn.ModuleList()
        self.cells.append(FirstNormalCell(168, 84, 336, 168))       # 168, 84, 336, 168
        self.cells.append(NormalCell(336, 168, 1008, 168))       # 336, 168, 1008, 168
        for _ in range(1, N - 2):
            self.cells.append(NormalCell(1008, 168, 1008, 168))  # 1008, 168, 1008, 168

        self.cells.append(ReductionCell(1008, 336))                  # 1008, 336, 1008, 336

        self.cells.append(FirstNormalCell(1008, 168, 1344, 336))# 1008, 168, 1344, 336
        self.cells.append(NormalCell(1344, 336, 2016, 336))     # 1344, 336, 2016, 336
        for _ in range(1, N - 2):
            self.cells.append(NormalCell(2016, 336, 2016, 336)) # 2016, 336, 2016, 336

        self.cells.append(ReductionCell(2016, 672))                  # 2016, 672, 2016, 672

        self.cells.append(FirstNormalCell(2016, 336, 2688, 672))# 2016, 336, 2688, 672
        self.cells.append(NormalCell(2688, 672, 4032, 672))     # 2688, 672, 4032, 672
        for _ in range(1, N - 2):
            self.cells.append(NormalCell(4032, 672, 4032, 672))  # 4032, 672, 4032, 672
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filters, num_classes)  # 4032, 1000
        self.dropout = nn.Dropout(0.5)
        self.bn = nn.BatchNorm2d(in_channels, eps=0.001, momentum=0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_stem_conv = self.stem(x)
        prev_prev_x = self.reduction_cell1(x_stem_conv)
        prev_x = self.reduction_cell2(x_stem_conv, prev_prev_x)
        for cell in self.cells:
            x = cell(prev_prev_x, prev_x)
            prev_prev_x = prev_x
            prev_x = x
        x = self.avg(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return x

model = NASNet().to(device)
summary(model, (3, 331, 331))


# --- 데이터셋 준비, 학습 실행
data_transforms = transforms.Compose([
    transforms.Resize((354, 354)),  # 임의로 resize 354 지정
    transforms.RandomCrop(331),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# 데이터셋 로드
data_dir = '/home/ivpl-d29/dataset/imagenet'
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=data_transforms)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=data_transforms)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=data_transforms)

# 데이터 로더
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, drop_last=True, pin_memory=True, prefetch_factor=8)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=4, drop_last=True, pin_memory=True, prefetch_factor=8)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=4, drop_last=True, pin_memory=True, prefetch_factor=8)

epochs = 6

# 손실 함수 및 optimizer 설정
criterion = nn.CrossEntropyLoss()
from torch.optim.lr_scheduler import CosineAnnealingLR
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
# optimizer = optim.RMSprop(model.parameters(), lr=0.1, weight_decay=0.9, eps=1.0)
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