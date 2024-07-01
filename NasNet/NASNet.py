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


class SepConv2d(nn.Module): # Separable Convolution 2D, 논문 Appendix.A.4. 참고
    def __init__(self, in_channels, out_channels, kernel, stride, padding, bias=False):
        super(SepConv2d, self).__init__()
        # depthwise convolution : 각 입력 채널에 대해 독립적으로 필터링 수행 (입력 채널 수 = 출력 채널 수)
        # groups=in_channels 설정하여 입력 채널 수로 그룹을 나누어 각 그룹에 대해 독립적 필터링 수행
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel, stride=stride, padding=padding, bias=bias, groups=in_channels)
        # pointwise convolution : 1x1 convolution 수행하여 채널 수를 변환.
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=bias)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        #x = self.relu(x)
        #x = self.depthwise(x)
        #x = self.pointwise(x)
        return x

class Resize(nn.Module):
    def __init__(self, in_channels_x, in_channels_p, out_channels):
        super(Resize, self).__init__()
        self.conv_left = nn.Conv2d(in_channels_x, out_channels, 1, stride=1, bias=False)
        self.bn_left = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1)
        self.conv_right = nn.Conv2d(in_channels_p, out_channels, 1, stride=1, bias=False)
        self.bn_right = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1)
        self.relu = nn.ReLU()
    def forward(self, x, p):
        # Process x
        x = self.relu(x)
        x = self.conv_left(x)
        x = self.bn_left(x)

        # Process p
        p = self.relu(p)
        p = self.conv_right(p)
        p = self.bn_right(p)
        return x, p



class NormalCell(nn.Module):
    def __init__(self, in_channels_x, in_channels_p, out_channels, resize=Resize):
        super(NormalCell, self).__init__()
        self.SepConv3x3 = SepConv2d(out_channels, out_channels, kernel=3, stride=1, padding=1)
        self.SepConv5x5 = SepConv2d(out_channels, out_channels, kernel=5, stride=1, padding=2)
        self.Avg3x3 = AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.resize = resize(in_channels_x, in_channels_p, out_channels)
        self.final_conv = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, x, p):
        # h_j = x(현재 hidden state)
        # h_{j-1} = p(revious)(이전 hidden state)
        x, p = self.resize(x, p)
        b0 = self.SepConv3x3(x) + x
        b1 = self.SepConv3x3(p) + self.SepConv5x5(x)
        b2 = self.Avg3x3(x) + p
        b3 = self.Avg3x3(p) + self.Avg3x3(p)
        b4 = self.SepConv5x5(p) + self.SepConv3x3(p)
        out = torch.cat([b0, b1, b2, b3, b4], dim=1)
        out = self.final_conv(out)  # concat 이후 32*5로 160이 된 채널 수를 32*2=64로 줄임

        return out


class ReductionCell(nn.Module):
    def __init__(self, in_channels_x, in_channels_p, out_channels, resize=Resize):
        super(ReductionCell, self).__init__()
        self.SepConv3x3 = SepConv2d(out_channels, out_channels, kernel=3, stride=1, padding=1)
        self.SepConv5x5 = SepConv2d(out_channels, out_channels, kernel=5, stride=1, padding=2)
        self.SepConv7x7 = SepConv2d(out_channels, out_channels, kernel=7, stride=1, padding=3)
        self.Avg3x3 = AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.Max3x3 = MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.resize = resize(in_channels_x, in_channels_p, out_channels)
        self.final_conv = nn.Conv2d(out_channels * 3, out_channels * 2, kernel_size=1, stride=2)
    def forward(self, x, p):
        x, p = self.resize(x, p)
        b0 = self.SepConv7x7(p) + self.SepConv5x5(x)
        b1 = self.Max3x3(x) + self.SepConv7x7(p)
        b2 = self.Avg3x3(x) + self.SepConv5x5(p)
        b3 = self.Max3x3(x) + self.SepConv3x3(b0)
        b4 = self.Avg3x3(b0) + b1
        out = torch.cat([b2, b3, b4], dim=1)
        out = self.final_conv(out)  # 채널 수 조정, stride 2로 이미지 w,h 절반 줄이기

        return out


class NASNet(nn.Module):    # NASNet-A 6@768 for CIFAR-10
    def __init__(self, in_channels=3, num_classes=10, N=6):
        super(NASNet, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.N = N
        self.init_channels = 32
        self.cells = nn.ModuleList()
        self.cells.append(NormalCell(32, 32, 48))
        for _ in range(1, N):
            self.cells.append(NormalCell(48, 48, 48))
        self.cells.append(ReductionCell(48, 48, 96))
        for _ in range(1, N):
            self.cells.append(NormalCell(192, 192, 192))
        self.cells.append(ReductionCell(192, 192, 384))
        for _ in range(1, N):
            self.cells.append(NormalCell(768, 768, 768))
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(768, num_classes)

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

# Initialize the model and print summary
model = NASNet().to(device)
summary(model, (3, 32, 32))  # Assuming CIFAR-10 image size is 32x32x3





transform = transforms.Compose([
    transforms.Resize((40, 40)),
    transforms.RandomCrop((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

epochs = 10
criterion = nn.CrossEntropyLoss()
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

model.train()
for epoch in range(epochs):
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            pbar.update(1)

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(total=len(test_loader), desc=f"Evaluating", unit="batch") as pbar:
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                pbar.update(1)

    print(f'Epoch: {epoch+1}, Test Accuracy: {100.*correct/total:.2f}%')