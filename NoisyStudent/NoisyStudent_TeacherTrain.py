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
from collections import Counter
from torchvision.ops import StochasticDepth

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device : {device}')

# 시드 설정- 재현가능하도록
seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 로그를 저장할 logs 폴더 경로
logs_dir = '/home/ivpl-d29/myProject/Study_Model/NoisyStudent/logs'

# SummaryWriter 생성
writer = SummaryWriter(logs_dir)

# Teacher model 준비 - EfficientNetB0
import torchvision.models as models
model = models.efficientnet_b0(pretrained=False)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(model.classifier[1].in_features, 100)
)
model.to(device)
summary(model, (3, 64, 64))
print(model)

# 시드 설정- 재현가능하도록
seed = 42
torch.manual_seed(seed)

# --- 데이터셋 준비
data_transforms = transforms.Compose([
    #transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # 논문Table6- 5% 범위 내의 random translation (Both)
    transforms.RandomHorizontalFlip(),  # 논문Table6- Random flip (Both)
    # transforms.RandAugment(num_ops=2, magnitude=27),  # 논문 - RandAugment (only Student Train)
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5080, 0.4875, 0.4418), (0.2487, 0.2418, 0.2558)),
    #transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])
testdata_transforms = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5080, 0.4875, 0.4418), (0.2487, 0.2418, 0.2558))
    #transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

# 데이터 경로 설정
data_dir = '/home/ivpl-d29/dataset/cifar100'

train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=data_transforms)
test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=testdata_transforms)

train_size = int(0.8 * len(train_dataset))
val_size = int(0.2 * len(train_dataset))

train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# DataLoader 설정
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

# 데이터셋 개수 출력
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")

# 클래스별 이미지 수 출력 함수
def print_class_distribution(dataset, dataset_name):
    labels = [sample[1] for sample in dataset]  # 라벨 추출
    counter = Counter(labels)
    print(f"\nClass distribution in {dataset_name}:")
    for class_id, count in sorted(counter.items()):
        print(f"Class {class_id}: {count} images")

# 클래스별 이미지 수 출력
print_class_distribution(train_dataset, "training set")
print_class_distribution(val_dataset, "validation set")
print_class_distribution(test_dataset, "test set")


# ---- 학습 준비
# epochs = 350  # 논문- B4이상 700, 미만은 350 epochs
epochs = 150

# 손실 함수 및 optimizer 설정
criterion = nn.CrossEntropyLoss()
from torch.optim.lr_scheduler import ExponentialLR
#optimizer = optim.SGD(model.parameters(), lr=0.128, weight_decay=5e-4, momentum=0.9, nesterov=True)
optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=0.01, momentum=0.9, nesterov=True)
# 논문 - epochs 350 + 2.4(step size) + 0.97(rate)
# 논문 - epochs 700 + 4.8(step size) + 0.97(rate)
step_size = 1.2
decay_rate = 0.97
gamma = decay_rate ** (1 / step_size)
lr_scheduler = ExponentialLR(optimizer, gamma=gamma)

# 모델 훈련 함수
def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=epochs, topk=5):
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

            # 50% 이상의 성능이 나오고, 현재 epoch이 최고의 성능일 때만 저장
            if val_accuracy > best_accuracy and val_accuracy >= 0.50:
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

# 모델 훈련
train_losses, val_losses, train_accuracies, val_accuracies, topk_accuracies, learning_rates = train_model(model, criterion, optimizer, lr_scheduler)
# 모델 테스트
test_loss, test_accuracy, test_topk_accuracy = test_model(model, criterion, test_loader, device)


# ----그래프 그리기
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
plt.plot(topk_accuracies, label=f'Top-5 validation accuracy')
plt.xlabel('Epoch')
plt.ylabel(f'Top-5 Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('output_image.png')
