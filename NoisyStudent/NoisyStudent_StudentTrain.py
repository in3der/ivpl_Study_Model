import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset, ConcatDataset
import os
import random
from collections import defaultdict
import shutil
from tqdm import tqdm
import pandas as pd  # pandas를 사용하여 CSV 파일로 Pseudo-Label 저장 및 로드
import torchvision.models as models
from torchsummary import summary
from torchvision.ops import StochasticDepth
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from PIL import Image
import ast

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device : {device}')

# 로그를 저장할 logs 폴더 경로
logs_dir = '/home/ivpl-d29/myProject/Study_Model/NoisyStudent/logs'

# SummaryWriter 생성
writer = SummaryWriter(logs_dir)

# 1. Student Model 로드 - EfficientNet-B7
import torchvision.models as models
model = models.efficientnet_b0(pretrained=False)
model.classifier = nn.Sequential(
    #StochasticDepth(0.2, mode='batch'),  # StochasticDepth 추가 (0.2 비율로 삭제)
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(model.classifier[1].in_features, 100)
)
model.to(device)
summary(model, (3, 64, 64))
print(model)


# 2. 데이터 로드 및 변환
# Transformations for the datasets
data_transforms = transforms.Compose([
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
    transforms.RandomHorizontalFlip(),  # 논문Table6- Random flip (Both)
    transforms.RandAugment(num_ops=2, magnitude=27),  # 논문 - RandAugment (only Student Train)
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5073, 0.4859, 0.4392), (0.2518, 0.2450, 0.2589)),
])

# Test transforms (no augmentations)
testdata_transforms = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5080, 0.4875, 0.4418), (0.2487, 0.2418, 0.2558))
])

# CIFAR-100 dataset
data_dir = '/home/ivpl-d29/dataset/cifar100'

train_dataset_cifar100 = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=data_transforms)
test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=testdata_transforms)

train_size = int(0.8 * len(train_dataset_cifar100))
val_size = len(train_dataset_cifar100) - train_size
train_dataset_cifar100, val_dataset_cifar100 = torch.utils.data.random_split(train_dataset_cifar100,
                                                                             [train_size, val_size])


# ----- Custom PseudoLabeledDataset for soft-labeled data -----
class HardPseudoLabeledDataset(Dataset):
    def __init__(self, image_paths, soft_labels, transform=None):
        self.image_paths = image_paths
        self.hard_labels = [torch.tensor(ast.literal_eval(soft_label)).argmax().item() for soft_label in soft_labels]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        hard_label = self.hard_labels[idx]

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, hard_label


# Load pseudo-labeled data (path and soft labels)
pseudo_data_dir = '/home/ivpl-d29/dataset/tiny-imagenet-200/balanced_data3/'
pseudo_labeled_images = []
pseudo_soft_labels = []


pseudo_label_file = '/home/ivpl-d29/myProject/Study_Model/NoisyStudent/filtered_pseudo_labels.csv'
df_pseudo_labels = pd.read_csv(pseudo_label_file)

for index, row in df_pseudo_labels.iterrows():
    image_path = os.path.join(pseudo_data_dir, row['Image Path'])
    soft_label = row['Soft Pseudo Label']  # Example format: '[0.1, 0.2, 0.7, ...]'
    pseudo_labeled_images.append(image_path)
    pseudo_soft_labels.append(soft_label)

# Create pseudo-labeled dataset
pseudo_labeled_dataset = HardPseudoLabeledDataset(pseudo_labeled_images, pseudo_soft_labels, transform=data_transforms)

# ----- Combine CIFAR-100 and PseudoLabeled datasets -----
combined_train_dataset = ConcatDataset([train_dataset_cifar100, pseudo_labeled_dataset])

# DataLoader setup
train_loader = DataLoader(combined_train_dataset, batch_size=512, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset_cifar100, batch_size=512, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=4)

# Verify the combined dataset sizes
print(f"Number of training samples (CIFAR-100 + Pseudo-Labeled): {len(combined_train_dataset)}")
print(f"Number of validation samples (CIFAR-100): {len(val_dataset_cifar100)}")
print(f"Number of test samples (CIFAR-100): {len(test_dataset)}")

# ---- 학습 준비
# epochs = 350  # 논문- B4이상 700, 미만은 350 epochs
epochs = 150

# 손실 함수 및 optimizer 설정
criterion = nn.CrossEntropyLoss()
from torch.optim.lr_scheduler import ExponentialLR
#optimizer = optim.SGD(model.parameters(), lr=0.128, weight_decay=5e-4, momentum=0.9, nesterov=True)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9, nesterov=True)
# 논문 - epochs 350 + 2.4(step size) + 0.97(rate)
# 논문 - epochs 700 + 4.8(step size) + 0.97(rate)
step_size = 4.8
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
                #torch.save(model, os.path.join(model_save_dir, f'model_epoch{epoch + 1}_acc{int(val_accuracy * 100):02d}.pt'))
                #torch.save(model.state_dict(), os.path.join(model_save_dir, f'model_weights_epoch{epoch + 1}_acc{int(val_accuracy * 100):02d}.pth'))
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                }
                #torch.save(checkpoint, os.path.join(model_save_dir, f'checkpoint_epoch{epoch + 1}_acc{int(val_accuracy * 100):02d}.pth'))

        lr_scheduler.step()
        print(f"Epoch {epoch + 1}: learning rate = {current_lr}")
        if epoch == num_epochs - 1:  # 마지막 epoch일 때만 저장
            model_save_dir = os.path.join(logs_dir, 'model')
            torch.save(model,
                       os.path.join(model_save_dir, f'model_epoch{num_epochs}_acc{int(val_accuracy * 100):02d}.pth'))

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

