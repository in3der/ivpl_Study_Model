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
logs_dir = '/home/ivpl-d29/myProject/Study_Model/NoisyStudent/logs'

# SummaryWriter 생성
writer = SummaryWriter(logs_dir)

# Teacher model 준비 - EfficientNetB0
import torchvision.models.efficientnet as EfficientNet
model = EfficientNet.efficientnet_b0().to(device)
summary(model, (3, 224, 224))

# --- 데이터셋 준비, 학습 실행
data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(256, scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0)),  # 224+32 = 256
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(policy=transforms.autoaugment.AutoAugmentPolicy.IMAGENET,
                           interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

# 데이터셋 로드
data_dir = '/home/ivpl-d29/dataset/imagenet_split/imagenet_small/'
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=data_transforms)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=data_transforms)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=data_transforms)

# 데이터 로더
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, drop_last=True, pin_memory=True, prefetch_factor=8)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4, drop_last=True, pin_memory=True, prefetch_factor=8)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4, drop_last=True, pin_memory=True, prefetch_factor=8)

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
                f"Top-{k} Accuracy: " + ", ".join(f"{val_topk_accuracies[k]:.4f}" for k in topk)
            )

            for k in topk:
                writer.add_scalar(f'Top{k}/train', topk_correct[k] / total, epoch)
                writer.add_scalar(f'Top{k}/val', val_topk_accuracies[k], epoch)

            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Accuracy/train', train_accuracy, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/val', val_accuracy, epoch)
            writer.add_scalar('Learning_Rate', current_lr, epoch)

            # 모델, 가중치, 체크포인트 저장
            model_save_dir = os.path.join(logs_dir, 'model')
            torch.save(model, os.path.join(model_save_dir, f'model_epoch{epoch + 1}_acc{int(val_accuracy * 100):02d}.pth'))
            torch.save(model, os.path.join(model_save_dir, f'model_epoch{epoch + 1}_acc{int(val_accuracy * 100):02d}.pt'))

            # 가중치 저장
            torch.save(model.state_dict(), os.path.join(model_save_dir, f'model_weights_epoch{epoch + 1}_acc{int(val_accuracy * 100):02d}.pth'))
            torch.save(model.state_dict(), os.path.join(model_save_dir, f'model_weights_epoch{epoch + 1}_acc{int(val_accuracy * 100):02d}.pt'))
            # 체크포인트 저장
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }
            torch.save(checkpoint, os.path.join(model_save_dir, f'checkpoint_epoch{epoch + 1}_acc{int(val_accuracy * 100):02d}.pth'))
            torch.save(checkpoint, os.path.join(model_save_dir, f'checkpoint_epoch{epoch + 1}_acc{int(val_accuracy * 100):02d}.pt'))

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
plt.savefig('output_image.png')


# 모델 저장
torch.save(model, os.path.join(logs_dir, 'model/', f'model_epoch{epochs}_acc{int(test_accuracy):d}.pth'))
torch.save(model, os.path.join(logs_dir, 'model/', f'model_epoch{epochs}_acc{int(test_accuracy):d}.pt'))

# 가중치 저장
torch.save(model.state_dict(), os.path.join(logs_dir, 'model/', f'model_weights_epoch{epochs}_acc{int(test_accuracy):d}.pth'))
torch.save(model.state_dict(), os.path.join(logs_dir, 'model/', f'model_weights_epoch{epochs}_acc{int(test_accuracy):d}.pt'))
# 체크포인트 저장
checkpoint = {
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, os.path.join(logs_dir, 'model/', f'checkpoint_epoch{epochs}_acc{int(test_accuracy):d}.pth'))
torch.save(checkpoint, os.path.join(logs_dir, 'model/', f'checkpoint_epoch{epochs}_acc{int(test_accuracy):d}.pt'))
