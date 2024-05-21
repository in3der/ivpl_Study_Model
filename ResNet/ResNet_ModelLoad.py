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

# GPU가 사용 가능한지 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Normalize를 위한 평균과 표준편차
# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]

# 데이터 전처리 및 augmentation
data_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 데이터셋 로드
data_dir = '/home/ivpl-d29/dataset/imagenet'
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=data_transforms)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=data_transforms)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=data_transforms)

# 데이터 로더
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, drop_last=True, pin_memory=True, prefetch_factor=2)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers=4, drop_last=True, pin_memory=True, prefetch_factor=2)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=4, drop_last=True, pin_memory=True, prefetch_factor=2)

# 1. 모델 저장 - torch.save(model, os.path.join(logs_dir+'/model', 'model.pth'.format(epochs)))


# 2. 가중치 저장 - torch.save(model.state_dict(), os.path.join(logs_dir+'/model', 'model_weights.pth'.format(epochs)))


# 3. 체크포인트 저장
# checkpoint = {
#     'epoch': epochs,
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     'loss': loss,
# }
# torch.save(checkpoint, os.path.join(logs_dir+'/model', 'checkpoint.pth'))

