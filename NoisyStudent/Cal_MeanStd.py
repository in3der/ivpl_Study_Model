from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
import torch
from tqdm import tqdm

# 하이퍼파라미터
batch_size = 64

# 데이터 전처리
data_transforms = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
])

# 미리 정의된 경로
dataset_dir1 = '/home/ivpl-d29/dataset/cifar100/CIFAR100_extracted'
dataset_dir2 = '/home/ivpl-d29/dataset/tiny-imagenet-200/balanced_data'  # 두 번째 데이터셋 경로

# 데이터셋의 개수 입력 받기
dataset_count = int(input('계산할 데이터셋의 개수를 구하시오 (1 또는 2 입력): '))

if dataset_count == 1:
    # 1개의 데이터셋 로드
    dataset = datasets.ImageFolder(root=dataset_dir1, transform=data_transforms)

elif dataset_count == 2:
    # 2개의 데이터셋 로드
    dataset1 = datasets.ImageFolder(root=dataset_dir1, transform=data_transforms)
    dataset2 = datasets.ImageFolder(root=dataset_dir2, transform=data_transforms)

    # 두 데이터셋을 결합
    dataset = ConcatDataset([dataset1, dataset2])

else:
    raise ValueError("잘못된 입력입니다. 1 또는 2를 입력하세요.")

# DataLoader 설정
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


def calculate_mean_std(data_loader):
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = 0

    # tqdm 적용: total=len(data_loader)는 전체 배치 수를 계산합니다.
    for images, _ in tqdm(data_loader, desc="Calculating Mean and Std", total=len(data_loader)):
        batch_mean = images.mean([0, 2, 3])
        batch_std = images.std([0, 2, 3])

        mean += batch_mean * images.size(0)
        std += batch_std * images.size(0)
        total_images += images.size(0)

    mean /= total_images
    std /= total_images

    return mean, std


# Mean and Std 계산
mean, std = calculate_mean_std(data_loader)

print("Mean:", mean)
print("Std:", std)
