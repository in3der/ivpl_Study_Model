import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, TensorDataset, random_split, ConcatDataset
import os
import random
from collections import defaultdict
from tqdm import tqdm
import pandas as pd  # pandas를 사용하여 CSV 파일로 Pseudo-Label 저장 및 로드
import torchvision.models as models
import csv
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
from torchsummary import summary
import matplotlib.pyplot as plt
from PIL import Image
import shutil
from collections import Counter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device : {device}')

# 시드 설정- 재현가능하도록
seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# -----1. Teacher 모델 로드 및 테스트 정확도 평가
# 저장된 모델 가중치 로드
model_load_path = '/home/ivpl-d29/myProject/Study_Model/NoisyStudent/logs/model/model_epoch150_acc47.pth'
model = torch.load(model_load_path, map_location=device)
model.to(device)

# 데이터셋 준비
data_transforms = transforms.Compose([
    #transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # 논문Table6- 5% 범위 내의 random translation (Both)
    transforms.RandomHorizontalFlip(),  # 논문Table6- Random flip (Both)
    # transforms.RandAugment(num_ops=2, magnitude=27),  # 논문 - RandAugment (only Student Train)
    transforms.Resize(64),
    transforms.ToTensor(),
    #transforms.Normalize((0.5080, 0.4875, 0.4418), (0.2487, 0.2418, 0.2558)), # Teacher1 이후
    # transforms.Normalize((0.5030, 0.4793, 0.4316), (0.2538, 0.2466, 0.2595)),    # Student 1 이후
    transforms.Normalize((0.5085, 0.4863, 0.4390), (0.2520, 0.2447, 0.2590)),   # Student 2 이후
    # transforms.Normalize((0.5073, 0.4859, 0.4392), (0.2518, 0.2450, 0.2589)),     # Student 3 이후
])
testdata_transforms = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5080, 0.4875, 0.4418), (0.2487, 0.2418, 0.2558))
])

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

# 모델 평가 함수
def evaluate_model(model, data_loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    accuracy = correct / total
    return accuracy

# Test 데이터셋으로 정확도 평가
test_accuracy = evaluate_model(model, test_loader)
print(f"Test Accuracy: {test_accuracy:.4f}")


# -----2. pseudo label을 생성할 데이터셋 로드
# 데이터셋 경로
dataset_dir = '/home/ivpl-d29/dataset/tiny-imagenet-200/train'

# 클래스별 이미지 개수 저장
tiny_imagenet_class_image_count = defaultdict(int)
# 각 클래스별 이미지 수를 계산
for class_name in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_name, 'images')
    if os.path.isdir(class_path):
        num_images = len(os.listdir(class_path))
        tiny_imagenet_class_image_count[class_name] = num_images

# 결과 프린트문으로 출력
# for class_name, count in class_image_count.items():
#     print(f'Class {class_name} has {count} images.')

# 클래스별 이미지 수 그래프로 시각화
plt.figure(figsize=(20, 8))
plt.scatter(tiny_imagenet_class_image_count.keys(), tiny_imagenet_class_image_count.values(), color='blue', marker='o')
plt.xlabel('Class Name')
plt.ylabel('Number of Images')
plt.title('Number of Images per Class(Tiny Imagenet)')
plt.xticks(rotation=90)
plt.savefig('1. class_per_image_before_filtering.png', bbox_inches='tight')
plt.close()

# -----3. Confidence기반 Filtering
model.eval()
# CSV 파일로 저장할 경로
output_csv_path = '/home/ivpl-d29/myProject/Study_Model/NoisyStudent/pseudo_labels.csv'

# CSV 파일 생성
print('confidence 계산, csv 파일 생성 시작')
# 전체 이미지 수를 계산
total_images = sum(len(files) for _, _, files in os.walk(dataset_dir))

with open(output_csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image Path', 'Confidence', 'Soft Pseudo Label', 'Hard Pseudo Label'])

    # tqdm으로 진행 상황 추적 (전체 이미지 개수 포함)
    labeled_images = 0  # Labeled 완료한 이미지 수 추적 변수

    with tqdm(total=total_images, desc=f"Processing: Labeled 0/{total_images}", unit="image") as pbar:
        # 모든 클래스에 대해 반복
        for class_name in os.listdir(dataset_dir):
            class_path = os.path.join(dataset_dir, class_name, 'images')
            if os.path.isdir(class_path):
                # 각 이미지에 대해 pseudo label 및 confidence 계산
                for image_name in os.listdir(class_path):
                    image_path = os.path.join(class_path, image_name)

                    # 이미지 로드 및 전처리 (transform 필요)
                    image = Image.open(image_path).convert('RGB')
                    image = testdata_transforms(image).unsqueeze(0).to(device)  # 전처리 후 batch 차원 추가

                    # 모델 예측
                    with torch.no_grad():
                        outputs = model(image)
                        softmax_outputs = F.softmax(outputs, dim=1)
                        confidence, hard_pseudo_label = torch.max(softmax_outputs, 1)

                    # 결과 csv에 저장 (이미지 경로, confidence값, soft pseudo label 리스트, hard pseudo label 값)
                    soft_pseudo_label = softmax_outputs.squeeze().cpu().tolist()  # soft pseudo 값들 리스트로 저장
                    writer.writerow([image_path, confidence.item(), soft_pseudo_label, hard_pseudo_label.item()])

                    labeled_images += 1  # Labeled된 이미지 수 증가, 진행상황 추적
                    pbar.set_description(f"Processing: Labeled {labeled_images}/{total_images}")
                    pbar.update(1)

# CSV 파일 로드
print("완성된 csv 파일 로드")
pseudo_labels_df = pd.read_csv(output_csv_path)

# Confidence 0.3 이상인 데이터만 필터링
print("confidence 0.3 기준 필터링")
pseudo_labels_df['Confidence'] = pd.to_numeric(pseudo_labels_df['Confidence'], errors='coerce')
filtered_df = pseudo_labels_df[pseudo_labels_df['Confidence'] >= 0.3]

# 필터링된 데이터 저장
filtered_csv_path = '/home/ivpl-d29/myProject/Study_Model/NoisyStudent/filtered_pseudo_labels.csv'
filtered_df.to_csv(filtered_csv_path, index=False)

# 필터링 후 클래스별 이미지 수 계산 (hard pseudo label 기준으로)
filtered_class_image_count = filtered_df['Hard Pseudo Label'].value_counts()

print("필터링 후 각 클래스별 이미지 수 (Hard Pseudo Label 기준):")
for class_id, count in filtered_class_image_count.items():
    print(f"Class {class_id}: {count} images")

# 클래스별 이미지 수 그래프로 시각화
plt.figure(figsize=(20, 8))
plt.scatter(filtered_class_image_count.index, filtered_class_image_count.values, color='blue', marker='o')
plt.xlabel('Hard Pseudo Label')
plt.ylabel('Number of Images')
plt.title('2. Number of Images per Class After Filtering (Hard Pseudo Label)')
plt.yscale('log')  # y축을 로그 스케일로 설정
plt.xticks(rotation=90)
plt.gca().set_xticks(filtered_class_image_count.index)  # x축의 틱 위치를 클래스 ID로 설정
plt.gca().set_xticklabels([f'Class {x}' for x in filtered_class_image_count.index], rotation=90, fontsize=8)  # 레이블을 클래스 ID로 설정
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('2. class_per_image_after_filtering_hard_pseudo_label.png', bbox_inches='tight')
plt.close()
print("2. class_per_image_after_filtering_hard_pseudo_label.png 저장 완료")



# -----4. Balancing (각 class(Cifar-100 기준)당 이미지 1000장)
print('Balancing 시작')

target_class_size = 1000
filtered_df = pd.read_csv(filtered_csv_path)

# 원본 데이터셋 경로
original_data_dir = '/home/ivpl-d29/dataset/tiny-imagenet-200/train'
# 결과 저장할 폴더
balanced_data_dir = '/home/ivpl-d29/dataset/tiny-imagenet-200/balanced_data3'
os.makedirs(balanced_data_dir, exist_ok=True)

# CIFAR-100 labels and paths
CIFAR100LABELS = ["apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
                  "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can",
                  "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock", "cloud",
                  "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur", "dolphin",
                  "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo",
                  "keyboard", "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man",
                  "maple_tree", "motorcycle", "mountain", "mouse", "mushroom", "oak_tree", "orange",
                  "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree", "plain",
                  "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road",
                  "rocket", "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper",
                  "snail", "snake", "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper",
                  "table", "tank", "telephone", "television", "tiger", "tractor", "train", "trout",
                  "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"]

cifar100_origin_path = '/home/ivpl-d29/dataset/cifar100/CIFAR100_extracted/'


# 각 클래스별 이미지 경로 수집
class_images = defaultdict(list)

# 모든 클래스에 대해 이미지 경로 수집 및 0개인 클래스도 확인
all_classes = set(range(100))  # CIFAR-100 클래스 ID: 0부터 99
filtered_classes = set(filtered_df['Hard Pseudo Label'].unique())  # 필터링된 클래스
missing_classes = all_classes - filtered_classes  # 필터링 후 이미지가 없는 클래스들

# CIFAR-100에서 이미지를 가져와서 복사
for missing_class in missing_classes:
    class_name = f'class_{missing_class:03d}'  # 숫자 형식을 'class_001' 형태로 변환
    class_dir = os.path.join(balanced_data_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)

    # 해당 클래스에 맞는 CIFAR-100 라벨을 찾음
    cifar_class_name = CIFAR100LABELS[missing_class]
    cifar_class_path = os.path.join(cifar100_origin_path, cifar_class_name)

    if not os.path.exists(cifar_class_path):
        print(f"Warning: {cifar_class_path} 존재하지 않음.")
        continue

    # CIFAR-100 이미지를 해당 클래스 디렉토리로 복사
    cifar_images = [os.path.join(cifar_class_path, img) for img in os.listdir(cifar_class_path) if
                    os.path.isfile(os.path.join(cifar_class_path, img))]

    for img_path in cifar_images:
        if os.path.isfile(img_path):
            shutil.copy(img_path, class_dir)
        else:
            print(f"Warning: {img_path} 존재하지 않음.")

print("CIFAR-100 데이터셋에서 누락된 클래스 이미지 복사 완료.")

for index, row in filtered_df.iterrows():
    hard_pseudo_label = row['Hard Pseudo Label']
    image_path = row['Image Path']
    class_images[hard_pseudo_label].append(image_path)

# Balancing: 각 클래스별로 1000장의 이미지 준비 (중복 허용)
for class_id, images in class_images.items():
    class_name = f'class_{class_id:03d}'
    class_dir = os.path.join(balanced_data_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)

    # 해당 클래스의 이미지를 가져옴 (필터링된 이미지가 있을 경우 사용)
    images = class_images[class_id]
    #print(f"Processing class {class_name}...")  # 클래스 작업 시작 로그 출력
    num_images = len(images)

    # Step 1: 이미지가 1000장보다 많으면 랜덤하게 삭제
    if num_images > target_class_size:
        selected_images = np.random.choice(images, target_class_size, replace=False).tolist()
    else:
        selected_images = images

    # Step 2: CIFAR-100에서 부족한 이미지 채우기
    if len(selected_images) < target_class_size:
        # CIFAR-100 이미지 경로 수집
        cifar_class_name = CIFAR100LABELS[class_id]
        cifar_class_path = os.path.join(cifar100_origin_path, cifar_class_name)
        if not os.path.exists(cifar_class_path):
            print(f"Warning: {cifar_class_path} does not exist.")
            continue
        cifar_images = [os.path.join(cifar_class_path, img) for img in os.listdir(cifar_class_path) if
                        os.path.isfile(os.path.join(cifar_class_path, img))]
        # 필요한 만큼 CIFAR-100 이미지 추가 (중복 허용)
        additional_images = np.random.choice(cifar_images, target_class_size - len(selected_images),
                                             replace=True).tolist()
        selected_images += additional_images
    # Step 3: 이미지 복사
    for img_path in selected_images:
        if os.path.isfile(img_path):
            shutil.copy(img_path, class_dir)
        else:
            print(f"Warning: {img_path} does not exist.")
print("Balancing 완료 및 이미지 복제 완료")

# 클래스별 이미지 개수 계산 및 시각화
class_image_count_after_balancing = defaultdict(int)
for class_name in os.listdir(balanced_data_dir):
    class_path = os.path.join(balanced_data_dir, class_name)
    if os.path.isdir(class_path):
        num_images = len(os.listdir(class_path))
        class_image_count_after_balancing[class_name] = num_images

# DataFrame 변환 및 클래스명 정렬
class_image_count_after_balancing_df = pd.DataFrame.from_dict(class_image_count_after_balancing, orient='index', columns=['Number of Images'])
class_image_count_after_balancing_df = class_image_count_after_balancing_df.sort_index()  # 클래스 이름을 기준으로 정렬

# 그래프 시각화
plt.figure(figsize=(20, 8))
plt.scatter(class_image_count_after_balancing_df.index, class_image_count_after_balancing_df['Number of Images'], color='blue', marker='o')
plt.xlabel('Class Name')
plt.ylabel('Number of Images')
plt.title('3. Number of Images per Class After Balancing')
plt.xticks(rotation=90)
plt.grid(True)
plt.savefig('3. class_per_image_after_balancing.png', bbox_inches='tight')
plt.close()
print("3. class_per_image_after_balancing.png 저장 완료")


# -----5. Student 학습 전 데이터셋 준비 -----

# CIFAR-100 데이터셋 로드
data_dir = '/home/ivpl-d29/dataset/cifar100'

train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=data_transforms)
test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=testdata_transforms)

# 학습 및 검증 데이터셋 분할
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size

train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# 기존 CIFAR-100 데이터셋의 개수 출력
print("기존 CIFAR-100 데이터셋 개수:")
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")


# Pseudo-labeled dataset 생성 (hard pseudo label만 사용)
pseudo_labeled_images = []

print('train loader에 들어가는지 확인')
pseudo_labeled_data_dir = balanced_data_dir
for root, _, files in os.walk(pseudo_labeled_data_dir):
    for file in files:
        image_path = os.path.join(root, file)
        pseudo_labeled_images.append(image_path)

# Pseudo-labeled 데이터셋 클래스 정의
class PseudoLabeledDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')  # 이미지 불러오기
        if self.transform:
            image = self.transform(image)
        return image  # hard label은 현재 필요하지 않으므로 반환 안함

pseudo_labeled_dataset = PseudoLabeledDataset(pseudo_labeled_images, transform=data_transforms)

# 기존 labeled 데이터셋과 pseudo-labeled 데이터셋을 결합
combined_train_dataset = ConcatDataset([train_dataset, pseudo_labeled_dataset])

# DataLoader 설정
train_loader = DataLoader(combined_train_dataset, batch_size=64, shuffle=True, num_workers=4)
print(f"Total number of training samples (after combining labeled and pseudo-labeled): {len(combined_train_dataset)}")


# 필터링된 및 balancing된 데이터셋 개수 계산
def count_images_in_directory(directory):
    count = 0
    for root, _, files in os.walk(directory):
        count += len(files)
    return count

print("Pseudo labeling 및 balancing 완료한 데이터셋 개수:")

num_balanced_images = count_images_in_directory(pseudo_labeled_data_dir)
print(f"Number of images in balanced dataset: {num_balanced_images}")

# 최종 학습 데이터의 개수 출력
train_dataset_labeled_size = len(train_dataset)
train_dataset_pseudo_labeled_size = num_balanced_images
val_dataset_labeled_size = len(val_dataset)
test_dataset_labeled_size = len(test_dataset)

total_training_size = train_dataset_labeled_size + train_dataset_pseudo_labeled_size

print("\n최종 학습을 위한 데이터 개수:")
print(f"Number of labeled training samples: {train_dataset_labeled_size}")
print(f"Number of pseudo-labeled images after balancing: {train_dataset_pseudo_labeled_size}")
print(f"Total number of training samples (labeled + pseudo-labeled): {total_training_size}")
print(f"Number of validation samples: {val_dataset_labeled_size}")
print(f"Number of test samples: {test_dataset_labeled_size}")


# Validation 및 Test DataLoader 설정
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
