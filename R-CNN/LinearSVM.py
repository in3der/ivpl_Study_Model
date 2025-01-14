import time
import copy
import os
import random
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import alexnet
sys.path.append("/home/ivpl-d29/myProject/Study_Model/R-CNN")
from utils.data.custom_classifier_dataset import CustomClassifierDataset
from utils.data.custom_hard_negative_mining_dataset import CustomHardNegativeMiningDataset
from utils.data.custom_batch_sampler import CustomBatchSampler
from utils.util import check_dir
from utils.util import save_model

"""
Hard Negative Mining을 적용한 SVM 기반 classifier 학습 
- SVM의 실제 역할
    - AlexNet의 conv lyaer는 227x227의 input image에서 feature를 추출
    - 추출된 feature들은 FC layer를 통과 
    - 마지막 SVM layer는 이 feature를 바탕으로 '이 이미지 영역에 차가 있다/없다'를 판단
    - Hinge Loss를 통해 margin을 최대화하는 방향으로 학습 

"""


# 정답(positive) 및 오답(negative) 배치 크기 설정
batch_positive = 32
batch_negative = 96
batch_total = 128

def load_data(data_root_dir):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_loaders = {}
    data_sizes = {}
    remain_negative_list = list()
    for name in ['train', 'val']:
        data_dir = os.path.join(data_root_dir, name)

        data_set = CustomClassifierDataset(data_dir, transform=transform)
        if name == 'train':
            """
            Hard Negative Mining 방법 사용
            초기 positive와 negative 비율을 1:1로 설정
            positive sample 수가 negative sample 수보다 훨씬 적으므로, positive sample 수를 기준으로 negative sample 랜덤 추출. 
            """
            positive_list = data_set.get_positives()
            negative_list = data_set.get_negatives()

            init_negative_idxs = random.sample(range(len(negative_list)), len(positive_list))
            init_negative_list = [negative_list[idx] for idx in range(len(negative_list)) if idx in init_negative_idxs]
            remain_negative_list = [negative_list[idx] for idx in range(len(negative_list))
                                    if idx not in init_negative_idxs]

            data_set.set_negative_list(init_negative_list)
            data_loaders['remain'] = remain_negative_list

        sampler = CustomBatchSampler(data_set.get_positive_num(), data_set.get_negative_num(),
                                     batch_positive, batch_negative)

        data_loader = DataLoader(data_set, batch_size=batch_total, sampler=sampler, num_workers=8, drop_last=True)
        data_loaders[name] = data_loader
        data_sizes[name] = len(sampler)
    return data_loaders, data_sizes


def hinge_loss(outputs, labels):
    """
    hinge loss 계산
    :param outputs: (N, num_classes)크기의 출력
    :param labels: (N)크기의 레이블
    :return: 손실값
    """
    num_labels = len(labels)
    corrects = outputs[range(num_labels), labels].unsqueeze(0).T

    # 최대 margin
    margin = 1.0
    margins = outputs - corrects + margin
    loss = torch.sum(torch.max(margins, 1)[0]) / len(labels)

    # # 正则化强度
    # reg = 1e-3
    # loss += reg * torch.sum(weight ** 2)

    return loss


def add_hard_negatives(hard_negative_list, negative_list, add_negative_list):
    # Hard Negative Mining을 통해 추가된 negative sample을 데이터셋에 추가
    for item in hard_negative_list:
        if len(add_negative_list) == 0:
             # 처음 negative sample 추가
            negative_list.append(item)
            add_negative_list.append(list(item['rect']))
        if list(item['rect']) not in add_negative_list:
            negative_list.append(item)
            add_negative_list.append(list(item['rect']))


def get_hard_negatives(preds, cache_dicts):
    """
    예측 결과를 기반으로 Hard Negative 샘플 추출
    """
    fp_mask = preds == 1
    tn_mask = preds == 0

    fp_rects = cache_dicts['rect'][fp_mask].numpy()
    fp_image_ids = cache_dicts['image_id'][fp_mask].numpy()

    tn_rects = cache_dicts['rect'][tn_mask].numpy()
    tn_image_ids = cache_dicts['image_id'][tn_mask].numpy()

    hard_negative_list = [{'rect': fp_rects[idx], 'image_id': fp_image_ids[idx]} for idx in range(len(fp_rects))]
    easy_negatie_list = [{'rect': tn_rects[idx], 'image_id': tn_image_ids[idx]} for idx in range(len(tn_rects))]

    return hard_negative_list, easy_negatie_list


import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2


def visualize_svm_predictions(model, image_path):
    """
    SVM 예측 결과를 시각화하는 함수
    Args:
        model: 학습된 AlexNet 기반 SVM 모델
        image_path: 입력 이미지 경로
    """
    # 모델을 평가 모드로 설정
    model.eval()

    # 이미지 전처리
    transform = transforms.Compose([
        transforms.Resize((227, 227)),  # AlexNet 입력 크기
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 원본 코드의 정규화 값 사용
    ])

    # 원본 이미지 로드 및 변환
    original_image = Image.open(image_path).convert('RGB')
    input_tensor = transform(original_image).unsqueeze(0)

    # GPU 사용 가능시 GPU로 이동
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to(device)

    # 특징맵 저장을 위한 변수
    features = []

    def hook_feature(module, input, output):
        features.append(output)
        print("visualize하는 feature map 크기 : ", output.size())

    # AlexNet의 마지막 컨볼루션 레이어에 훅 등록
    hook = model.features[-1].register_forward_hook(hook_feature)

    # 예측 수행
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.nn.functional.softmax(output, dim=1)
        confidence = prob.max().item()
        pred_class = output.argmax(1).item()

    # 특징맵 추출 및 시각화 준비
    """
    feature map의 붉은 부분 : 각 grid cell의 활성화 갚이 높다=해당 영역이 차량의 특징을 가지고 있다.
    !! 이 6x6 grid는 SVM 학습과는 직접적 관련 없음
    SVM은:
    1. AlexNet의 feature extractor가 이미지 처리
    2. Flatten된 features들 (6x6 channels)를 입력으로 받아
    3. '전체 이미지에 대한' 단일 예측 (차O/차X)을 수행함. 
    """
    feature_map = features[0].squeeze().mean(dim=0).cpu()
    feature_map = feature_map.numpy()
    feature_map = np.maximum(feature_map, 0)  # ReLU
    feature_map = feature_map / feature_map.max()  # 정규화

    # 시각화
    plt.figure(figsize=(15, 5))

    # 원본 이미지
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')

    # 특징맵
    plt.subplot(1, 3, 2)
    plt.imshow(feature_map, cmap='jet')
    plt.title(f'Feature Map (Class: {pred_class})')
    plt.axis('off')

    # 특징맵 오버레이
    feature_map_resized = cv2.resize(feature_map,
                                     (original_image.size[0], original_image.size[1]))
    feature_map_color = cv2.applyColorMap(np.uint8(255 * feature_map_resized),
                                          cv2.COLORMAP_JET)
    feature_map_color = cv2.cvtColor(feature_map_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(np.array(original_image), 0.6,
                              feature_map_color, 0.4, 0)

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title(f'Overlay (Confidence: {confidence:.2%})')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # 훅 제거
    hook.remove()

    return pred_class, confidence


def inspect_batch_predictions(model, data_loader, device, num_batches=1):
    """
    지정된 수의 배치에 대해 각 이미지의 SVM 예측 결과를 검사하고 출력합니다.

    Args:
        model: 학습된 SVM 모델
        data_loader: 데이터 로더
        device: 계산 장치 (CPU/GPU)
        num_batches: 검사할 배치 수 (기본값: 1)
    """
    model.eval()  # 모델을 평가 모드로 설정

    with torch.no_grad():  # 그래디언트 계산 비활성화
        for batch_idx, (inputs, labels, cache_dicts) in enumerate(data_loader):
            if batch_idx >= num_batches:
                break

            inputs = inputs.to(device)
            labels = labels.to(device)

            # 모델 예측
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

            # 각 이미지에 대한 예측 결과 출력
            print(f"\nBatch {batch_idx + 1} Results:")
            print("-" * 50)

            for idx in range(len(inputs)):
                image_id = cache_dicts['image_id'][idx]
                true_label = labels[idx].item()
                predicted_prob = probabilities[idx][1].item()  # class 1(차량)에 대한 확률

                # 예측 결과가 0.5를 넘으면 차량으로 판단 (1), 아니면 비차량으로 판단 (0)
                predicted_label = 1 if predicted_prob > 0.5 else 0

                # 예측이 맞았는지 표시
                correct = "✓" if predicted_label == true_label else "✗"

                print(
                    f"Image: {image_id:20} | True: {true_label} | Pred: {predicted_label} {correct} | Conf: {predicted_prob:6.2%}")

            print("-" * 50)
            print(f"Batch Size: {len(inputs)}")
            print(f"Positive Samples: {torch.sum(labels == 1).item()}")
            print(f"Negative Samples: {torch.sum(labels == 0).item()}")





# 모델 학습 함수 정의
def train_model(data_loaders, model, criterion, optimizer, lr_scheduler, num_epochs=25, device=None):
    # 학습 시작 시간 기록
    since = time.time()

    # 최적의 모델 가중치 초기화 (현재 모델의 초기 가중치로 설정)
    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0  # 최고 정확도 기록용

    # 전체 에폭 반복
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 각 에폭은 'train'(학습 단계)과 'val'(검증 단계)로 구성
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 모델을 학습 모드로 설정
            else:
                model.eval()  # 모델을 검증 모드로 설정

            # 손실 및 정확도 통계 초기화
            running_loss = 0.0
            running_corrects = 0

            # 현재 데이터셋의 positive 및 negative 샘플 수 출력
            data_set = data_loaders[phase].dataset
            print('{} - positive_num: {} - negative_num: {} - data size: {}'.format(
                phase, data_set.get_positive_num(), data_set.get_negative_num(), data_sizes[phase]))

            # 데이터 로더를 반복하여 배치 단위로 처리
            for inputs, labels, cache_dicts in data_loaders[phase]:
                inputs = inputs.to(device)  # 데이터를 GPU/CPU 장치로 이동
                labels = labels.to(device)  # 레이블을 GPU/CPU 장치로 이동

                # 옵티마이저의 그래디언트 초기화
                optimizer.zero_grad()

                # 순전파(forward pass): 학습 단계에서만 이력 기록
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)  # 모델 출력 계산
                    _, preds = torch.max(outputs, 1)  # 클래스 예측 (확률이 가장 높은 클래스 선택)
                    loss = criterion(outputs, labels)  # 손실 계산

                    # 역전파 및 옵티마이저 스텝은 학습 단계에서만 실행
                    if phase == 'train':
                        loss.backward()  # 역전파 계산
                        optimizer.step()  # 옵티마이저 스텝 실행

                # 배치 손실 및 정확도 누적 계산
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # 학습 단계에서는 학습률 스케줄러를 업데이트
            if phase == 'train':
                lr_scheduler.step()

            # 현재 단계의 에폭 손실 및 정확도 계산
            epoch_loss = running_loss / data_sizes[phase]
            epoch_acc = running_corrects.double() / data_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            print("\nInspecting training batch:")
            inspect_batch_predictions(model, data_loaders['train'], device)

            if phase == 'val':
                print("\nInspecting validation batch:")
                inspect_batch_predictions(model, data_loaders['val'], device)

            # 검증 단계에서 최고의 정확도를 달성한 경우, 모델 가중치를 갱신
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())

        # 에폭이 끝난 후 Hard Negative Mining 수행
        train_dataset = data_loaders['train'].dataset  # 학습 데이터셋 가져오기
        remain_negative_list = data_loaders['remain']  # 남아 있는 negative 샘플 목록
        jpeg_images = train_dataset.get_jpeg_images()  # 이미지 데이터 가져오기
        transform = train_dataset.get_transform()  # 데이터 변환(transform) 함수 가져오기

        # Hard Negative Mining용 데이터셋 준비
        with torch.set_grad_enabled(False):  # 그래디언트 비활성화 (추론 모드)
            remain_dataset = CustomHardNegativeMiningDataset(
                remain_negative_list, jpeg_images, transform=transform
            )
            remain_data_loader = DataLoader(
                remain_dataset, batch_size=batch_total, num_workers=8, drop_last=True
            )

            # 학습 데이터셋의 negative 샘플 리스트 가져오기
            negative_list = train_dataset.get_negatives()
            add_negative_list = data_loaders.get('add_negative', [])  # 추가된 negative 샘플 목록

            running_corrects = 0  # 정확도 초기화

            # Hard Negative Mining 데이터셋 반복 처리
            for inputs, labels, cache_dicts in remain_data_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 모델 출력 및 예측
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                # 정확도 계산
                running_corrects += torch.sum(preds == labels.data)

                # Hard Negative 샘플 및 Easy Negative 샘플 분류
                hard_negative_list, easy_negative_list = get_hard_negatives(preds.cpu().numpy(), cache_dicts)
                add_hard_negatives(hard_negative_list, negative_list, add_negative_list)  # Negative 샘플 추가

            remain_acc = running_corrects.double() / len(remain_negative_list)  # 남은 negative 샘플 정확도
            print('remain negative size: {}, acc: {:.4f}'.format(len(remain_negative_list), remain_acc))

            # 학습 데이터셋의 negative 샘플 업데이트
            train_dataset.set_negative_list(negative_list)
            tmp_sampler = CustomBatchSampler(
                train_dataset.get_positive_num(), train_dataset.get_negative_num(),
                batch_positive, batch_negative
            )
            data_loaders['train'] = DataLoader(
                train_dataset, batch_size=batch_total, sampler=tmp_sampler,
                num_workers=8, drop_last=True
            )
            data_loaders['add_negative'] = add_negative_list  # 추가된 negative 샘플 목록 갱신
            data_sizes['train'] = len(tmp_sampler)  # 학습 데이터 크기 업데이트

        # 특정 이미지를 불러와 SVM 예측 시각화 (옵션)
        image_path = "/home/ivpl-d29/dataset/VOC/voc_car/train/JPEGImages/000153.jpg"
        visualize_svm_predictions(model, image_path)

        # 각 에폭 종료 후 모델 저장 (옵션)
        # save_model(model, './logs/model/linear_svm_alexnet_car_%d.pth' % epoch)

        # 마지막 에폭에서만 최종 모델 저장
        if epoch == num_epochs - 1:
            save_epoch = epoch + 1
            save_model(model, './logs/model/linear_svm_alexnet_car_final_%d.pth' % save_epoch)

    # 전체 학습 시간 출력
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # 최고 정확도 출력
    print('Best val Acc: {:4f}'.format(best_acc))

    # 최적의 모델 가중치를 로드하여 반환
    model.load_state_dict(best_model_weights)
    return model



if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'

    data_loaders, data_sizes = load_data('/home/ivpl-d29/dataset/VOC/voc_car/classifier_car')

    # AlexNet 모델 로드
    model_path = './logs/model/finetuning_alexnet_car.pth'
    model = alexnet()
    num_classes = 2
    num_features = model.classifier[6].in_features
    # 모델의 마지막 분류기를 2개 클래스 (차O/차X)로 교체함.
    model.classifier[6] = nn.Linear(num_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # AlexNet의 모든 layer를 freeze(고정)==feature extract 부분 고정
    for param in model.parameters():
        param.requires_grad = False
    # SVM classifier 생성, AlexNet의 마지막 lyer를 새로운 Linear SVM으로 교체
    model.classifier[6] = nn.Linear(num_features, num_classes)
    # print(model)
    model = model.to(device)

    criterion = hinge_loss
    # 초기 학습 데이터셋 크기가 작으므로 학습률을 낮춤
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    # 총 10epoch 학습, 4 epoch마다 학습률 감소
    lr_schduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    # SVM 모델 학습, 가장 좋은 정확도의 모델을 best_model로 저장
    best_model = train_model(data_loaders, model, criterion, optimizer, lr_schduler, num_epochs=10, device=device)
    # 최적의 모델 저장함
    save_model(best_model, './logs/model/linear_svm_alexnet_car_best.pth')