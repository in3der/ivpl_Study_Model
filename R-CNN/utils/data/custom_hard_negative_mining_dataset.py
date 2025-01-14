# -*- coding: utf-8 -*-

"""
@date: 2020/3/18 오후 3:37
@file: custom_hard_negative_mining_dataset.py
@author: zj
@description: 하드 네거티브 마이닝을 위한 커스텀 데이터셋 클래스 정의
"""

import torch.nn as nn
from torch.utils.data import Dataset
import sys
sys.path.append("/home/ivpl-d29/myProject/Study_Model/R-CNN")
from utils.data.custom_classifier_dataset import CustomClassifierDataset


class CustomHardNegativeMiningDataset(Dataset):
    """
    하드 네거티브 마이닝 데이터셋 클래스.
    잘못 분류된 네거티브 샘플을 처리하고 네트워크 학습에 사용.
    """

    def __init__(self, negative_list, jpeg_images, transform=None):
        """
        초기화 메서드.

        :param negative_list: 네거티브 샘플 정보 리스트
        :param jpeg_images: 원본 이미지 데이터
        :param transform: 데이터 변환(transform) 함수
        """
        self.negative_list = negative_list  # 네거티브 샘플 리스트 저장
        self.jpeg_images = jpeg_images  # 원본 이미지 저장
        self.transform = transform  # transform 저장

    def __getitem__(self, index: int):
        """
        데이터셋에서 index에 해당하는 샘플 반환.

        :param index: 가져올 샘플의 인덱스
        :return: 이미지, 타겟(레이블), 네거티브 샘플 정보 딕셔너리
        """
        target = 0  # 네거티브 샘플이므로 타겟 값은 0

        # 네거티브 샘플 정보에서 좌표와 이미지 ID를 가져옴
        negative_dict = self.negative_list[index]
        xmin, ymin, xmax, ymax = negative_dict['rect']  # 바운딩 박스 좌표
        image_id = negative_dict['image_id']  # 이미지 ID

        # 바운딩 박스에 해당하는 이미지 영역 추출
        image = self.jpeg_images[image_id][ymin:ymax, xmin:xmax]

        # transform이 정의된 경우 이미지에 transform 적용
        if self.transform:
            image = self.transform(image)

        return image, target, negative_dict  # 이미지, 타겟(0), 추가 정보 반환

    def __len__(self) -> int:
        """
        데이터셋의 전체 길이 반환.
        """
        return len(self.negative_list)


if __name__ == '__main__':
    # 디버깅용 코드: 데이터셋 로드 및 테스트
    root_dir = '/home/ivpl-d29/dataset/VOC/voc_car/classifier_car/train'
    data_set = CustomClassifierDataset(root_dir)  # 기본 데이터셋 클래스 로드

    # 네거티브 샘플 리스트와 원본 이미지, 변환 정보 가져오기
    negative_list = data_set.get_negatives()
    jpeg_images = data_set.get_jpeg_images()
    transform = data_set.get_transform()

    # 하드 네거티브 데이터셋 생성
    hard_negative_dataset = CustomHardNegativeMiningDataset(negative_list, jpeg_images, transform=transform)

    # 디버깅용 샘플 추출 및 정보 출력
    sample_index = 100  # 임의로 100번 샘플을 선택
    image, target, negative_dict = hard_negative_dataset.__getitem__(sample_index)

    # 디버깅용 출력
    print(f"[DEBUG] 샘플 인덱스: {sample_index}")
    print(f"[DEBUG] 추출된 이미지 크기: {image.shape}")  # 이미지 크기 출력
    print(f"[DEBUG] 타겟(레이블): {target}")  # 타겟 값(0)
    print(f"[DEBUG] 네거티브 샘플 정보: {negative_dict}")  # 네거티브 샘플 정보
