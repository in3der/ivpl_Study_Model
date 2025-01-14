"""
PASCAL VOC 2007 dataset에서 Car 카테고리를 추출
"""

import os
import sys
import shutil
import random
import numpy as np
import xmltodict
sys.path.append("/home/ivpl-d29/myProject/Study_Model/R-CNN")
from utils.util import check_dir

suffix_xml = '.xml'
suffix_jpeg = '.jpg'


def extract_car_images(annotation_dir):
    """
    "car" 클래스를 포함하는 이미지 파일명을 추출합니다.
    """
    car_samples = []
    for xml_file in os.listdir(annotation_dir):
        if xml_file.endswith('.xml'):
            with open(os.path.join(annotation_dir, xml_file), 'rb') as f:
                xml_dict = xmltodict.parse(f)
                objects = xml_dict['annotation'].get('object', [])

                # 객체가 리스트인지 단일 객체인지 확인
                if isinstance(objects, dict):
                    objects = [objects]

                # "car" 클래스 있는지 확인
                for obj in objects:
                    if obj['name'] == 'car' and int(obj['difficult']) == 0:  # 난이도 낮은 car만 사용
                        car_samples.append(xml_file.replace('.xml', ''))
                        break

    return np.array(car_samples)


def split_train_val(samples, train_ratio=0.8):
    """
    추출된 car 이미지를 train과 val로 나눕니다.
    """
    np.random.shuffle(samples)
    train_size = int(len(samples) * train_ratio)
    train_samples = samples[:train_size]
    val_samples = samples[train_size:]
    return train_samples, val_samples


def save_samples_to_txt(train_samples, val_samples, output_dir):
    """
    train과 val 샘플을 텍스트 파일로 저장합니다.
    """
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, 'car_train.txt')
    val_path = os.path.join(output_dir, 'car_val.txt')

    np.savetxt(train_path, train_samples, fmt='%s')
    np.savetxt(val_path, val_samples, fmt='%s')
    print(f"Saved {len(train_samples)} train samples to {train_path}")
    print(f"Saved {len(val_samples)} val samples to {val_path}")


def parse_train_val(data_path):
    """
    저장한 카테고리의 이미지 파일명을 추출합니다.
    """
    samples = []

    with open(data_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            sample_name = line.strip()  # 각 줄의 이미지 파일명만 추출
            if sample_name:  # 빈 줄이 아닐 경우에만 추가
                samples.append(sample_name)

    return np.array(samples)


def sample_train_val(samples):
    """
    데이터 셋 수를 줄이기 위해 무작위로 샘플링 + 1/10만 사용
    데이터 셋 수를 줄이기 위해 무작위로 샘플링  - 나는 전부 사용
    """
    for name in ['train', 'val']:
        dataset = samples[name]
        length = len(dataset)

        # random_samples = random.sample(range(length), int(length / 10))
        random_samples = random.sample(range(length), int(length))
        # print(random_samples)
        new_dataset = dataset[random_samples]
        samples[name] = new_dataset

    return samples


# def parse_car(sample_list):
#     """
#     자동차가 포함된 모든 annotated file과 filter sample을 탐색
#     """
#
#     car_samples = list()
#     for sample_name in sample_list:
#         annotation_path = os.path.join(voc_annotation_dir, sample_name + suffix_xml)
#         with open(annotation_path, 'rb') as f:
#             xml_dict = xmltodict.parse(f)
#             # print(xml_dict)
#
#             bndboxs = list()
#             objects = xml_dict['annotation']['object']
#             if isinstance(objects, list):
#                 for obj in objects:
#                     obj_name = obj['name']
#                     difficult = int(obj['difficult'])
#                     if 'car'.__eq__(obj_name) and difficult != 1:
#                         car_samples.append(sample_name)
#             elif isinstance(objects, dict):
#                 obj_name = objects['name']
#                 difficult = int(objects['difficult'])
#                 if 'car'.__eq__(obj_name) and difficult != 1:
#                     car_samples.append(sample_name)
#             else:
#                 pass
#
#     return car_samples


def save_car(car_samples, data_root_dir, data_annotation_dir, data_jpeg_dir):
    """
    자동차 카테고리의 샘플 사진과 주석 파일을 저장
    """
    for sample_name in car_samples:
        src_annotation_path = os.path.join(voc_annotation_dir, sample_name + suffix_xml)
        dst_annotation_path = os.path.join(data_annotation_dir, sample_name + suffix_xml)
        shutil.copyfile(src_annotation_path, dst_annotation_path)

        src_jpeg_path = os.path.join(voc_jpeg_dir, sample_name + suffix_jpeg)
        dst_jpeg_path = os.path.join(data_jpeg_dir, sample_name + suffix_jpeg)
        shutil.copyfile(src_jpeg_path, dst_jpeg_path)

    csv_path = os.path.join(data_root_dir, 'car.csv')
    np.savetxt(csv_path, np.array(car_samples), fmt='%s')


car_train_path = '/home/ivpl-d29/dataset/VOC/voc_car/car_train.txt'
car_val_path = '/home/ivpl-d29/dataset/VOC/voc_car/car_val.txt'

voc_annotation_dir = '/home/ivpl-d29/dataset/VOC/VOC2007/Annotations/'
voc_jpeg_dir = '/home/ivpl-d29/dataset/VOC/VOC2007/JPEGImages/'

car_root_dir = '/home/ivpl-d29/dataset/VOC/voc_car/'


if __name__ == '__main__':
    # "car" 클래스 이미지 파일명 추출
    car_samples = extract_car_images(voc_annotation_dir)
    print(f"Found {len(car_samples)} car samples.")

    # train/val split
    train_samples, val_samples = split_train_val(car_samples)
    print(f"Split into {len(train_samples)} train and {len(val_samples)} val samples.")

    # 텍스트 파일로 저장
    save_samples_to_txt(train_samples, val_samples, car_root_dir)


    samples = {'train': parse_train_val(car_train_path), 'val': parse_train_val(car_val_path)}
    print(samples)
    # samples = sample_train_val(samples)
    # print(samples)

    check_dir(car_root_dir)
    for name in ['train', 'val']:
        data_root_dir = os.path.join(car_root_dir, name)
        data_annotation_dir = os.path.join(data_root_dir, 'Annotations')
        data_jpeg_dir = os.path.join(data_root_dir, 'JPEGImages')

        check_dir(data_root_dir)
        check_dir(data_annotation_dir)
        check_dir(data_jpeg_dir)
        save_car(samples[name], data_root_dir, data_annotation_dir, data_jpeg_dir)

    print('done')