import numpy  as np
import os
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import sys
sys.path.append("/home/ivpl-d29/myProject/Study_Model/R-CNN")
from utils.util import parse_car_csv

"""
각 이미지에 대해 positive, negative annotation을 처리하여 딥러닝 모델 학습에 사용할 수 있도록 준비하는 클래스
특히, Hard Negative Mining에 사용할 수 있도록 negative sample을 추가적으로 교체하거나 수정할 수 있음
"""
class CustomClassifierDataset(Dataset):
    # 초기화 함수
    def __init__(self, root_dir, transform=None):
        samples = parse_car_csv(root_dir)

        jpeg_images = list()
        positive_list = list()
        negative_list = list()
        for idx in range(len(samples)):
            sample_name = samples[idx]
            jpeg_images.append(cv2.imread(os.path.join(root_dir, 'JPEGImages', sample_name + ".jpg")))

            positive_annotation_path = os.path.join(root_dir, 'Annotations', sample_name + '_1.csv')
            try:
                positive_annotations = np.loadtxt(positive_annotation_path, dtype=int, delimiter=' ')
                if len(positive_annotations.shape) == 1:
                    if positive_annotations.shape[0] == 4:
                        positive_dict = {"rect": positive_annotations, "image_id": idx}
                        positive_list.append(positive_dict)
                else:
                    for positive_annotation in positive_annotations:
                        positive_dict = {"rect": positive_annotation, "image_id": idx}
                        positive_list.append(positive_dict)
            except FileNotFoundError:
                print(f"Warning: {positive_annotation_path} not found. Skipping.")

            negative_annotation_path = os.path.join(root_dir, 'Annotations', sample_name + '_0.csv')
            try:
                negative_annotations = np.loadtxt(negative_annotation_path, dtype=int, delimiter=' ')
                if len(negative_annotations.shape) == 1:
                    if negative_annotations.shape[0] == 4:
                        negative_dict = {"rect": negative_annotations, "image_id": idx}
                        negative_list.append(negative_dict)
                else:
                    for negative_annotation in negative_annotations:
                        negative_dict = {"rect": negative_annotation, "image_id": idx}
                        negative_list.append(negative_dict)
            except FileNotFoundError:
                print(f"Warning: {negative_annotation_path} not found. Skipping.")

        self.transform = transform
        self.jpeg_images = jpeg_images
        self.positive_list = positive_list
        self.negative_list = negative_list

    def __getitem__(self, index: int):
        # 定位下标所属图像
        if index < len(self.positive_list):
            # 正样本
            target = 1
            positive_dict = self.positive_list[index]

            xmin, ymin, xmax, ymax = positive_dict['rect']
            image_id = positive_dict['image_id']

            image = self.jpeg_images[image_id][ymin:ymax, xmin:xmax]
            cache_dict = positive_dict
        else:
            # 负样本
            target = 0
            idx = index - len(self.positive_list)
            negative_dict = self.negative_list[idx]

            xmin, ymin, xmax, ymax = negative_dict['rect']
            image_id = negative_dict['image_id']

            image = self.jpeg_images[image_id][ymin:ymax, xmin:xmax]
            cache_dict = negative_dict

        # print('index: %d image_id: %d target: %d image.shape: %s [xmin, ymin, xmax, ymax]: [%d, %d, %d, %d]' %
        #       (index, image_id, target, str(image.shape), xmin, ymin, xmax, ymax))
        if self.transform:
            image = self.transform(image)

        return image, target, cache_dict

    def __len__(self) -> int:
        return len(self.positive_list) + len(self.negative_list)

    def get_transform(self):
        return self.transform

    def get_jpeg_images(self) -> list:
        return self.jpeg_images

    def get_positive_num(self) -> int:
        return len(self.positive_list)

    def get_negative_num(self) -> int:
        return len(self.negative_list)

    def get_positives(self) -> list:
        return self.positive_list

    def get_negatives(self) -> list:
        return self.negative_list

    # 用于hard negative mining
    # 替换负样本
    def set_negative_list(self, negative_list):
        self.negative_list = negative_list


def test(idx):
    root_dir = '/home/ivpl-d29/dataset/VOC/voc_car/classifier_car/val'
    train_data_set = CustomClassifierDataset(root_dir)

    print('positive num: %d' % train_data_set.get_positive_num())
    print('negative num: %d' % train_data_set.get_negative_num())
    print('total num: %d' % train_data_set.__len__())

    # 测试id=3/66516/66517/530856
    image, target, cache_dict = train_data_set.__getitem__(idx)
    print('target: %d' % target)
    print('dict: ' + str(cache_dict))

    image = Image.fromarray(image)
    print(image)
    print(type(image))

    # cv2.imshow('image', image)
    # cv2.waitKey(0)


def test2():
    root_dir = '/home/ivpl-d29/dataset/VOC/voc_car/classifier_car/train'
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data_set = CustomClassifierDataset(root_dir, transform=transform)
    image, target, cache_dict = train_data_set.__getitem__(230856)
    print('target: %d' % target)
    print('dict: ' + str(cache_dict))
    print('image.shape: ' + str(image.shape))


def test3():
    root_dir = '../../data/classifier_car/train'
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data_set = CustomClassifierDataset(root_dir, transform=transform)
    data_loader = DataLoader(train_data_set, batch_size=128, num_workers=8, drop_last=True)

    inputs, targets, cache_dicts = next(data_loader.__iter__())
    print(targets)
    print(inputs.shape)


if __name__ == '__main__':
    # test(159622)
    test(4051)
    #test(24768)
    # test2()
    # test3()