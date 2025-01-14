# -*- coding: utf-8 -*-

"""
AlexNet fine-tuning 데이터셋 만들기
"""

import time
import sys
import shutil
import numpy as np
import cv2
import os
sys.path.append("/home/ivpl-d29/myProject/Study_Model/R-CNN")
import SelectiveSearch
from utils.util import check_dir
from utils.util import parse_car_csv
from utils.util import parse_xml
from utils.util import compute_ious


# train
# positive num: 66517
# negatie num: 464340
# val
# positive num: 64712
# negative num: 415134
# 논문에서.. 총 batch size: 128
# positive: 32 / negative: 96

"""
이 코드는 AlexNet을 미세 조정하기 위한 데이터셋을 생성하는 파이썬 스크립트입니다. 주요 기능은 다음과 같습니다:

Selective Search를 사용하여 이미지에서 후보 영역(Region Proposal)을 추출합니다.
추출된 후보 영역과 실제 객체 영역(Bounding Box) 간의 IoU(Intersection over Union)를 계산하여 Positive 및 Negative 샘플을 구분합니다.
Positive 샘플은 IoU가 0.5 이상인 영역, Negative 샘플은 IoU가 0.5 미만이며 실제 객체 영역의 1/5보다 큰 영역으로 정의합니다.
생성된 Positive 및 Negative 샘플을 파일로 저장하고, 해당 이미지도 복사합니다.
주요 함수 설명:
parse_annotation_jpeg(annotation_path, jpeg_path, gs):

이 함수는 주어진 이미지와 해당 어노테이션 파일을 사용하여 Positive 및 Negative 샘플을 추출합니다.
SelectiveSearch.get_rects(gs)를 통해 후보 영역을 추출하고, compute_ious(rects, bndboxs)를 통해 IoU를 계산합니다.
IoU를 기준으로 Positive 및 Negative 샘플을 분류하고, 각각의 리스트를 반환합니다.
main 함수:

train 및 val 데이터셋에 대해 각각 Positive 및 Negative 샘플을 생성하고, 이를 파일로 저장합니다.
각 이미지에 대해 parse_annotation_jpeg 함수를 호출하여 샘플을 추출하고, 이를 저장합니다.
코드 실행 결과:
train 및 val 데이터셋에 대해 Positive 및 Negative 샘플이 생성되고, 이미지와 함께 저장됩니다.
각 샘플의 수를 출력하여 데이터셋의 크기를 확인할 수 있습니다.
이 코드는 AlexNet을 미세 조정하기 위한 데이터셋을 생성하는 데 사용되며, 특히 객체 검출 작업에서 중요한 역할을 합니다.
"""


def parse_annotation_jpeg(annotation_path, jpeg_path, gs):
    """
    negative/positive sample을 얻음 (cf. difficult 속성이 True인 레이블이 지정된  bounding box를 무시)
    positive sample: IoU 0.5 이상
    negative sample: IoU 0.5 미만, negative sample 수를 더욱 제한하려면 크기가 annotated bounding box의 1/5보다 커야 함.
    """
    img = cv2.imread(jpeg_path)

    SelectiveSearch.config(gs, img, strategy='f')
    # Regional Proposal 계산
    rects = SelectiveSearch.get_rects(gs)
    # Label annotated bounding box를 가져옴
    bndboxs = parse_xml(annotation_path)
    print("rects: ", rects)
    print("bndboxes: ")
    for bndbox in bndboxs: print(bndbox)

    # Label box 크기
    maximum_bndbox_size = 0
    for bndbox in bndboxs:
        xmin, ymin, xmax, ymax = bndbox
        bndbox_size = (ymax - ymin) * (xmax - xmin)
        if bndbox_size > maximum_bndbox_size:
            maximum_bndbox_size = bndbox_size

    # Regional Proposal과 annotated bounding box의 IoU를 iou_list로 가져옴
    iou_list = compute_ious(rects, bndboxs)

    positive_list = list()
    negative_list = list()
    for i in range(len(iou_list)):
        xmin, ymin, xmax, ymax = rects[i]
        rect_size = (ymax - ymin) * (xmax - xmin)

        iou_score = iou_list[i]
        if iou_list[i] >= 0.5:
            # Positive Sample
            positive_list.append(rects[i])
        if 0 < iou_list[i] < 0.5 and rect_size > maximum_bndbox_size / 5.0:
            # Negative Sample
            negative_list.append(rects[i])
        else:
            pass

    return positive_list, negative_list


if __name__ == '__main__':
    car_root_dir = '/home/ivpl-d29/dataset/VOC/voc_car/'
    finetune_root_dir = '/home/ivpl-d29/dataset/VOC/voc_car/finetune_car/'
    check_dir(finetune_root_dir)

    gs = SelectiveSearch.get_selective_search()
    for name in ['train', 'val']:
        src_root_dir = os.path.join(car_root_dir, name)
        src_annotation_dir = os.path.join(src_root_dir, 'Annotations')
        src_jpeg_dir = os.path.join(src_root_dir, 'JPEGImages')

        dst_root_dir = os.path.join(finetune_root_dir, name)
        dst_annotation_dir = os.path.join(dst_root_dir, 'Annotations')
        dst_jpeg_dir = os.path.join(dst_root_dir, 'JPEGImages')
        check_dir(dst_root_dir)
        check_dir(dst_annotation_dir)
        check_dir(dst_jpeg_dir)

        total_num_positive = 0
        total_num_negative = 0

        samples = parse_car_csv(src_root_dir)
        # CSV 파일 복사
        src_csv_path = os.path.join(src_root_dir, 'car.csv')
        dst_csv_path = os.path.join(dst_root_dir, 'car.csv')
        shutil.copyfile(src_csv_path, dst_csv_path)
        for sample_name in samples:
            since = time.time()

            src_annotation_path = os.path.join(src_annotation_dir, sample_name + '.xml')
            src_jpeg_path = os.path.join(src_jpeg_dir, sample_name + '.jpg')
            # Positive 및 Negative sample을 얻음
            positive_list, negative_list = parse_annotation_jpeg(src_annotation_path, src_jpeg_path, gs)
            total_num_positive += len(positive_list)
            total_num_negative += len(negative_list)

            dst_annotation_positive_path = os.path.join(dst_annotation_dir, sample_name + '_1' + '.csv')
            dst_annotation_negative_path = os.path.join(dst_annotation_dir, sample_name + '_0' + '.csv')
            dst_jpeg_path = os.path.join(dst_jpeg_dir, sample_name + '.jpg')
            # 이미지 저장
            shutil.copyfile(src_jpeg_path, dst_jpeg_path)
            # Positive, Negative sample의 annotation을 저장함
            np.savetxt(dst_annotation_positive_path, np.array(positive_list), fmt='%d', delimiter=' ')
            np.savetxt(dst_annotation_negative_path, np.array(negative_list), fmt='%d', delimiter=' ')

            time_elapsed = time.time() - since
            print('parse {}.png in {:.0f}m {:.0f}s'.format(sample_name, time_elapsed // 60, time_elapsed % 60))
        print('%s positive num: %d' % (name, total_num_positive))
        print('%s negative num: %d' % (name, total_num_negative))
    print('done')