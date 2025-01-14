# -*- coding: utf-8 -*-

"""
@date: 2020/3/1 下午7:17
@file: create_classifier_data.py
@author: zj
@description: 분류기 데이터셋 생성 스크립트
"""

import random
import numpy as np
import shutil
import time
import cv2
import os
import xmltodict
import sys
sys.path.append("/home/ivpl-d29/myProject/Study_Model/R-CNN")
import SelectiveSearch as selectivesearch
from utils.util import check_dir
from utils.util import parse_car_csv
from utils.util import parse_xml
from utils.util import iou
from utils.util import compute_ious

# train
# positive num: 625
# negative num: 366028
# val
# positive num: 625
# negative num: 321474

def parse_annotation_jpeg(annotation_path, jpeg_path, gs):
    """
    주어진 이미지와 주석 파일로부터 정/오 샘플을 추출.
    - 정답 샘플: 주석으로 표시된 바운딩 박스(Ground Truth, GT)
    - 오답 샘플: IoU 값이 (0, 0.3] 사이인 후보 바운딩 박스
               추가로, 후보 박스의 크기가 GT 크기의 1/5 이상이어야 함
    - 주석에 difficult 속성이 True인 GT 박스는 무시

    :param annotation_path: 주석 파일 경로(XML)
    :param jpeg_path: 이미지 파일 경로(JPEG)
    :param gs: Selective Search 설정 객체
    :return: 정 샘플 리스트(바운딩 박스 좌표), 오 샘플 리스트
    """
    img = cv2.imread(jpeg_path) # 이미지 읽어옴

    # Selective Search 설정 및 후보 영역 계산
    selectivesearch.config(gs, img, strategy='q')  # 'q' 전략 사용
    rects = selectivesearch.get_rects(gs)  # 후보 바운딩 박스 리스트 생성

    # XML에서 Ground Truth 바운딩 박스 정보 추출
    bndboxs = parse_xml(annotation_path)

    # 가장 큰 GT 박스 크기 계산
    maximum_bndbox_size = 0
    for bndbox in bndboxs:
        xmin, ymin, xmax, ymax = bndbox
        bndbox_size = (ymax - ymin) * (xmax - xmin)
        if bndbox_size > maximum_bndbox_size:
            maximum_bndbox_size = bndbox_size

    # 후보 박스와 GT 박스 간 IoU 계산
    iou_list = compute_ious(rects, bndboxs)

    positive_list = []  # 정 샘플 리스트
    negative_list = []  # 오 샘플 리스트

    # 후보 박스별 IoU 점수 및 크기 조건 확인
    for i in range(len(iou_list)):
        xmin, ymin, xmax, ymax = rects[i]
        rect_size = (ymax - ymin) * (xmax - xmin)

        iou_score = iou_list[i]  # 후보 박스와 GT 박스 간 IoU
        if 0 < iou_score <= 0.3 and rect_size > maximum_bndbox_size / 5.0:
            # IoU가 0보다 크고 0.3 이하이며 크기 조건을 만족하는 경우
            negative_list.append(rects[i])  # 오 샘플로 추가
        else:
            pass  # 조건에 맞지 않는 경우 무시

    return bndboxs, negative_list  # 정 샘플(GT), 오 샘플 반환


if __name__ == '__main__':
    car_root_dir = '/home/ivpl-d29/dataset/VOC/voc_car/'
    classifier_root_dir = '/home/ivpl-d29/dataset/VOC/voc_car/classifier_car/'
    check_dir(classifier_root_dir)

    gs = selectivesearch.get_selective_search()
    for name in ['train', 'val']:
        # 원본 데이터 디렉토리 설정
        src_root_dir = os.path.join(car_root_dir, name)
        src_annotation_dir = os.path.join(src_root_dir, 'Annotations')
        src_jpeg_dir = os.path.join(src_root_dir, 'JPEGImages')

        # 결과 저장 디렉토리 설정
        dst_root_dir = os.path.join(classifier_root_dir, name)
        dst_annotation_dir = os.path.join(dst_root_dir, 'Annotations')
        dst_jpeg_dir = os.path.join(dst_root_dir, 'JPEGImages')
        check_dir(dst_root_dir)
        check_dir(dst_annotation_dir)
        check_dir(dst_jpeg_dir)

        # 전체 정답 / 오답 샘플 수
        total_num_positive = 0
        total_num_negative = 0

        # 데이터셋 샘플 파일 이름 리스트 읽기
        samples = parse_car_csv(src_root_dir)
        # csv 파일 복사
        src_csv_path = os.path.join(src_root_dir, 'car.csv')
        dst_csv_path = os.path.join(dst_root_dir, 'car.csv')
        shutil.copyfile(src_csv_path, dst_csv_path)

        # 각 샘플 처리
        for sample_name in samples:
            since = time.time()     # 처리 시작 시간 기록

            # annotation 파일 및 이미지 파일 경로 설정
            src_annotation_path = os.path.join(src_annotation_dir, sample_name + '.xml')
            src_jpeg_path = os.path.join(src_jpeg_dir, sample_name + '.jpg')
            # 정답, 오답 샘플 추출
            positive_list, negative_list = parse_annotation_jpeg(src_annotation_path, src_jpeg_path, gs)
            total_num_positive += len(positive_list)
            total_num_negative += len(negative_list)
            # 정답, 오답 샘플 결과 저장 경로 설정
            dst_annotation_positive_path = os.path.join(dst_annotation_dir, sample_name + '_1' + '.csv')
            dst_annotation_negative_path = os.path.join(dst_annotation_dir, sample_name + '_0' + '.csv')
            dst_jpeg_path = os.path.join(dst_jpeg_dir, sample_name + '.jpg')
            # 이미지 저장
            shutil.copyfile(src_jpeg_path, dst_jpeg_path)
            # 정답, 오답 샘플 저장
            np.savetxt(dst_annotation_positive_path, np.array(positive_list), fmt='%d', delimiter=' ')
            np.savetxt(dst_annotation_negative_path, np.array(negative_list), fmt='%d', delimiter=' ')
            # 처리 시간 출력
            time_elapsed = time.time() - since
            print('parse {}.png in {:.0f}m {:.0f}s'.format(sample_name, time_elapsed // 60, time_elapsed % 60))
        # 처리 결과 출력
        print('%s positive num: %d' % (name, total_num_positive))
        print('%s negative num: %d' % (name, total_num_negative))
    print('done')
