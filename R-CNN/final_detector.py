
from tqdm import tqdm
import time
import os
import copy
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import alexnet
import torchvision.transforms as transforms
import SelectiveSearch
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from utils.util import (
    parse_xml,
    iou,
)

"""
1. Extract only Car data
2. Prepare the dataset
3. Fine-tune AlexNet
        model(1) 파일 뽑아냄
4. Train SVM, extract the model
        model(1) 넣고 SVM 학습, model(2) 뽑음
5. Attach BBox regression
        model(2) 넣고 BBox regression, model(3) 뽑음
6. Final detection + NMS - model(3) 넣고 최종 detection 
"""


def get_bbox_regression_model(device=None):
    """bbox regression 모델 로드"""
    model = alexnet()
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, 4)
    model.load_state_dict(torch.load('/home/ivpl-d29/myProject/Study_Model/R-CNN/logs/model/bbox_regression_best.pth'), strict=False)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False
    if device:
        model = model.to(device)
    return model


def apply_bbox_regression(rect, regression_output):
    """bbox regression 결과를 적용하여 bbox 조정"""
    xmin, ymin, xmax, ymax = rect

    # 현재 bbox의 중심점과 width, height 계산
    w = xmax - xmin
    h = ymax - ymin
    cx = xmin + w / 2
    cy = ymin + h / 2

    # regression 출력값 추출
    dx, dy, dw, dh = regression_output

    # 새로운 중심점과 width, height 계산
    new_cx = cx + dx * w
    new_cy = cy + dy * h
    new_w = w * np.exp(dw)
    new_h = h * np.exp(dh)

    # 새로운 bbox 좌표 계산
    new_xmin = new_cx - new_w / 2
    new_ymin = new_cy - new_h / 2
    new_xmax = new_cx + new_w / 2
    new_ymax = new_cy + new_h / 2

    return [int(new_xmin), int(new_ymin), int(new_xmax), int(new_ymax)]


def get_transform():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform


def get_model(device=None):
    model = alexnet()
    num_classes = 2
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, num_classes)
    #model.load_state_dict(torch.load('./models/best_linear_svm_alexnet_car.pth'))
    model.load_state_dict(torch.load('/home/ivpl-d29/myProject/Study_Model/R-CNN/logs/model/linear_svm_alexnet_car_best.pth'), strict=False)
    model.eval()

    # gradient 추적 취소
    for param in model.parameters():
        param.requires_grad = False
    if device:
        model = model.to(device)

    return model


def draw_box_with_text(img, rect_list, score_list):
    """
    bounding box와 분류 확률 기재
    :param img:
    :param rect_list:
    :param score_list:
    :return:
    """
    for i in range(len(rect_list)):
        xmin, ymin, xmax, ymax = rect_list[i]
        score = score_list[i]

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(0, 0, 255), thickness=1)
        cv2.putText(img, "{:.3f}".format(score), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def nms(rect_list, score_list):
    """
    NMS
    :param rect_list: list，크기:[N, 4]
    :param score_list： list，크기:[N]
    """
    nms_rects = list()
    nms_scores = list()

    rect_array = np.array(rect_list)
    score_array = np.array(score_list)

    # 한 번 정렬한 후
    # 분류 확률을 기준으로 큰 것부터 작은 것까지 정렬
    idxs = np.argsort(score_array)[::-1]
    rect_array = rect_array[idxs]
    score_array = score_array[idxs]

    thresh = 0.3  # 0.3
    while len(score_array) > 0:
        # 분류 확률이 가장 높은 bounding box를 추가
        nms_rects.append(rect_array[0])
        nms_scores.append(score_array[0])
        rect_array = rect_array[1:]
        score_array = score_array[1:]

        length = len(score_array)
        if length <= 0:
            break

        # IoU 계산
        iou_scores = iou(np.array(nms_rects[len(nms_rects) - 1]), rect_array)
        # print(iou_scores)
        # IoU가 threshold보다 작으면 제거
        idxs = np.where(iou_scores < thresh)[0]
        rect_array = rect_array[idxs]
        score_array = score_array[idxs]

    return nms_rects, nms_scores


def visualize_detections(img, gt_boxes, detected_boxes, detected_scores):
    """
    Visualize ground truth and detected bounding boxes
    Args:
        img: Input image (RGB)
        gt_boxes: Ground truth bounding boxes
        detected_boxes: Detected bounding boxes from model
        detected_scores: Confidence scores for detected boxes
    """
    fig, ax = plt.subplots(1, figsize=(12, 8))

    # Display image
    ax.imshow(img)

    # Draw ground truth boxes in green
    for box in gt_boxes:
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin
        rect = patches.Rectangle((xmin, ymin), width, height,
                                 linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)

    # Draw detected boxes in red
    for box, score in zip(detected_boxes, detected_scores):
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin
        rect = patches.Rectangle((xmin, ymin), width, height,
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        # Add confidence score
        plt.text(xmin, ymin - 5, f'{score:.2f}', color='red')

    ax.set_title('Car Detection Results\nGreen: Ground Truth, Red: Detected')
    ax.axis('off')
    plt.tight_layout()
    return fig


def process_single_image(img_path, xml_path, model, bbox_model, transform, device):
    """단일 이미지에 대한 처리를 수행하는 함수 (bbox regression 추가)"""
    img = cv2.imread(img_path)
    bndboxs = parse_xml(xml_path)

    gs = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    SelectiveSearch.config(gs, img, strategy='f')
    rects = SelectiveSearch.get_rects(gs)

    score_list = []
    positive_list = []
    svm_thresh = 0.6

    for rect in rects:
        xmin, ymin, xmax, ymax = rect
        rect_img = img[ymin:ymax, xmin:xmax]

        # Classification 예측
        rect_transform = transform(rect_img).to(device)
        output = model(rect_transform.unsqueeze(0))[0]

        if torch.argmax(output).item() == 1:
            probs = torch.softmax(output, dim=0).cpu().numpy()
            if probs[1] >= svm_thresh:
                # Bbox regression 예측
                bbox_output = bbox_model(rect_transform.unsqueeze(0))[0]
                refined_rect = apply_bbox_regression(rect, bbox_output.cpu().detach().numpy())

                # 이미지 경계 확인
                h, w = img.shape[:2]
                refined_rect[0] = max(0, min(refined_rect[0], w))
                refined_rect[1] = max(0, min(refined_rect[1], h))
                refined_rect[2] = max(0, min(refined_rect[2], w))
                refined_rect[3] = max(0, min(refined_rect[3], h))

                score_list.append(probs[1])
                positive_list.append(refined_rect)

    nms_rects, nms_scores = nms(positive_list, score_list)
    return img, bndboxs, nms_rects, nms_scores


def visualize_multiple_detections(image_list, base_path):
    """여러 이미지의 detection 결과를 시각화하는 함수 (bbox regression 포함)"""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    transform = get_transform()
    model = get_model(device=device)
    bbox_model = get_bbox_regression_model(device=device)

    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    fig.suptitle('Car Detection Results with Bbox Regression', fontsize=16, y=1.02)
    axes_flat = axes.flatten()

    for idx, img_name in enumerate(image_list):
        img_path = f"{base_path}/JPEGImages/{img_name}.jpg"
        xml_path = f"{base_path}/Annotations/{img_name}.xml"

        img, gt_boxes, detected_boxes, detected_scores = process_single_image(
            img_path, xml_path, model, bbox_model, transform, device
        )

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax = axes_flat[idx]
        ax.imshow(img_rgb)

        # Ground truth boxes (green)
        for box in gt_boxes:
            xmin, ymin, xmax, ymax = box
            width = xmax - xmin
            height = ymax - ymin
            rect = patches.Rectangle((xmin, ymin), width, height,
                                     linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)

        # Detected boxes with regression (red)
        for box, score in zip(detected_boxes, detected_scores):
            xmin, ymin, xmax, ymax = box
            width = xmax - xmin
            height = ymax - ymin
            rect = patches.Rectangle((xmin, ymin), width, height,
                                     linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(xmin, ymin - 5, f'{score:.2f}', color='red', fontsize=8)

        ax.set_title(f'Image: {img_name}', fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    return fig


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes."""
    # Convert inputs to numpy arrays and ensure they're the right shape
    box1 = np.array(box1, dtype=float)
    box2 = np.array(box2, dtype=float)

    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    # Calculate IoU
    if union <= 0:
        return 0
    return intersection / union


def calculate_precision_recall(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    num_pred = len(pred_boxes)
    num_gt = len(gt_boxes)

    if num_pred == 0 or num_gt == 0:
        return np.array([0.0]), np.array([0.0])

    # Calculate IoU matrix between all predicted and ground truth boxes
    iou_matrix = np.zeros((num_pred, num_gt))
    for i in range(num_pred):
        for j in range(num_gt):
            iou_matrix[i, j] = calculate_iou(pred_boxes[i], gt_boxes[j])

    # Sort predictions by confidence score
    sorted_indices = np.argsort(pred_scores)[::-1]

    # Initialize variables for precision-recall calculation
    true_positives = np.zeros(num_pred)
    false_positives = np.zeros(num_pred)
    gt_matched = np.zeros(num_gt)

    # Match predictions to ground truth boxes
    for i in range(num_pred):
        pred_idx = sorted_indices[i]
        max_iou = 0
        max_gt_idx = -1

        # Find the best matching ground truth box
        for j in range(num_gt):
            if gt_matched[j]:
                continue
            if iou_matrix[pred_idx, j] > max_iou:
                max_iou = iou_matrix[pred_idx, j]
                max_gt_idx = j

        # If IoU exceeds threshold, count as true positive
        if max_iou >= iou_threshold and max_gt_idx >= 0:
            true_positives[i] = 1
            gt_matched[max_gt_idx] = 1
        else:
            false_positives[i] = 1

    # Calculate cumulative precision and recall
    cumsum_tp = np.cumsum(true_positives)
    cumsum_fp = np.cumsum(false_positives)
    recalls = cumsum_tp / num_gt if num_gt > 0 else np.ones_like(cumsum_tp)
    precisions = cumsum_tp / (cumsum_tp + cumsum_fp)

    return precisions, recalls


def calculate_ap(precisions, recalls):
    """Calculate Average Precision using 11-point interpolation."""
    ap = 0
    for t in np.arange(0, 1.1, 0.1):  # 11-point interpolation
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11
    return ap


def calculate_map(base_path, model, bbox_model, transform, device):
    """Calculate mAP across all validation images."""
    val_images = [f.split('.')[0] for f in os.listdir(os.path.join(base_path, 'JPEGImages'))
                  if f.endswith('.jpg')]
    aps = []

    for img_name in tqdm(val_images, desc="Calculating mAP"):
        img_path = os.path.join(base_path, 'JPEGImages', f"{img_name}.jpg")
        xml_path = os.path.join(base_path, 'Annotations', f"{img_name}.xml")

        try:
            # Get predictions and ground truth for single image
            img, gt_boxes, pred_boxes, pred_scores = process_single_image(
                img_path, xml_path, model, bbox_model, transform, device
            )

            if not pred_boxes or not gt_boxes:
                continue

            # Calculate precision-recall curve
            precisions, recalls = calculate_precision_recall(pred_boxes, pred_scores, gt_boxes)

            # Calculate AP for this image
            ap = calculate_ap(precisions, recalls)
            aps.append(ap)

        except Exception as e:
            print(f"Error processing {img_name}: {str(e)}")
            continue

    # Calculate mAP
    mAP = np.mean(aps) if aps else 0

    return mAP, aps


def evaluate_detection_performance(base_path):
    """Evaluate detection performance on validation set."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    transform = get_transform()
    model = get_model(device=device)
    bbox_model = get_bbox_regression_model(device=device)

    mAP, aps = calculate_map(base_path, model, bbox_model, transform, device)

    print(f"\nDetection Performance Metrics:")
    print(f"Mean Average Precision (mAP): {mAP:.4f}")
    print(f"Number of images evaluated: {len(aps)}")
    print(f"AP Standard Deviation: {np.std(aps):.4f}")

    return mAP, aps


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 1. 이미지 로드 및 ground truth 바운딩 박스 표시
    # test_img_path의 이미지를 읽고, XML 파일로부터 바운딩 박스 정보를 읽어옵니다.
    # 이를 통해 ground truth 바운딩 박스를 dst 이미지에 그립니다.
    # test_img_path = '../imgs/000007.jpg'
    # test_xml_path = '../imgs/000007.xml'
    test_img_path = '/home/ivpl-d29/dataset/VOC/voc_car/val/JPEGImages/000091.jpg'
    test_xml_path = '/home/ivpl-d29/dataset/VOC/voc_car/val/Annotations/000091.xml'

    img = cv2.imread(test_img_path)
    dst = copy.deepcopy(img)

    bndboxs = parse_xml(test_xml_path)
    for bndbox in bndboxs:
        xmin, ymin, xmax, ymax = bndbox
        cv2.rectangle(dst, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=1)

    # 2. selective search 객체 생성 = get_selective_search()
    gs = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    # 2. Selective Search로 proposal region 제안
    SelectiveSearch.config(gs, img, strategy='f')   # fast mode로 search진행
    rects = SelectiveSearch.get_rects(gs)
    print('추천 proposal region 수： %d' % len(rects))

    # softmax = torch.softmax()

    svm_thresh = 0.1  # 0.6

    # 양성 sample bounding box를 저장
    score_list = list()
    positive_list = list()

    # tmp_score_list = list()
    # tmp_positive_list = list()

    # 3. 데이터 transformation
    transform = get_transform()

    # 4. SVM 학습한 모델 불러오기
    model = get_model(device=device)


    start = time.time()
    for rect in rects:
        '''
        # 5. Selective Search의 proposal region에 대해 예측 수행
        # 각 region proposal을 반복하여, 바운딩 박스 좌표 (xmin, ymin, xmax, ymax)을 추출합니다.
        # 해당 부분 이미지를 잘라내고, 전처리한 후 모델에 입력하여 예측을 수행합니다.
        '''
        xmin, ymin, xmax, ymax = rect
        rect_img = img[ymin:ymax, xmin:xmax]

        rect_transform = transform(rect_img).to(device)
        output = model(rect_transform.unsqueeze(0))[0]

        '''
        # 6. 자동차로 분류된 region의 확률을 저장
        # 모델의 예측 결과에서 자동차로 분류된 바운딩 박스를 저장합니다.
        # Softmax를 사용하여 예측 확률을 계산하고, 자동차에 대한 확률이 svm_thresh=0.6 이상인 경우에만 positive_list와 score_list에 추가합니다.
        '''
        if torch.argmax(output).item() == 1:
            """
            자동차에 대한 예측 
            """
            probs = torch.softmax(output, dim=0).cpu().numpy()

            # tmp_score_list.append(probs[1])
            # tmp_positive_list.append(rect)

            if probs[1] >= svm_thresh:  # svm_thresh=0.6
                score_list.append(probs[1])
                positive_list.append(rect)
                # cv2.rectangle(dst, (xmin, ymin), (xmax, ymax), color=(0, 0, 255), thickness=2)
                print(rect, output, probs)
    end = time.time()
    print('detect time: %d s' % (end - start))

    # tmp_img2 = copy.deepcopy(dst)
    # draw_box_with_text(tmp_img2, tmp_positive_list, tmp_score_list)
    # cv2.imshow('tmp', tmp_img2)
    #
    # tmp_img = copy.deepcopy(dst)
    # draw_box_with_text(tmp_img, positive_list, score_list)
    # cv2.imshow('tmp2', tmp_img)

    '''
    # 7. Non-Maximum Suppression (NMS) 적용
    # nms() 함수를 호출하여, 중복된 바운딩 박스를 제거합니다.
    # IoU(Intersection over Union) 점수가 0.3 이상인 바운딩 박스는 중복으로 판단하여 삭제하고, 최종 선택된 바운딩 박스만 남깁니다.
    '''
    nms_rects, nms_scores = nms(positive_list, score_list)
    print(nms_rects)
    print(nms_scores)
    '''
    # 8. 탐지된 바운딩 박스를 이미지에 표시
    '''
    base_path = '/home/ivpl-d29/dataset/VOC/voc_car/val'
    mAP, aps = evaluate_detection_performance(base_path)
    image_list = ['000404', '000431', '003256', '003806', '006218',
                  '008388', '008483', '008586', '009810', '009898']
    # 시각화 함수 호출
    fig = visualize_multiple_detections(image_list, base_path)

    # 결과 저장
    plt.savefig('detected_image.png', bbox_inches='tight', dpi=300)
    plt.show()

    # draw_box_with_text(dst, nms_rects, nms_scores)

    # cv2.imshow('img', dst)
    # cv2.waitKey(0)

