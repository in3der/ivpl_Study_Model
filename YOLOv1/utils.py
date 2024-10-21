import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
import textwrap

# IoU
def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]  # (N, 1)
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    # intersection 좌표 구하기
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)

# NMS
def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Does Non Max Suppression given bboxes
    Parameters:
        bboxes (list): 탐지된 모든 bounding box의 리스트 / 형식: [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): 두 박스 간 IoU가 임계값보다 크면 겹치는 박스 중 하나를 제거
        threshold (float): 박스의 confidence 낮은거 필터링하는 임계값
        box_format (str): "midpoint" or "corners" 박스의 좌표 형식
    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]  # 객체가 존재할 확률(confidence)이 threshold보다 높은 것들만 사용
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)  # confidence 기준으로 박스 정렬
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)  # 가장 높은 확률 가진 box를 chosen_box로 선택, pop함

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]  # class가 다른 경우 남겨둔다
               or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
               < iou_threshold  # iou_threshold보다 작은 것 남겨둔다 -> 겹치는 부분이 적은 것은 남겨둔다.
        ]

        bboxes_after_nms.append(chosen_box)  # 남길 것 append - NMS 적용 후 살아남은 박스들 담음

    return bboxes_after_nms     # 중복 제거된 bounding box 리스트 반환


def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20):
    """
    mAP 계산
    Parameters:
        pred_boxes (list): 예측된 바운딩 박스 리스트. 각 bounding box 형식: [train_idx(이미지번호), class_prediction(예측클래스), prob_score(확률점수), x1, y1, x2, y2]
        true_boxes (list): 실제 정답 바운딩 박스 리스트
        iou_threshold (float)
        box_format (str): 바운딩 박스 좌표 형식 (midpoint 혹은 corners)
        num_classes (int): 클래스 수(VOC 20개)
    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """

    # 클래스별 AP를 저장할 리스트
    average_precisions = []

    # 수치적으로 안정성을 확보하기 위한 값
    epsilon = 1e-6

    # 각 클래스별로 AP (Average Precision)을 계산
    for c in range(num_classes):
        detections = []     # 현재 클래스 c에 해당하는 예측 바운딩 박스 리스트 만들기
        ground_truths = []  # 현재 클래스 c에 해당하는 실제 바운딩 박스 리스트 만들기

        # 예측 박스와 실제 박스를 클래스별로 필터링
        # 예측 박스들에서 현재 클래스 c에 해당하는 것들만 필터링
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)
        # 실제 박스들에서 현재 클래스 c에 해당하는 것들만 필터링
        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        # 이미지별 실제 객체(ground truth) 수 계산 - 중복 탐지를 피하기 위해 사용함
        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0]바운딩박스 3개, 1:torch.tensor[0,0,0,0,0] 바운딩박스 5개}
        #
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)   # 해당 이미지에서 ground truth 객체 수만큼 0으로 채워진 텐서 생성
            # 0: 객체 아직 탐지되지 않음 -> 1: 객체 탐지된 경우. 이렇게 바꿔서 중복처리 방지함

        # 예측 박스를 확률에 따라 내림차순으로 정렬 - 확률 있는 인덱스가 [2]
        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # 해당 c 클래스의 ground truth 없으면 그냥 넘어감 - AP 계산 필요없음
        if total_true_bboxes == 0:
            continue

        # 각 예측 박스에 대해 동일한 이미지의 ground truth 가져오기
        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            # 각 ground truth와 예측 박스 사이의 IoU 계산
            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            # iou 임계값에 따른 TP, FP 결정
            # iou가 임계값보다 크면,
            if best_iou > iou_threshold:
                # TP로 간주 (처음 객체 탐지의 건)
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1   # TP로 처리
                    amount_bboxes[detection[0]][best_gt_idx] = 1    # 탐지 완료 표시 (중복탐지 방지)
                # 이미 객체 탐지된 경우 중복으로 보고 FP로 처리
                else:
                    FP[detection_idx] = 1

            # iou가 임계값보다 작으면 FP
            else:
                FP[detection_idx] = 1
        # 누적 TP와 FP계산 -> Recall과 Precision 계산
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon) # 예측한 것 중 옳은 것
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon)) # ground truth 중 옳은 것
        # Precision과 Recall 값에 torch.cat()을 사용하여 초기 값(Recall=0, Precision=1)을 추가
        # 이는 PR 곡선을 0부터 시작하게 만들어 AP 계산 시 정확한 결과를 얻기 위함
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz 사용하여 PR 곡선 아래 면적 (=AP) 구함
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)    # 평균내서 mAP 최종 계산


class_dict = {
    0: '__background__',
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'diningtable',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'pottedplant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tvmonitor',
}

def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # 이미지 데이터를 Axes 객체에 추가, 그려지는 준비
    ax.imshow(im)

    annotations = []       # 박스 정보 기록용 리스트

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle potch
    for box in boxes:
        class_num = int(box[0])  # 클래스 번호
        class_name = class_dict.get(class_num, "Unknown")  # 클래스 이름 가져오기
        confidence = box[1]

        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2

        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

        # 주석에 클래스 이름과 좌표 정보 추가
        annotations.append(f"Class: {class_name}, Confidence: {confidence}, Box: {box}")

    # NMS 결과 및 클래스 이름을 하단에 텍스트로 추가
    annotation_text = "\n".join(annotations)
    # 텍스트가 너무 길어지지 않도록 줄바꿈 적용
    wrapped_text = "\n".join(textwrap.wrap(annotation_text, width=70))
    plt.text(
        0.5, -0.15, wrapped_text,
        ha='center',
        va='top',
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(facecolor='white', alpha=0.7)
    )
    plt.subplots_adjust(bottom=0.3) # 하단 여백

    plt.show()

def get_bboxes(     # DataLoader에서 모델 사용하여 bounding box 예측 -> 예측 박스와 실제 박스 정리하여 반환하는 함수
        loader,
        model,
        iou_threshold,  # NMS 임계값
        threshold,      # confidence 임계값
        pred_format="cells",    # 예측 박스 형식
        box_format="midpoint",  # 바운딩박스 좌표 형식
        device="cuda",
):
    # 예측박스, 실제박스 저장할 리스트 초기화
    all_pred_boxes = []
    all_true_boxes = []

    # 모델 평가 모드로 전환, train_idx로 훈련 인덱스 추적
    model.eval()
    train_idx = 0

    # 배치 단위 학습
    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)  # 예측 진행

        batch_size = x.shape[0] # 현재 배치 크기 가져옴
        true_bboxes = cellboxes_to_boxes(labels)    # 실제 레이블과 모델 예측결과를 bounding box 형태로 변환하는 함수 호출
        bboxes = cellboxes_to_boxes(predictions)

        # NMS 수행 및 박스 저장
        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            if batch_idx == 0 and idx == 0:
                plot_image(x[idx].permute(1,2,0).to("cpu"), nms_boxes)
                # print(f"nms_boxes : {nms_boxes}")

            # 예측 박스 및 실제 박스 저장
            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1  # 배치 끝날때마다 train_idx 증가

    num_pred_boxes = len(all_pred_boxes)
    num_true_boxes = len(all_true_boxes)
    model.train()
    return all_pred_boxes, all_true_boxes, num_pred_boxes, num_true_boxes


def convert_cellboxes(predictions, S=7):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios. Tried to do this
    vectorized, but this resulted in quite difficult to read
    code... Use as a black box? Or implement a more intuitive,
    using 2 for loops iterating range(S) and convert them one
    by one, resulting in a slower but more readable implementation.
    """

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, 30)
    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26:30]
    scores = torch.cat(
        (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(
        -1
    )
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds


def cellboxes_to_boxes(out, S=7):
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])