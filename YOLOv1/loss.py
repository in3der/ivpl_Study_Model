import torch
import torch.nn as nn
from utils import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")  # 'sum' : loss의 크기를 전체 batch에 대해 더함 - 하나의 scalar 값으로 반환
        #  여러 box예측으로 prediction 더 잘 반영, lambda 가중치 더 잘 반영
        self.S = S  # 7; feature size
        self.B = B  # 2; num_boxes
        self.C = C  # 20; num_class
        self.lambda_coord = 5   # bounding box loss 가중치 값
        self.lambda_noobj = 0.5 # 객체 없는 경우 loss 가중치 값

    def forward(self, predictions, target):
        # predictions 리스트는 torch.Tensor 형태로 (N=배치크기, 7, 7, 30) 형태 가짐
        # 0~19: 20개 클래스 확률
        # 20: confidence score (1번째 bounding box의 객체 존재 여부)
        # 21~24: 1번째 bounding box의 좌표 (x,y,w,h)
        # 25: confidence score (2번째 bounding box의 객체 존재 여부)
        # 26~30: 2번째 bounding box의 좌표 (x,y,w,h)
        # 예측 결과(predictions)가 (N, S*S*(C+B*5))형태로 들어오는데, 이를 (N, S, S, C+B*5)로 변환
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # 2개의 bounding box와 target box에 대한 iou 계산함
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0) # 2개의 iou값을 tensor로 합침
        iou_maxes, bestbox = torch.max(ious, dim=0)  # 두 iou중 큰 값(iou_maxes) 선택, 어떤 박스인지 표시 (bestbox= 0 or 1)
        exists_box = target[..., 20].unsqueeze(3)  # exists_box(1obj_i,j)-> 객체 존재 1, 객체 존재 안하면 0

        # 1. Localization Loss - bounding box에 대한 loss 계산
        box_predictions = exists_box * (    # 객체 있는 grid (exists_box=1)에만 loss를 계산
            (
                    (1 - bestbox) * predictions[..., 21:25]   # bestbox=0이면 1번째 박스 선택
                    + bestbox * predictions[..., 26:30]   # bestbox=1 이면 2번째 박스 선택
            )
        )

        # target box(ground truth) 좌표 (exist_box=1인; 객체 있는 grid만 대상)
        box_targets = exists_box * target[..., 21:25]


        # 예측한 box의 w, h에 sqrt 연산 적용, 음수방지로 sign 사용
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )

        # (N, S, S, 25)
        # ground truth도 w,h에 sqrt 연산 적용
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])  # gt의 h,w 루트 연산 in paper

        # flatten 전: (N, S, S, 4) -> flatten 후: (N*S*S, 4)
        # bounding box loss 계산 - (N,S,S,4)-flatten->(N*S*S,4) 이후 MSE 계싼
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )

        # 2. Confidence Loss - 객체 존재 여부에 대한 loss 계산
        pred_box = ( # bestbox 1, 0에 따른 box 선택
                (1 - bestbox) * predictions[..., 20:21] + bestbox * predictions[..., 25:26]
        )

        # (N*S*S)
        # 객체가 있는 그리드 셀에 대해 예측한 confidence score와 IoU를 곱한 값을 이용해 MSE 계산
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21] * iou_maxes)
            # confidence score를 이용해 loss 계산하기 위해 iou_maxes를 곱해준다. - confidence score를 더 정확하게 조정하기 위해 논문상관없이 씀
        )

        # FOR NO OBJECT LOSS
        #  flatten 전: (N, S, S, 1) -> flatten 후: (N, S*S)
        # 객체가 없는 그리드 셀에 대한 손실 계산
        # 객체 없는 첫 번째 바운딩 박스의 confidence score에 대해 MSE 계산
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        # 객체 없는 두 번째 바운딩 박스의 confidence score에 대해 MSE 계산
        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        # 3. Classification Loss - 어떤 클래스인지 판별
        # flatten 전: (N,S,S,20) -> flatten 후: (N*S*S, 20)
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
            torch.flatten(exists_box * target[..., :20], end_dim=-2)
        )

        loss = (
                self.lambda_coord * box_loss  # bounding box 좌표 loss (Localization Loss)
                + object_loss   # 객체 있는 grid loss   (Confidence Loss)
                + self.lambda_noobj * no_object_loss    # 객체 없는 grid loss
                + class_loss    # class prediction loss (Classification Loss)
        )

        return loss