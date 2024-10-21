import torch
import os
import pandas as pd
from PIL import Image


class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None):
        # 1. CSV 파일에서 이미지와 라벨 경로 읽기
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir  # 이미지 저장된 경로
        self.label_dir = label_dir  # 라벨 저장된 경로
        self.transform = transform
        self.S = S  # 이미지 그리드 S x S
        self.B = B  # 바운딩 박스의 개수 = 2
        self.C = C  # 클래스의 수 (VOC 데이터셋에서는 20)

    def __len__(self):  # 2. 데이터셋의 전체 이미지 수 반환 (CSV 파일의 행 수) -> 모델 학습 시 데이터셋 순차적 처리에 이용
        return len(self.annotations)

    def __getitem__(self, index):   # 3. 특정 인덱스에 해당하는 데이터(라벨) 불러오기
        """
        :param index: 데이터셋에서의 인덱스
        :return: 변환된 이미지와 해당하는 라벨 매트릭스
        """
        # 3-1. 주어진 인덱스에 해당하는 라벨 파일 열기
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []

        # 3-2. 라벨 파일에서 바운딩 박스 정보 읽어오기
        with open(label_path) as f:
            for label in f.readlines():
                # 라벨 파일의 각 줄에서 class_label, x, y, width, height를 읽어와서 리스트로 변환
                # ex) 10 0.523 0.7786666666666666 0.43 0.44266666666666665
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]
                # 변환한 정보를 box라는 리스트에 저장
                boxes.append([class_label, x, y, width, height])

        # 4. 인덱스 번호에 해당하는 이미지 불러오기
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)

        # 5. 바운딩 박스 정보(class_label, x, y, width, height)를 텐서로 변환
        boxes = torch.tensor(boxes)

        # 6. 라벨, 이미지 같이 전처리
        if self.transform:
            # augmentation하는 transform 있을 경우 이미지와 bounding box를 함께 전처리
            image, boxes = self.transform(image, boxes)

        # 7. YOLO에 맞도록 라벨 매트릭스(행렬) 구성
        # 바운딩 박스와 클래스 정보를 YOLO 모델에 맞는 포맷으로 변환함
        # -> 이미지 크기 기준으로 받아온 box 정보를 grid cell 기준으로 변환 (grid cell 내의 상대좌표로...)

        # S x S x (C + 5B) 크기의 라벨 매트릭스 초기화 (클래스 확률 + 바운딩 박스 정보) 저장 위함
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))

        # 각 바운딩 박스에 대해 매트릭스(행렬)에 정보 할당
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # 이미지 좌표를 그리드 좌표로 변환
            i, j = int(self.S * y), int(self.S * x)  # 그리드 셀의 위치 계산
            x_cell, y_cell = self.S * x - j, self.S * y - i  # 그리드 셀 내에서의 상대 좌표
            width_cell, height_cell = width * self.S, height * self.S  # 그리드 셀 크기로 바운딩 박스 크기 조정

            # 해당 그리드 셀에 객체가 없을 경우 = 아직 객체 예측을 하지 않은 경우
            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1  # 객체가 있음을 표시 (objectness score 1) = 이제 객체 처리할거임 이라고 표시
                # 이후 해당 객체의 bounding box 좌표와 class 정보를 grid cell 라벨 행렬에 저장하는 것.
                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])  # 바운딩 박스 좌표
                label_matrix[i, j, 21:25] = box_coordinates  # 좌표 정보 할당
                label_matrix[i, j, class_label] = 1  # 클래스 확률 (one-hot 인코딩 방식으로 클래스 할당)

        # 이미지와 라벨 매트릭스 반환
        return image, label_matrix
