import os
import shutil
import random


def split_dataset(source_folder, large_folder, small_folder, split_ratio=0.7):
    """
    source_folder에서 large_folder와 small_folder로 이미지를 split_ratio에 따라 분리하는 함수
    """
    # 클래스 폴더 목록 가져오기
    classes = os.listdir(source_folder)

    for class_name in classes:
        # 각 클래스 폴더에 있는 이미지 목록 가져오기
        class_folder = os.path.join(source_folder, class_name)
        if not os.path.isdir(class_folder):
            continue

        images = os.listdir(class_folder)
        random.shuffle(images)

        # 분리할 기준 설정
        split_point = int(len(images) * split_ratio)

        large_class_folder = os.path.join(large_folder, class_name)
        small_class_folder = os.path.join(small_folder, class_name)

        os.makedirs(large_class_folder, exist_ok=True)
        os.makedirs(small_class_folder, exist_ok=True)

        # large_folder에 이미지 복사
        for img in images[:split_point]:
            shutil.copy(os.path.join(class_folder, img), os.path.join(large_class_folder, img))

        # small_folder에 이미지 복사
        for img in images[split_point:]:
            shutil.copy(os.path.join(class_folder, img), os.path.join(small_class_folder, img))


# 경로 설정
source_root = "/home/ivpl-d29/dataset/imagenet/"
target_root = "/home/ivpl-d29/dataset/imagenet_split/"

folders = ['train', 'val']

for folder in folders:
    source_folder = os.path.join(source_root, folder)
    large_folder = os.path.join(target_root, f"imagenet_large/{folder}")
    small_folder = os.path.join(target_root, f"imagenet_small/{folder}")

    split_dataset(source_folder, large_folder, small_folder)
