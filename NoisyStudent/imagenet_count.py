import os


def count_images(folder):
    """
    주어진 폴더 내의 모든 이미지 개수를 세는 함수
    """
    total_images = 0
    class_folders = os.listdir(folder)

    for class_name in class_folders:
        class_folder = os.path.join(folder, class_name)
        if os.path.isdir(class_folder):
            total_images += len(os.listdir(class_folder))

    return total_images


def check_split_ratio(large_folder, small_folder, split_ratio=0.7):
    """
    large_folder와 small_folder의 이미지 개수를 읽어 7:3 비율이 맞는지 확인하는 함수
    """
    large_count = count_images(large_folder)
    small_count = count_images(small_folder)

    total_count = large_count + small_count
    expected_large_count = total_count * split_ratio
    expected_small_count = total_count * (1 - split_ratio)

    print(f"Large folder image count: {large_count}")
    print(f"Small folder image count: {small_count}")
    print(f"Expected large count: {expected_large_count}")
    print(f"Expected small count: {expected_small_count}")

    if abs(large_count - expected_large_count) < 1 and abs(small_count - expected_small_count) < 1:
        print("The split ratio is approximately correct (7:3).")
    else:
        print("The split ratio is not correct.")


# 경로 설정
target_root = "/home/ivpl-d29/dataset/imagenet_split/"
large_folder = os.path.join(target_root, "imagenet_large")
small_folder = os.path.join(target_root, "imagenet_small")

# train, val 각각 확인
folders = ['train', 'val']
for folder in folders:
    print(f"\nChecking split ratio for {folder} dataset:")
    check_split_ratio(os.path.join(large_folder, folder), os.path.join(small_folder, folder))
