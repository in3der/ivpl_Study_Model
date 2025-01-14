import sys
sys.path.append("/home/ivpl-d29/myProject/Study_Model/R-CNN")
import numpy  as np
import random
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils.data.custom_finetune_dataset import CustomFinetuneDataset


class CustomBatchSampler(Sampler):
    def __init__(self, num_positive, num_negative, batch_positive, batch_negative) -> None:
        """
        CustomBatchSampler 초기화:
        - 데이터셋 내 양성 샘플과 음성 샘플의 개수를 받고,
        - 단일 배치에 포함될 양성/음성 샘플 수를 설정

        2개의 classification dataset
        한 batch 일괄 처리에 batch_positive 샘플과 batch_negative 샘플이 포함됨.
        @param num_positive: 양성 샘플 수
        @param num_negative: 음성 샘플 수
        @param batch_positive: 단일 양성 샘플 수
        @param batch_negative: 단일 음성 샘플 수
        """
        self.num_positive = num_positive
        self.num_negative = num_negative
        self.batch_positive = batch_positive
        self.batch_negative = batch_negative

        # 전체 데이터셋 인덱스 리스트 생성
        length = num_positive + num_negative
        self.idx_list = list(range(length))

        # 배치 크기과 전체 배치 개수 계산
        self.batch = batch_negative + batch_positive
        self.num_iter = length // self.batch

    # 원본
    # def __iter__(self):
    #     """
    #             CustomBatchSampler의 핵심 로직:
    #             - 각 배치를 구성할 양성/음성 샘플을 무작위로 선택하여 배치를 생성.
    #             - 모든 배치를 리스트로 반환.
    #     """
    #     sampler_list = list()
    #     for i in range(self.num_iter):
    #         tmp = np.concatenate(
    #             (random.sample(self.idx_list[:self.num_positive], self.batch_positive),
    #              random.sample(self.idx_list[self.num_positive:], self.batch_negative))
    #         )
    #         random.shuffle(tmp)
    #         sampler_list.extend(tmp)    # 배치 리스트에 추가
    #     return iter(sampler_list)

    def __iter__(self): # 양성/음성 샘플의 인덱스를 무작위로 반복하지 않고 '중복없이' 배치를 생성하는 코드
        positive_indices = self.idx_list[:self.num_positive]
        negative_indices = self.idx_list[self.num_positive:]

        # 양성/음성 인덱스를 순환하며 샘플링
        sampler_list = []
        for i in range(self.num_iter):
            positive_batch = np.random.choice(positive_indices, self.batch_positive, replace=False)
            negative_batch = np.random.choice(negative_indices, self.batch_negative, replace=False)

            tmp = np.concatenate((positive_batch, negative_batch))
            np.random.shuffle(tmp)
            sampler_list.extend(tmp)
        return iter(sampler_list)

    def __len__(self) -> int:
        """
        전체 샘플러 길이 (배치 개수 * 배치 크기)
        """
        return self.num_iter * self.batch

    def get_num_batch(self) -> int:
        """
        생성 가능한 배치의 총 개수 반환
        """
        return self.num_iter

def test():
    root_dir = '/home/ivpl-d29/dataset/VOC/voc_car/finetune_car/train'
    # 데이터셋 초기화
    train_data_set = CustomFinetuneDataset(root_dir)
    # CustomBatchSampler 초기화
    train_sampler = CustomBatchSampler(
        train_data_set.get_positive_num(),  # 양성 샘플 개수
        train_data_set.get_negative_num(),  # 음성 샘플 개수
        32,  # 배치 내 양성 샘플 개수
        96  # 배치 내 음성 샘플 개수
    )

    # 전체 샘플러 길이와 배치 개수 확인
    print('sampler len(전체 샘플러 길이): %d' % train_sampler.__len__())  # 전체 샘플 크기
    print('sampler batch num(배치 개수): %d' % train_sampler.get_num_batch())  # 배치 개수

    # 양성/음성 샘플 개수 확인:
    print(
        f"Positive samples: {train_data_set.get_positive_num()}, Negative samples: {train_data_set.get_negative_num()}")

    print("/n")
    # 첫 번째, 두 번째, 세 번째 배치 샘플링 결과 확인
    batch_indices = list(train_sampler.__iter__())

    # 첫 번째 배치 확인
    first_batch = batch_indices[:128]
    print("첫 번째 배치의 샘플 인덱스 리스트: ", first_batch)
    print("첫 번째 배치 길이: ", len(first_batch))
    print('positive batch(첫 번째 배치에서 양성 샘플 개수): %d' % np.sum(np.array(first_batch) < 66517))
    print('negative batch(첫 번째 배치에서 음성 샘플 개수): %d' % np.sum(np.array(first_batch) >= 66517))

    # 두 번째 배치 확인
    second_batch = batch_indices[128:256]
    print("\n두 번째 배치의 샘플 인덱스 리스트: ", second_batch)
    print("두 번째 배치 길이: ", len(second_batch))
    print('positive batch(두 번째 배치에서 양성 샘플 개수): %d' % np.sum(np.array(second_batch) < 66517))
    print('negative batch(두 번째 배치에서 음성 샘플 개수): %d' % np.sum(np.array(second_batch) >= 66517))

    # 세 번째 배치 확인
    third_batch = batch_indices[256:384]
    print("\n세 번째 배치의 샘플 인덱스 리스트: ", third_batch)
    print("세 번째 배치 길이: ", len(third_batch))
    print('positive batch(세 번째 배치에서 양성 샘플 개수): %d' % np.sum(np.array(third_batch) < 66517))
    print('negative batch(세 번째 배치에서 음성 샘플 개수): %d' % np.sum(np.array(third_batch) >= 66517))


def test2():
    root_dir = '/home/ivpl-d29/dataset/VOC/voc_car/finetune_car/train'
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data_set = CustomFinetuneDataset(root_dir, transform=transform)
    train_sampler = CustomBatchSampler(train_data_set.get_positive_num(), train_data_set.get_negative_num(), 32, 96)
    data_loader = DataLoader(train_data_set, batch_size=128, sampler=train_sampler, num_workers=8, drop_last=True)

    inputs, targets = next(data_loader.__iter__())
    print(targets)
    print(inputs.shape)


if __name__ == '__main__':
    test()
    test2()