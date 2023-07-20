# -*-coding: utf-8
from typing import Union, Optional, Callable, List, Dict, Tuple

import argparse

import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    root: str
    phase: Union[str, None]
    transforms: Optional[Union[Callable, None]]

    inputs: List[str]  # 입력 이미지 데이터 파일 이름
    labels: Union[List[int], List[str]]  # 클래스 넘버 또는 파일

    classes: Dict[str:int]  # 사용자가 정의하는 부분, classes = {"Class01":0, "Class02":1}

    def __init__(self, root, phase, transforms) -> None:
        super().__init__()
        assert root is not None, "데이터 경로가 정의되지 않았습니다."
        assert phase is not None, "학습(train), 평가(valid) 또는 테스트(test) 인지 명확하지 않습니다."
        assert transforms is not None, "데이터 Transform 함수가 정의되어 있지 않습니다."
        self.root = root
        self.phase = phase

        self.transforms = transforms

        self.inputs = []
        self.labels = []

    def load_database(self) -> None:
        raise NotImplemented()

    @staticmethod
    def add_argparser(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        raise NotImplemented()

    @staticmethod
    def read_image(file_path):
        raise NotImplemented("file open 구현")

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def __len__(self) -> int:
        assert len(self.inputs) == len(self.labels), "입력 데이터와 라벨 데이터의 개수가 맞지 않습니다."
        return len(self.inputs)


class ClassificationVisionDataset(BaseDataset):
    labels: List[int]  # 입력 이미지 Class Number 정보
    distribution: torch.Tensor

    def __init__(self, root, phase, transforms) -> None:
        super().__init__(root, phase, transforms)

    def __str__(self) -> str:
        msg = f"{self.__class__.__name__} | Phase: {self.phase}\n"
        msg += f"Distribution of Each Class\n"
        for name, label in self.classes.items():
            msg += f"  - {name}({label}): {self.distribution[label]}\n"
        msg += f">> Total: {self.__len__()}\n"
        return msg

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path = self.inputs[idx]
        label = self.labels[idx]

        image = self.read_image(file_path=file_path)
        label = torch.tensor(label, dtype=torch.long)

        if self.transforms:
            image = self.transforms(image)

        return image, label


class SegmentationVisionDataset(BaseDataset):
    labels: List[str]  # 입력 이미지에 대응되는 Mask 파일 이름

    def __init__(self, root, phase, transforms) -> None:
        super().__init__(root, phase, transforms)

    def __str__(self) -> str:
        msg = f"{self.__class__.__name__} | Phase: {self.phase}\n"
        msg += "Segmentation Label\n"
        for name, label in self.classes.items():
            msg += f"  - {name}: {label}\n"
        msg += f">> Total: {self.__len__()}\n"
        return msg

    @staticmethod
    def read_label(file_path):
        raise NotImplemented("read label file 구현 필요")

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.inputs[idx]
        label_path = self.labels[idx]

        image = self.read_image(file_path=image_path)
        label = self.read_label(file_path=label_path)

        if self.transforms:
            image, label = self.transforms(image, label)

        return image, label
