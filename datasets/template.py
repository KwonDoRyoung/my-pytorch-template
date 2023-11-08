# -*- coding: utf-8 -*-
from abc import abstractstaticmethod
from typing import Union, Optional, Callable, List, Dict, Tuple
import os
from argparse import ArgumentParser

import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    root: str
    is_train: Union[bool, None]
    transforms: Optional[Union[Callable, None]]

    inputs: List[str]  # 입력 이미지 데이터 파일 이름
    labels: Union[List[int], List[str]]  # 클래스 넘버 또는 파일

    classes: Dict  # 사용자가 정의하는 부분, classes = {"Class01":0, "Class02":1}
    inv_classes: Dict # {0:"Class01", 1:"Class02"}

    def __init__(self, root, is_train, transforms) -> None:
        super().__init__()
        assert root is not None, "데이터 경로가 정의되지 않았습니다."
        assert is_train is not None, "학습(train: True) 또는 테스트(test:False) 인지 명확하지 않습니다."
        assert transforms is not None, "데이터 Transform 함수가 정의되어 있지 않습니다."
        self.root = root
        self.is_train = is_train

        self.transforms = transforms

        self.inputs = []
        self.labels = []
        
    @abstractstaticmethod
    def max_pixel_value():
        raise NotImplemented("이미지의 최대 Pixel 값을 정의하시오")

    def load_database(self) -> None:
        raise NotImplemented()

    def add_argparser(self, parent_parser: ArgumentParser) -> ArgumentParser:
        raise NotImplemented()

    def read_image(self, file_path):
        raise NotImplemented("file open 구현")
    
    def read_mask(self, file_path):
        raise NotImplemented("Mask File open 미구현")

    def set_transforms(self, transforms):
        self.transforms = transforms

    @property
    def num_classes(self) -> int:
        return len(set(self.classes.values()))

    def __len__(self) -> int:
        assert len(self.inputs) == len(self.labels), "입력 데이터와 라벨 데이터의 개수가 맞지 않습니다."
        return len(self.inputs)


class ClassificationVisionDataset(BaseDataset):
    distribution: torch.Tensor

    def __init__(self, root, is_train, transforms) -> None:
        super().__init__(root, is_train, transforms)

    def __str__(self) -> str:
        phase = "Train" if self.is_train else "Test"
        msg = f"{self.__class__.__name__} | Phase: {phase}\n"
        msg += f"Distribution of Each Class\n"
        for name, label in self.classes.items():
            msg += f"  - {name}({label}): {self.distribution[label]}\n"
        if self.distribution.sum() == self.__len__():
            msg += f">> Total: {self.__len__()}\n"
        else:
            msg += f">> Difference between distribution{self.distribution.sum()} & total{self.__len__()}"
        return msg

    def __getitem__(self, idx) -> Tuple:
        file_path = self.inputs[idx]
        label = self.labels[idx]

        image = self.read_image(file_path=file_path)
        label = torch.tensor(label, dtype=torch.long)

        if self.transforms:
            image = self.transforms(image)["image"]

        if self.is_train:
            return image, label
        else:
            file_name = os.path.basename(file_path).split(".")[0]
            return image, label, file_name


class SegmentationVisionDataset(BaseDataset):
    def __init__(self, root, is_train, transforms) -> None:
        super().__init__(root, is_train, transforms)

    def __str__(self) -> str:
        phase = "Train" if self.is_train else "Test"
        msg = f"{self.__class__.__name__} | Phase: {phase}\n"
        msg += f">> Total: {self.__len__()}\n"
        return msg
    
    def __getitem__(self, idx) -> Tuple:
        file_path = self.inputs[idx]
        mask_path = self.labels[idx]
        
        input_image = self.read_image(file_path=file_path)
        target_mask = self.read_mask(file_path=mask_path)
    
        if self.transforms:
            transformed = self.transforms(image=input_image,mask=target_mask)
            input_image = transformed["image"]
            target_mask = transformed["mask"] 
    
        if self.is_train:
            return input_image, target_mask.long()
        else:
            file_name = os.path.basename(file_path).split(".")[0]
            return input_image, target_mask.long(), file_name