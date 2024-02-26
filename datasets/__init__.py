# -*-coding: utf-8
from typing import Callable
import argparse
from torch.utils.data import Dataset

from .mnist import MNISTDataset
from .cat_dog import CatDogDataset


def add_argparser_dataset(parent_parser: argparse.ArgumentParser, dataset_name: str) -> argparse.ArgumentParser:
    """데이터셋 이름을 전달하여 데이터셋에 관련된 argument 추가로 받기 위함
        예시)
        dataset = str(dataset).lower()
        if dataset == "dataset-name":
            return dataset-class.add_argparser(parent_parser), dataset-class.max_pixel_value()
        else:
            raise ValueError(f"{dataset} is not existed!")
    Args:
        parent_parser (argparse.ArgumentParser): 기본으로 호출된 argument
        dataset_name (str): 호출할 데이터셋 이름

    Returns:
        argument_parser: 반환 및 이미지의 Max Pixel value 값
    """
    dataset_name = str(dataset_name).lower()
    if dataset_name == "mnist":
        return MNISTDataset.add_argparser(parent_parser), MNISTDataset.max_pixel_value()
    elif dataset_name == "cat_dog":
        return CatDogDataset.add_argparser(parent_parser), CatDogDataset.max_pixel_value()
    else:
        raise ValueError(f"{dataset_name} is not existed!")


def get_dataset(dataset_name: str, is_train: bool, transforms: Callable = None, **kwargs) -> Dataset:
    """
    데이터셋 이름을 전달 받아서 클래스 자체를 반환
        예시)
        dataset = str(dataset).lower()
        if dataset == "dataset-name":
            return dataset-class
        else:
            raise ValueError(f"{dataset} is not existed!")
    Args:
        dataset (str): 호출할 데이터셋 이름

    Returns:
        Dataset Class: 클래스 자체를 반환
    """
    dataset_name = str(dataset_name).lower()
    if dataset_name == "mnist":
        return MNISTDataset(
            root=kwargs.get("root"),
            is_train=is_train,
            transforms=transforms,
        )
    elif dataset_name == "cat_dog":
        return CatDogDataset(
            root=kwargs.get("root"),
            is_train=is_train,
            transforms=transforms,
        )
    else:
        raise ValueError(f"{dataset_name} is not existed!")
