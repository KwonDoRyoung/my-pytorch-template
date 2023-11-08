# -*-coding: utf-8
from typing import Callable
import argparse
from torch.utils.data import Dataset


def add_argparser_dataset(
    parent_parser: argparse.ArgumentParser, dataset: str
) -> argparse.ArgumentParser:
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


def get_dataset(dataset: str, phase: str, transforms: Callable, **kwargs) -> Dataset:
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
