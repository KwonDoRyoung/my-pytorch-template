# -*- coding: utf-8 -*-
from typing import (
    Any,
    List,
    Union,
    Tuple,
)
from argparse import ArgumentParser

import cv2
import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2

from .utils import ConverterCh1toCh3, flip


def add_argparser_seg_transform(parent_parser: ArgumentParser, is_train: bool) -> ArgumentParser:
    if is_train:
        return SegmentationPresetTrain.add_argparser(parent_parser)
    else:
        return SegmentationPresetEval.add_argparser(parent_parser)


def create_seg_transform(is_train: bool, max_pixel_value: float, **kwargs):
    if is_train:
        train_transforms = SegmentationPresetTrain(
            resize=kwargs.get("resize"),
            degree=kwargs.get("degree", 0),
            hflip_prob=kwargs.get("hflip_prob", 0),
            vflip_prob=kwargs.get("vflip_prob", 0),
            mean=kwargs.get("mean"),
            std=kwargs.get("std"),
            max_pixel_value=max_pixel_value,
        )
        valid_transforms = SegmentationPresetEval(
            resize=kwargs.get("resize"),
            mean=kwargs.get("mean"),
            std=kwargs.get("std"),
            max_pixel_value=max_pixel_value,
        )
        return train_transforms, valid_transforms
    else:
        test_transforms = SegmentationPresetEval(
            resize=kwargs.get("resize"),
            mean=kwargs.get("mean"),
            std=kwargs.get("std"),
            max_pixel_value=max_pixel_value,
        )
        return test_transforms


class SegmentationPresetTrain:
    def __init__(
        self,
        resize: Union[List[int], int],
        degree: int = 0,
        hflip_prob: float = 0.0,
        vflip_prob: float = 0.0,
        max_pixel_value: float = 255.0,
        mean: Tuple[float] = (0, 0, 0),
        std: Tuple[float] = (1, 1, 1),
    ) -> None:
        trans = []

        if isinstance(resize, int):
            resize = [resize, resize]

        trans.append(A.Resize(height=resize[0], width=resize[1], always_apply=True))

        if degree != 0:
            trans.append(
                A.Rotate(
                    limit=degree,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0.0,
                )
            )

        trans.extend(flip(hflip_prob, vflip_prob))
        trans.extend(
            [
                ConverterCh1toCh3(),
                A.Normalize(mean=mean, std=std, max_pixel_value=max_pixel_value),
                ToTensorV2(),
            ]
        )

        self.transforms = A.Compose(transforms=trans)

    def __str__(self) -> str:
        return print(self.transforms)

    @staticmethod
    def add_argparser(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group("TrainTransforms")
        parser.add_argument("--resize", required=True, type=int, nargs="+")
        parser.add_argument("--degree", default=0, type=float)
        parser.add_argument("--hflip-prob", default=0.5, type=float)
        parser.add_argument("--vflip-prob", default=0.5, type=float)
        parser.add_argument("--mean", default=[0, 0, 0], type=float, nargs="+")
        parser.add_argument("--std", default=[1, 1, 1], type=float, nargs="+")
        return parent_parser

    def __call__(self, image, mask) -> Any:
        return self.transforms(image=image, mask=mask)


class SegmentationPresetEval:
    def __init__(
        self,
        resize: Union[List[int], int],
        max_pixel_value: float = 255.0,
        mean: Tuple[float] = (0, 0, 0),
        std: Tuple[float] = (1, 1, 1),
    ) -> None:
        if isinstance(resize, int):
            resize = [resize, resize]

        self.transforms = A.Compose(
            [
                A.Resize(height=resize[0], width=resize[1], always_apply=True),
                ConverterCh1toCh3(),
                A.Normalize(mean=mean, std=std, max_pixel_value=max_pixel_value),
                ToTensorV2(),
            ]
        )

    def __str__(self) -> str:
        return print(self.transforms)

    @staticmethod
    def add_argparser(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group("TrainTransforms")
        parser.add_argument("--resize", required=True, type=int, nargs="+")
        parser.add_argument("--mean", default=[0, 0, 0], type=float, nargs="+")
        parser.add_argument("--std", default=[1, 1, 1], type=float, nargs="+")
        return parent_parser

    def __call__(self, image, mask) -> Any:
        return self.transforms(image=image, mask=mask)
