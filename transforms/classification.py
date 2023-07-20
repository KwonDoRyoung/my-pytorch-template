# -*-coding: utf-8 -*-
from typing import (
    Union,
    List,
    Tuple,
)
import argparse
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


class ClassificationPresetTrain:
    def __init__(
        self,
        crop_size: Union[List[int], None],
        resize: Union[List[int], None],
        brightness: float = 0.3,
        contrast: float = 0.0,
        saturation: float = 0.0,
        hue: float = 0,
        degree: int = 20,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        mean: Union[Tuple[float, float, float], None] = (0, 0, 0),
        std: Union[Tuple[float, float, float], None] = (1, 1, 1),
    ) -> None:
        trans = []
        if isinstance(crop_size, list):
            assert len(crop_size) == 2, "crop size 의 크기는 2이다."
            trans = [
                T.RandomResizedCrop(
                    crop_size, interpolation=InterpolationMode.BILINEAR
                ),
                T.Resize(resize),
            ]
        else:
            trans = [
                T.Resize(resize),
            ]
        trans.extend(
            [
                T.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue,
                )
            ]
        )
        trans.extend([T.RandomRotation(degree, InterpolationMode.BILINEAR)])

        if hflip_prob > 0:
            trans.extend([T.RandomHorizontalFlip(hflip_prob)])
        if vflip_prob > 0:
            trans.extend([T.RandomVerticalFlip(vflip_prob)])

        trans.extend(
            [
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=mean, std=std),
            ]
        )

        self.transforms = T.Compose(trans)

    @staticmethod
    def add_argparser(parent_parser: argparse.ArgumentParser):
        parser = parent_parser.add_argument_group("ClassificationTransformTrain")
        parser.add_argument("--crop_size", default=None, type=int, nargs="+")
        parser.add_argument("--resize", default=[224, 224], type=int, nargs="+")
        parser.add_argument("--brightness", default=0.3, type=float)
        parser.add_argument("--contrast", default=0.0, type=float)
        parser.add_argument("--saturation", default=0.0, type=float)
        parser.add_argument("--hue", default=0, type=float)
        parser.add_argument("--degree", default=20, type=float)
        parser.add_argument("--hflip_prob", default=0.5, type=float)
        parser.add_argument("--vflip_prob", default=0.5, type=float)
        parser.add_argument("--mean", default=[0, 0, 0], type=float, nargs="+")
        parser.add_argument("--std", default=[1, 1, 1], type=float, nargs="+")
        return parent_parser

    def __call__(self, image) -> Union[torch.Tensor, None]:
        return self.transforms(image)


class ClassificationPresetEval:
    def __init__(
        self,
        resize,
        mean=(0, 0, 0),
        std=(1, 1, 1),
    ) -> None:
        self.transforms = T.Compose(
            [
                T.Resize(resize),
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=mean, std=std),
            ]
        )

    @staticmethod
    def add_argparser(parent_parser: argparse.ArgumentParser):
        parser = parent_parser.add_argument_group("ClassificationTransformEval")
        parser.add_argument("--resize", default=[224, 224], type=int, nargs="+")
        parser.add_argument("--mean", default=[0, 0, 0], type=float, nargs="+")
        parser.add_argument("--std", default=[1, 1, 1], type=float, nargs="+")
        return parent_parser

    def __call__(self, image) -> Union[torch.Tensor, None]:
        return self.transforms(image)
