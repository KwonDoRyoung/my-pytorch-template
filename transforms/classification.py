# -*- coding: utf-8 -*-
from typing import (
    Union,
    List,
    Tuple,
    Any,
)
from argparse import ArgumentParser

import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch

class ClassificationPresetTrain:
    def __init__(
        self,
        resize: Union[List[int], int],
        crop_size: Union[List[int], int, None] = None,
        brightness: Union[Tuple, float] = 0.0,
        contrast: Union[Tuple, float] = 0.0,
        saturation: Union[Tuple, float] = 0.0,
        hue: Union[Tuple, float] = 0,
        degree: int = 0,
        hflip_prob: float = 0.0,
        vflip_prob: float = 0.0,
        is_noise: bool = False,
        mean: Tuple[float] = (0,0,0),
        std: Tuple[float] = (1,1,1),
        max_pixel_value: float = 255.0,
        to3channel:bool=False,
    ) -> None:
        self.to3channel = to3channel
        trans = []

        if isinstance(resize, int):
            resize = [resize, resize]
        if isinstance(crop_size, int):
            crop_size = [crop_size, crop_size]
        # trans.append(A.Resize(height=resize[0], width=resize[1], always_apply=True))  # 단순 Resize
        trans.append(A.LongestMaxSize(max_size=max(resize)))
        trans.append(A.PadIfNeeded(min_height=resize[0], min_width=resize[0],border_mode=cv2.BORDER_CONSTANT, value=0.))

        if crop_size is not None:
            trans.append(A.RandomCrop(height=crop_size[0], width=crop_size[1]))

        if isinstance(brightness, list) and len(brightness) == 1:
            brightness = brightness[0]
        if isinstance(contrast, list) and len(contrast) == 1:
            contrast = contrast[0]
        if isinstance(saturation, list) and len(saturation) == 1:
            saturation = saturation[0]
        if isinstance(hue, list) and len(hue) == 1:
            hue = hue[0]

        trans.append(
            A.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
                p=1,
            )
        )

        trans.append(A.Rotate(limit=degree, p=1.0,border_mode=cv2.BORDER_CONSTANT, value=0.))

        if 0 < hflip_prob < 1:
            trans.append(A.HorizontalFlip(p=hflip_prob))
        elif hflip_prob >= 1:
            trans.append(A.HorizontalFlip(p=1))

        if 0 < vflip_prob < 1:
            trans.append(A.VerticalFlip(p=vflip_prob))
        elif vflip_prob >= 1:
            trans.append(A.VerticalFlip(p=1))

        if is_noise:
            trans.append(
                A.OneOf(
                    [
                        A.MotionBlur(p=1),
                        A.OpticalDistortion(p=1),
                        A.GaussNoise(p=1),
                    ],
                    p=.5,
                )
            )
        trans.append(A.Normalize(mean=mean, std=std, max_pixel_value=max_pixel_value))
        trans.append(ToTensorV2())

        self.transforms = A.Compose(transforms=trans)

    @staticmethod
    def add_argparser(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group("ClsTrainTransforms")
        parser.add_argument("--resize", required=True, type=int, nargs="+")
        parser.add_argument("--crop_size", default=None, type=int, nargs="+")
        parser.add_argument("--brightness", default=0.0, type=float, nargs="+")
        parser.add_argument("--contrast", default=0.0, type=float, nargs="+")
        parser.add_argument("--saturation", default=0.0, type=float, nargs="+")
        parser.add_argument("--hue", default=0, type=float, nargs="+")
        parser.add_argument("--degree", default=0, type=float)
        parser.add_argument("--hflip_prob", default=0.5, type=float)
        parser.add_argument("--vflip_prob", default=0.5, type=float)
        parser.add_argument(
            "--is-noise",
            default=False,
            type=bool,
        )
        parser.add_argument("--mean", default=[0, 0, 0], type=float, nargs="+")
        parser.add_argument("--std", default=[1, 1, 1], type=float, nargs="+")
        parser.add_argument("--to3channel", action="store_true")
        return parent_parser

    def __call__(self, image) -> Any:
        imgaug = self.transforms(image=image)
        image = imgaug["image"]
        if self.to3channel:
            new_image = torch.cat([image, image,image], dim=0)
            imgaug["image"] = new_image
            return imgaug
        else:
            return imgaug


class ClassificationPresetEval:
    def __init__(
        self,
        resize: Union[List[int], int],
        mean: Tuple[float] = (0,0,0),
        std: Tuple[float] = (1,1,1),
        max_pixel_value: float = 255.0,
        to3channel:bool=False,
    ) -> None:
        self.to3channel = to3channel
        trans = []

        if isinstance(resize, int):
            resize = [resize, resize]
        # trans.append(A.Resize(height=resize[0], width=resize[1], always_apply=True))
        trans.append(A.LongestMaxSize(max_size=max(resize)))
        trans.append(A.PadIfNeeded(min_height=resize[0], min_width=resize[0],border_mode=cv2.BORDER_CONSTANT, value=0.))

        trans.append(A.Normalize(mean=mean, std=std,max_pixel_value=max_pixel_value))
        trans.append(ToTensorV2())

        self.transforms = A.Compose(transforms=trans)

    @staticmethod
    def add_argparser(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group("ClsTrainTransforms")
        parser.add_argument("--resize", required=True, type=int, nargs="+")
        parser.add_argument("--mean", default=[0, 0, 0], type=float, nargs="+")
        parser.add_argument("--std", default=[1, 1, 1], type=float, nargs="+")
        parser.add_argument("--to3channel", action="store_true")
        return parent_parser

    def __call__(self, image) -> Any:
        imgaug = self.transforms(image=image)
        image = imgaug["image"]
        if self.to3channel:
            new_image = torch.cat([image, image,image], dim=0)
            imgaug["image"] = new_image
            return imgaug
        else:
            return imgaug
