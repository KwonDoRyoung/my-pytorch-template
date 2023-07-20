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


class SegmentationPresetTrain:
    def __init__(self) -> None:
        pass

    @staticmethod
    def add_argparser(parent_parser: argparse.ArgumentParser):
        pass

    def __call__(
        self, image, label
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, None]]:
        return self.transforms(image, label)


class SegmentationPresetEval:
    def __init__(self) -> None:
        pass

    @staticmethod
    def add_argparser(parent_parser: argparse.ArgumentParser):
        pass

    def __call__(
        self, image, label
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, None]]:
        return self.transforms(image, label)
