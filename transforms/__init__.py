# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from .classification import ClassificationPresetTrain, ClassificationPresetEval
from .segmentation import SegmentationPresetTrain, SegmentationPresetEval


def add_argparser_transform(
    parent_parser: ArgumentParser, task: str, is_train: bool
) -> ArgumentParser:
    if task == "cls":
        if is_train:
            return ClassificationPresetTrain.add_argparser(parent_parser)
        else:
            return ClassificationPresetEval.add_argparser(parent_parser)
    elif task == "seg":
        if is_train:
            return SegmentationPresetTrain.add_argparser(parent_parser)
        else:
            return SegmentationPresetEval.add_argparser(parent_parser)
    else:
        raise NotImplementedError(f"{task} is not supported")


def get_transform(task: str, is_train: bool, max_pixel_value: float, **kwargs):
    if task == "cls":
        if is_train:
            train_transforms = ClassificationPresetTrain(
                resize=kwargs.get("resize"),
                crop_size=kwargs.get("crop_size"),
                brightness=kwargs.get("brightness"),
                contrast=kwargs.get("contrast"),
                saturation=kwargs.get("saturation"),
                hue=kwargs.get("hue"),
                degree=kwargs.get("degree"),
                hflip_prob=kwargs.get("hflip_prob"),
                vflip_prob=kwargs.get("vflip_prob"),
                mean=kwargs.get("mean"),
                std=kwargs.get("std"),
                max_pixel_value=max_pixel_value,
                to3channel=kwargs.get("to3channel"),
            )
            valid_transforms = ClassificationPresetEval(
                resize=kwargs.get("resize"),
                mean=kwargs.get("mean"),
                std=kwargs.get("std"),
                max_pixel_value=max_pixel_value,
                to3channel=kwargs.get("to3channel"),
            )
            return train_transforms, valid_transforms
        else:
            test_transforms = ClassificationPresetEval(
                resize=kwargs.get("resize"),
                mean=kwargs.get("mean"),
                std=kwargs.get("std"),
                max_pixel_value=max_pixel_value,
                to3channel=kwargs.get("to3channel"),
            )
            return test_transforms
    elif task == "seg":
        if is_train:
            train_transforms = SegmentationPresetTrain(
                resize=kwargs.get("resize"),
                crop_size=kwargs.get("crop_size"),
                brightness=kwargs.get("brightness"),
                contrast=kwargs.get("contrast"),
                saturation=kwargs.get("saturation"),
                hue=kwargs.get("hue"),
                degree=kwargs.get("degree"),
                hflip_prob=kwargs.get("hflip_prob"),
                vflip_prob=kwargs.get("vflip_prob"),
                mean=kwargs.get("mean"),
                std=kwargs.get("std"),
                max_pixel_value=max_pixel_value,
                to3channel=kwargs.get("to3channel"),
            )
            valid_transforms = SegmentationPresetEval(
                resize=kwargs.get("resize"),
                mean=kwargs.get("mean"),
                std=kwargs.get("std"),
                max_pixel_value=max_pixel_value,
                to3channel=kwargs.get("to3channel"),
            )
            return train_transforms, valid_transforms
        else:
            test_transforms = SegmentationPresetEval(
                resize=kwargs.get("resize"),
                mean=kwargs.get("mean"),
                std=kwargs.get("std"),
                max_pixel_value=max_pixel_value,
                to3channel=kwargs.get("to3channel"),
            )
            return test_transforms
    else:
        raise NotImplementedError(f"{task} is not supported")
