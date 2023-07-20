# -*-coding: utf-8 -*-
import argparse

from classification import ClassificationPresetTrain, ClassificationPresetEval
from segmentation import SegmentationPresetTrain, SegmentationPresetEval


def add_argparser_transform(
    parent_parser: argparse.ArgumentParser,
    task: str,
) -> argparse.ArgumentParser:
    if task == "cls":
        return ClassificationPresetTrain.add_argparser(parent_parser)
    elif task == "seg":
        return SegmentationPresetTrain.add_argparser(parent_parser)
    else:
        raise NotImplementedError(f"{task} is not supported")


def get_transform(task: str, **kwargs):
    if task == "cls":
        train_transforms = ClassificationPresetTrain()
        valid_transforms = ClassificationPresetEval()
    elif task == "seg":
        train_transforms = SegmentationPresetTrain()
        valid_transforms = SegmentationPresetEval()
    else:
        raise NotImplementedError(f"{task} is not supported")
    return train_transforms, valid_transforms
