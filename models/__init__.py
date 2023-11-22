# -*-coding: utf-8
import argparse

import torch.nn as nn

from .segmentation import add_argparser_seg_model, get_seg_model
from .classification import add_argparser_cls_model, get_cls_model


def add_argparser_model(
    task: str,
    parent_parser: argparse.ArgumentParser,
    model_name: str,
    is_inference: bool = False,
) -> argparse.ArgumentParser:
    task = str(task).lower()
    if task == "seg":
        return add_argparser_seg_model(
            parent_parser,
            model_name,
            is_inference,
        )
    elif task == "cls":
        return add_argparser_cls_model(
            parent_parser,
            model_name,
            is_inference,
        )
    else:
        raise ValueError(f"{model_name} is not supported!")


def get_model(
    task: str,
    model_name: str,
    num_classes: int,
    is_inference: bool = False,
    criterion_name: str = "",
    **kwargs,
) -> nn.Module:
    """
    모델 이름 및 파라미터를 전달 받아서 모델 호출
        예시)
        model = str(model).lower()
        pretrained = kwargs.get("pretrained", False)
        if model.startswith(model-prefix):
            return model-class(model, num_classes, pretrained)
        else:
            raise RuntimeError(f"{model} is not supported!")
    Args:
        model (str): 호출할 데이터셋 이름
        num_classes (int): 모델 최종 output

    Returns:
        Dataset Class: 클래스 자체를 반환
    """
    task = str(task).lower()
    if task == "seg":
        return get_seg_model(
            model_name,
            num_classes,
            is_inference,
            criterion_name,
            **kwargs,
        )
    elif task == "cls":
        return get_cls_model(
            model_name,
            num_classes,
            is_inference,
            criterion_name,
            **kwargs,
        )
    else:
        raise RuntimeError(f"{model_name} is not supported!")
