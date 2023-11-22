# -*-coding: utf-8
from argparse import ArgumentParser

import torch.nn as nn

from .vgg import VGG
from .densenet import DenseNet
from .resnet import ResNet
from .convnext import ConvNext
from .efficientnet import EfficientNet


def add_argparser_cls_model(
    parent_parser: ArgumentParser, model_name: str, is_inference: bool
) -> ArgumentParser:
    """모델 이름을 전달하여 모델 관련된 argument 추가로 받기 위함
       예시)
       model = str(model).lower()
       if model.startswith(model-prefix):
           return model-class.add_argparser(parent_parser)
       else:
           raise  ValueError(f"{model} is not supported!")
    Args:
        parent_parser (argparse.ArgumentParser): 기본으로 호출된 argument
        model_name (str): 호출할 모델 이름

    Returns:
        argument_parser: 클래스 자체를 반환
    """
    model_name = str(model_name).lower()
    if model_name.startswith("vgg"):
        return VGG.add_argparser(parent_parser, is_inference)
    elif model_name.startswith("densenet"):
        return DenseNet.add_argparser(parent_parser, is_inference)
    elif ("resnet" in model_name) or model_name.startswith("resnext"):
        return ResNet.add_argparser(parent_parser, is_inference)
    elif model_name.startswith("convnext"):
        return ConvNext.add_argparser(parent_parser, is_inference)
    elif model_name.startswith("efficientnet"):
        return EfficientNet.add_argparser(parent_parser, is_inference)
    else:
        raise ValueError(f"{model_name} is not supported!")


def get_cls_model(
    model_name: str,
    num_classes: int,
    is_inference: bool,
    criterion_name="",
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
        model_name (str): 호출할 데이터셋 이름
        num_classes (int): 모델 최종 output

    Returns:
        Dataset Class: 클래스 자체를 반환
    """
    model_name = str(model_name).lower()
    if model_name.startswith("vgg"):
        return VGG(
            model_name=model_name,
            num_classes=num_classes,
            is_inference=is_inference,
            criterion_name=criterion_name,
            **kwargs,
        )
    elif model_name.startswith("densenet"):
        return DenseNet(
            model_name=model_name,
            num_classes=num_classes,
            is_inference=is_inference,
            criterion_name=criterion_name,
            **kwargs,
        )
    elif ("resnet" in model_name) or model_name.startswith("resnext"):
        return ResNet(
            model_name=model_name,
            num_classes=num_classes,
            is_inference=is_inference,
            criterion_name=criterion_name,
            **kwargs,
        )
    elif model_name.startswith("convnext"):
        return ConvNext(
            model_name=model_name,
            num_classes=num_classes,
            is_inference=is_inference,
            criterion_name=criterion_name,
            **kwargs,
        )
    elif model_name.startswith("efficientnet"):
        return EfficientNet(
            model_name=model_name,
            num_classes=num_classes,
            is_inference=is_inference,
            criterion_name=criterion_name,
            **kwargs,
        )
    else:
        raise RuntimeError(f"{model_name} is not supported!")
