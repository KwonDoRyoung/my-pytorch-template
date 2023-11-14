# -*-coding: utf-8
import argparse

import torch.nn as nn


def add_argparser_mdoel(
    parent_parser: argparse.ArgumentParser, model_name: str
) -> argparse.ArgumentParser:
    """모델 이름을 전달하여 모델 관련된 argument 추가로 받기 위함
       예시)
       model_name = str(model_name).lower()
       if model_name.startswith(model-prefix):
           return model-class.add_argparser(parser)
       else:
           raise  ValueError(f"{model} is not supported!")
    Args:
        parent_parser (argparse.ArgumentParser): 기본으로 호출된 argument
        model (str): 호출할 모델 이름

    Returns:
        argument_parser: 클래스 자체를 반환
    """


def get_model(model_name: str, num_classes: int, **kwargs) -> nn.Module:
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
