# -*-coding: utf-8
from argparse import ArgumentParser

import torch.nn as nn

import timm

def add_argparser_cls_model(
    parent_parser: ArgumentParser,
    model_name: str,
) -> ArgumentParser:
    model_name = str(model_name).lower()
    if model_name in timm.list_models():
        parser = parent_parser.add_argument_group("TIMM model arguments")
        parser.add_argument("--pretrained", action="store_true")
        return parent_parser
    else:
        raise ValueError(f"{model_name} is not supported!")


def create_cls_model(
    model_name: str,
    num_classes: int,
    **kwargs,
) -> nn.Module:
    model_name = str(model_name).lower()
    num_classes = 1 if kwargs.get("task") == "binary" else num_classes
    if model_name in timm.list_models():
        return timm.create_model(
            model_name,
            num_classes=num_classes,
            pretrained=kwargs.pop("pretrained"),
        )
    else:
        raise RuntimeError(f"{model_name} is not supported!")
