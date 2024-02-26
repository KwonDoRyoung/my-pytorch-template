# -*-coding: utf-8
import argparse

import torch.nn as nn

from .fcn import add_argparser_fcn_model, create_fcn_model
from .unet import add_argparser_unet_model, create_unet_model
from .deeplab import add_argparser_deeplab_model, create_deeplab_model


def add_argparser_seg_model(
    parent_parser: argparse.ArgumentParser,
    model_name: str,
) -> argparse.ArgumentParser:
    model_name = str(model_name).lower()
    if model_name.startswith("fcn"):
        return add_argparser_fcn_model(parent_parser)
    elif model_name.startswith("unet"):
        return add_argparser_unet_model(parent_parser, model_name)
    elif model_name.startswith("deeplab"):
        return add_argparser_deeplab_model(parent_parser, model_name)
    else:
        raise ValueError(f"{model_name} is not supported!")


def create_seg_model(
    model_name: str,
    num_classes: int,
    **kwargs,
) -> nn.Module:
    model_name = str(model_name).lower()
    if model_name.startswith("fcn"):
        return create_fcn_model(
            model_name=model_name,
            num_classes=num_classes,
            **kwargs,
        )
    elif model_name.startswith("unet"):
        return create_unet_model(
            model_name=model_name,
            num_classes=num_classes,
            **kwargs,
        )
    elif model_name.startswith("deeplab"):
        return create_deeplab_model(
            model_name=model_name,
            num_classes=num_classes,
            **kwargs,
        )
    else:
        raise RuntimeError(f"{model_name} is not supported!")
