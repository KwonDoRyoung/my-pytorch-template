# -*-coding: utf-8 -*-
# https://kuklife.tistory.com/121
# https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/deeplabv3.py
from typing import Dict, List, Sequence
from argparse import ArgumentParser

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

from ..modules import ASPP


def add_argparser_deeplab_model(parent_parser: ArgumentParser, model_name: str) -> ArgumentParser:
    if model_name == "deeplabv3":
        return DeepLabV3.add_argparser(parent_parser)
    elif model_name == "deeplabv3+":
        return DeepLabV3Plus.add_argparser(parent_parser)
    else:
        raise RuntimeError(f"{model_name} is not supported!")


def create_deeplab_model(
    model_name: str,
    num_classes: int,
    **kwargs,
) -> nn.Module:
    model_name = str(model_name).lower()
    num_classes = 1 if kwargs.get("task") == "binary" else num_classes
    if model_name == "deeplabv3":
        return DeepLabV3(
            num_classes=num_classes,
            backbone_name=kwargs.get("backbone_name"),
            backbone_pretrained=kwargs.get("backbone_pretrained"),
            atrous_rates=kwargs.get("atrous_rates"),
        )
    elif model_name == "deeplabv3+":
        return DeepLabV3Plus(
            num_classes=num_classes,
            backbone_name=kwargs.get("backbone_name"),
            backbone_pretrained=kwargs.get("backbone_pretrained"),
            aspp_out_channels=kwargs.get("aspp_out_channels"),
            atrous_rates=kwargs.get("atrous_rates"),
            proj_out_channels=kwargs.get("proj_out_channels"),
            cls_inter_channels=kwargs.get("cls_inter_channels"),
        )
    else:
        raise RuntimeError(f"{model_name} is not supported!")


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels: int, num_classes: int, atrous_rates: Sequence[int] = (12, 24, 36)) -> None:
        super().__init__(
            ASPP(in_channels, atrous_rates),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1),
        )


class DeepLabV3(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone_name: str,
        backbone_pretrained: bool = False,
        atrous_rates: Sequence[int] = (12, 24, 36),
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            model_name=backbone_name,
            pretrained=backbone_pretrained,
            features_only=True,
            out_indices=(-1,),
        )
        self.backbone_name = backbone_name
        self.backbone_pretrained = backbone_pretrained
        self.feature_info = self.backbone.feature_info

        in_channels = self.feature_info.channels()[-1]
        self.classifier = DeepLabHead(in_channels=in_channels, num_classes=num_classes, atrous_rates=atrous_rates)

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.backbone(x)

    def add_argparser(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("DeepLabV3")
        parser.add_argument("--backbone-name", type=str, required=True)
        parser.add_argument("--backbone-pretrained", action="store_true")
        parser.add_argument("--atrous-rates", type=int, nargs="+")
        return parent_parser

    def __str__(self) -> str:
        msg = f"DeepLabV3 - Backbone name: {self.backbone_name} [pretrained = {self.backbone_pretrained}]\n"
        return msg

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        result = OrderedDict()
        input_shape = x.shape[-2:]
        features = self.forward_features(x)[-1]
        logits = self.classifier(features)
        logits = F.interpolate(logits, size=input_shape, mode="bilinear", align_corners=False)
        result["out"] = logits
        return result


class DeepLabV3Plus(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone_name: str,
        backbone_pretrained: bool = False,
        aspp_out_channels: int = 256,
        atrous_rates: Sequence[int] = (12, 24, 36),
        proj_out_channels: int = 48,
        cls_inter_channels: int = 256,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            model_name=backbone_name,
            pretrained=backbone_pretrained,
            features_only=True,
            out_indices=(-3, -1),
        )
        self.backbone_name = backbone_name
        self.backbone_pretrained = backbone_pretrained
        self.feature_info = self.backbone.feature_info

        in_channels = self.feature_info.channels()[-1]
        self.aspp = ASPP(in_channels=in_channels, atrous_rates=atrous_rates, out_channels=aspp_out_channels)

        low_level_in_channels = self.feature_info.channels()[-2]
        self.projection = nn.Sequential(
            nn.Conv2d(low_level_in_channels, proj_out_channels, 1, bias=False),
            nn.BatchNorm2d(proj_out_channels),
            nn.ReLU(inplace=True),
        )

        in_channels = proj_out_channels + aspp_out_channels
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, cls_inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(cls_inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(cls_inter_channels, num_classes, 1),
        )

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.backbone(x)

    def add_argparser(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("DeepLabV3+")
        parser.add_argument("--backbone-name", type=str, required=True)
        parser.add_argument("--backbone-pretrained", action="store_true")
        parser.add_argument("--aspp-out-channels", default=256, type=int)
        parser.add_argument("--atrous-rates", default=(12, 24, 36), type=int, nargs="+")
        parser.add_argument("--proj-out-channels", default=48, type=int)
        parser.add_argument("--cls-inter-channels", default=256, type=int)
        return parent_parser

    def __str__(self) -> str:
        msg = f"DeepLabV3+ - Backbone name: {self.backbone_name} [pretrained = {self.backbone_pretrained}]\n"
        return msg

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        result = OrderedDict()
        input_shape = x.shape[-2:]
        features = self.forward_features(x)
        high_level_features = features[-1]
        low_level_features = features[-2]

        x = self.aspp(high_level_features)
        proj_x = self.projection(low_level_features)
        proj_x_shape = proj_x.shape[-2:]
        x = F.interpolate(x, size=proj_x_shape, mode="bilinear", align_corners=False)

        x = torch.cat([x, proj_x], dim=1)
        logits = self.classifier(x)
        logits = F.interpolate(logits, size=input_shape, mode="bilinear", align_corners=False)
        result["out"] = logits
        return result
