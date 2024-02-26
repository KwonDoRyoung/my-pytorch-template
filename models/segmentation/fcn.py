# -*-coding: utf-8 -*-
# https://github.com/wkentaro/pytorch-fcn/blob/main/torchfcn/models/fcn8s.py
# https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/fcn.py
# https://kuklife.tistory.com/117
from typing import List, Dict, Optional
from argparse import ArgumentParser

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm


def add_argparser_fcn_model(parent_parser: ArgumentParser) -> ArgumentParser:
    return FCN.add_argparser(parent_parser)


def create_fcn_model(
    model_name: str,
    num_classes: int,
    **kwargs,
) -> nn.Module:
    model_name = str(model_name).lower()
    num_classes = 1 if kwargs.get("task") == "binary" else num_classes
    if model_name == "fcn32":
        fcn_model = FCN32
    elif model_name == "fcn16":
        fcn_model = FCN16
    elif model_name == "fcn8":
        fcn_model = FCN8
    else:
        raise RuntimeError(f"{model_name} is not supported!")
    return fcn_model(
        num_classes=num_classes,
        backbone_name=kwargs.get("backbone_name"),
        backbone_pretrained=kwargs.get("backbone_pretrained"),
        drop_rate=kwargs.get("drop_rate"),
    )


class FCN(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        backbone_pretrained: bool = False,
        out_indices: Optional[tuple] = None,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            model_name=backbone_name,
            pretrained=backbone_pretrained,
            features_only=True,
            out_indices=out_indices,
        )
        self.backbone_name = backbone_name
        self.backbone_pretrained = backbone_pretrained
        self.feature_info = self.backbone.feature_info

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.backbone(x)
    
    @staticmethod
    def add_argparser(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("FCN")
        parser.add_argument("--backbone-name", type=str, required=True)
        parser.add_argument("--backbone-pretrained", action="store_true")
        parser.add_argument("--drop-rate", type=float, default=0.0)
        return parent_parser

    def __str__(self) -> str:
        msg = f"Backbone name: {self.backbone_name} [pretrained = {self.backbone_pretrained}]\n"
        return msg


class FCN32(FCN):
    def __init__(
        self,
        num_classes: int,
        backbone_name: str,
        backbone_pretrained: bool = False,
        drop_rate: float = 0.5,
    ) -> None:
        super().__init__(backbone_name, backbone_pretrained, out_indices=(-1,))
        in_channels = self.feature_info.channels()[-1]
        if backbone_name.startswith("vgg"):
            inter_channels = 4096
            self.classifier = nn.Sequential(
                nn.Conv2d(in_channels, inter_channels, 7),
                nn.ReLU(inplace=True),
                nn.Dropout2d(drop_rate),
                nn.Conv2d(inter_channels, inter_channels, 1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(drop_rate),
                nn.Conv2d(inter_channels, num_classes, 1),
            )
        else:
            inter_channels = in_channels // 4
            self.classifier = nn.Sequential(
                nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(),
                nn.Dropout(drop_rate),
                nn.Conv2d(inter_channels, num_classes, 1),
            )

    def __str__(self) -> str:
        msg = f"FCN32 - {super().__str__()}"
        return msg

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        result = OrderedDict()
        input_shape = x.shape[-2:]
        features = self.forward_features(x)[-1]
        logits = self.classifier(features)
        logits = F.interpolate(logits, size=input_shape, mode="bilinear", align_corners=False)
        result["out"] = logits
        return result


class FCN16(FCN):
    def __init__(
        self,
        num_classes: int,
        backbone_name: str,
        backbone_pretrained: bool = False,
        drop_rate: float = 0.5,
    ) -> None:
        super().__init__(backbone_name, backbone_pretrained, out_indices=(-2, -1))
        in_channels = self.feature_info.channels()[-1]
        pool4_in_channels = self.feature_info.channels()[-2]
        self.score_pool4 = nn.Conv2d(pool4_in_channels, num_classes, 1)

        if backbone_name.startswith("vgg"):
            inter_channels = 4096
            self.classifier = nn.Sequential(
                nn.Conv2d(in_channels, inter_channels, 1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(drop_rate),
                nn.Conv2d(inter_channels, inter_channels, 1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(drop_rate),
                nn.Conv2d(inter_channels, num_classes, 1),
            )
        else:
            inter_channels = in_channels // 4
            self.classifier = nn.Sequential(
                nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(),
                nn.Dropout(drop_rate),
                nn.Conv2d(inter_channels, num_classes, 1),
            )

    def __str__(self) -> str:
        msg = f"FCN16 - {super().__str__()}"
        return msg

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        result = OrderedDict()
        input_shape = x.shape[-2:]
        features = self.forward_features(x)
        logits = self.classifier(features[-1])

        pool4_shape = features[-2].shape[-2:]
        logits = F.interpolate(logits, size=pool4_shape, mode="bilinear", align_corners=False)
        logits = logits + self.score_pool4(features[-2])
        logits = F.interpolate(logits, size=input_shape, mode="bilinear", align_corners=False)
        result["out"] = logits
        return result


class FCN8(FCN):
    def __init__(
        self,
        num_classes: int,
        backbone_name: str,
        backbone_pretrained: bool = False,
        drop_rate: float = 0.5,
    ) -> None:
        super().__init__(backbone_name, backbone_pretrained, out_indices=(-3, -2, -1))
        in_channels = self.feature_info.channels()[-1]
        pool4_in_channels = self.feature_info.channels()[-2]
        pool3_in_channels = self.feature_info.channels()[-3]
        self.score_pool4 = nn.Conv2d(pool4_in_channels, num_classes, 1)
        self.score_pool3 = nn.Conv2d(pool3_in_channels, num_classes, 1)

        if backbone_name.startswith("vgg"):
            inter_channels = 4096
            self.classifier = nn.Sequential(
                nn.Conv2d(in_channels, inter_channels, 1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(drop_rate),
                nn.Conv2d(inter_channels, inter_channels, 1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(drop_rate),
                nn.Conv2d(inter_channels, num_classes, 1),
            )
        else:
            inter_channels = in_channels // 4
            self.classifier = nn.Sequential(
                nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(),
                nn.Dropout(drop_rate),
                nn.Conv2d(inter_channels, num_classes, 1),
            )

    def __str__(self) -> str:
        msg = f"FCN8 - {super().__str__()}"
        return msg

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        result = OrderedDict()
        input_shape = x.shape[-2:]
        features = self.forward_features(x)
        logits = self.classifier(features[-1])

        pool4_shape = features[-2].shape[-2:]
        pool3_shape = features[-3].shape[-2:]
        logits = F.interpolate(logits, size=pool4_shape, mode="bilinear", align_corners=False)
        logits = logits + self.score_pool4(features[-2])
        logits = F.interpolate(logits, size=pool3_shape, mode="bilinear", align_corners=False)
        logits = logits + self.score_pool3(features[-3])
        logits = F.interpolate(logits, size=input_shape, mode="bilinear", align_corners=False)
        result["out"] = logits
        return result
