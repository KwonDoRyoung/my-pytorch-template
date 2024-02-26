# -*-coding: utf-8 -*-
from typing import Dict
from argparse import ArgumentParser

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules import DoubleConv, UpConv2x


def add_argparser_unet_model(parent_parser: ArgumentParser, model_name: str) -> ArgumentParser:
    if model_name == "unet":
        return UNet.add_argparser(parent_parser)
    elif model_name == "unet++":
        return NestedUNet.add_argparser(parent_parser)
    else:
        raise RuntimeError(f"{model_name} is not supported!")


def create_unet_model(
    model_name: str,
    num_classes: int,
    **kwargs,
) -> nn.Module:
    model_name = str(model_name).lower()
    num_classes = 1 if kwargs.get("task") == "binary" else num_classes
    if model_name == "unet":
        return UNet(
            num_classes=num_classes,
            layer_depth=kwargs.get("layer_depth"),
            root_features=kwargs.get("root_features"),
            drop_rate=kwargs.get("drop_rate"),
        )
    elif model_name == "unet++":
        return NestedUNet(
            num_classes=num_classes,
            deep_supervision=kwargs.get("deep_supervision"),
            drop_rate=kwargs.get("drop_rate"),
        )
    else:
        raise RuntimeError(f"{model_name} is not supported!")


class UNet(nn.Module):
    def __init__(self, num_classes: int, layer_depth: int, root_features: int, drop_rate: float = 0.0) -> None:
        super().__init__()
        self.layer_depth = layer_depth
        self.root_features = root_features
        self.drop_rate = drop_rate

        self.contract = torch.nn.ModuleDict()
        out_channels = self.root_features
        in_channels = 3
        for l in range(1, layer_depth):
            self.contract["conv-{}".format(l)] = DoubleConv(in_channels, out_channels, out_channels, drop_rate)
            self.contract["pool-{}".format(l)] = nn.MaxPool2d(2, 2)
            in_channels = out_channels
            out_channels *= 2
        self.contract["conv-{}".format(self.layer_depth)] = DoubleConv(
            in_channels, out_channels, out_channels, drop_rate
        )

        self.expansive = torch.nn.ModuleDict()
        expansive_root_features = self.root_features * (2 ** (self.layer_depth - 1))
        in_channels = expansive_root_features
        out_channels = expansive_root_features // 2
        for l in range(layer_depth - 1, 0, -1):
            self.expansive["up-{}".format(l)] = UpConv2x(in_channels, out_channels)
            self.expansive["conv-{}".format(l)] = DoubleConv(in_channels, out_channels, out_channels, drop_rate)
            in_channels = out_channels
            out_channels //= 2

        self.classifier = nn.Conv2d(root_features, num_classes, 1, 1)

    @staticmethod
    def add_argparser(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("UNet")
        parser.add_argument("--layer-depth", default=4, type=int)
        parser.add_argument("--root-features", default=32, type=int)
        parser.add_argument("--drop-rate", default=0.0, type=float)
        return parent_parser

    def __str__(self) -> str:
        msg = "\nU-Net\n"
        msg += f" - Layer Depth: {self.layer_depth}\n"
        msg += f" - Root Features: {self.root_features}\n"
        msg += f" - Drop out rate: {self.drop_rate:.2f}\n"
        return msg

    @staticmethod
    def center_crop_concat(contract: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        h, w = contract.size(2), contract.size(3)
        t_h, t_w = up.size(2), up.size(3)
        diff_h = (h - t_h) // 2
        diff_w = (w - t_w) // 2
        crop_contract = contract[:, :, diff_h : diff_h + t_h, diff_w : diff_w + t_w]
        return torch.cat([crop_contract, up], dim=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        result = OrderedDict()
        input_shape = x.shape[-2:]
        skip_connection = OrderedDict()
        for l in range(1, self.layer_depth):
            x = self.contract["conv-{}".format(l)](x)
            skip_connection["contract-{}".format(l)] = x
            x = self.contract["pool-{}".format(l)](x)

        x = self.contract["conv-{}".format(self.layer_depth)](x)
        skip_connection["latent-{}".format(self.layer_depth)] = x

        x = skip_connection["latent-{}".format(self.layer_depth)]
        for l in range(self.layer_depth - 1, 0, -1):
            x = self.expansive["up-{}".format(l)](x)
            x = self.center_crop_concat(skip_connection["contract-{}".format(l)], x)
            x = self.expansive["conv-{}".format(l)](x)

        logits = self.classifier(x)
        logits = F.interpolate(logits, size=input_shape, mode="bilinear", align_corners=False)
        result["out"] = logits
        return result


class NestedUNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        drop_rate: float,
        deep_supervision: bool,
    ) -> None:
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.drop_rate = drop_rate
        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv0_0 = DoubleConv(3, nb_filter[0], nb_filter[0], drop_rate)
        self.conv1_0 = DoubleConv(nb_filter[0], nb_filter[1], nb_filter[1], drop_rate)
        self.conv2_0 = DoubleConv(nb_filter[1], nb_filter[2], nb_filter[2], drop_rate)
        self.conv3_0 = DoubleConv(nb_filter[2], nb_filter[3], nb_filter[3], drop_rate)
        self.conv4_0 = DoubleConv(nb_filter[3], nb_filter[4], nb_filter[4], drop_rate)

        self.conv0_1 = DoubleConv(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0], drop_rate)
        self.conv1_1 = DoubleConv(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1], drop_rate)
        self.conv2_1 = DoubleConv(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2], drop_rate)
        self.conv3_1 = DoubleConv(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3], drop_rate)

        self.conv0_2 = DoubleConv(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0], drop_rate)
        self.conv1_2 = DoubleConv(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1], drop_rate)
        self.conv2_2 = DoubleConv(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2], drop_rate)

        self.conv0_3 = DoubleConv(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0], drop_rate)
        self.conv1_3 = DoubleConv(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1], drop_rate)

        self.conv0_4 = DoubleConv(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0], drop_rate)

        if self.deep_supervision:
            self.classifier1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.classifier2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.classifier3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.classifier4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.classifier = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def __str__(self) -> str:
        msg = "\nUNet++\n"
        msg += f"Drop out rate: {self.drop_rate}"
        return msg

    @staticmethod
    def add_argparser(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("Nested UNet")
        parser.add_argument("--deep-supervision", action="store_true")
        parser.add_argument("--drop-rate", type=float, default=0.0)
        return parent_parser

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        result = OrderedDict()
        input_shape = x.shape[-2:]
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            outputs1 = self.classifier1(x0_1)
            outputs2 = self.classifier2(x0_2)
            outputs3 = self.classifier3(x0_3)
            outputs4 = self.classifier4(x0_4)
            logits = (outputs1 + outputs2 + outputs3 + outputs4) / 4
        else:
            logits = self.classifier(x0_4)

        logits = F.interpolate(logits, size=input_shape, mode="bilinear", align_corners=False)
        result["out"] = logits
        return result
