# -*-coding: utf-8 -*-
from argparse import ArgumentParser

import os
import time
import logging
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..template import SegmentationModelTemplate


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Dropout2d(p=dropout),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Dropout2d(p=dropout),
        )

    def forward(self, inputs):
        x = self.conv1(inputs)
        out = self.conv2(x)
        return out


class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpConv, self).__init__()
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, inputs):
        return self.up_conv(inputs)


class ContractingPart(nn.Module):
    def __init__(self, layer_depth, root_features, dropout=0.0):
        super(ContractingPart, self).__init__()
        self.layer_depth = layer_depth
        self.root_features = root_features
        self.contract = torch.nn.ModuleDict()
        in_ch = 1
        out_ch = self.root_features
        for l in range(1, layer_depth):
            self.contract["conv-{}".format(l)] = DoubleConv(in_ch, out_ch, dropout)
            self.contract["pool-{}".format(l)] = nn.MaxPool2d(2, 2)
            in_ch = out_ch
            out_ch *= 2

        self.contract["conv-{}".format(self.layer_depth)] = DoubleConv(
            in_ch, out_ch, dropout
        )

    def forward(self, inputs):
        skip_connection = OrderedDict()
        x = inputs
        for l in range(1, self.layer_depth):
            x = self.contract["conv-{}".format(l)](x)
            skip_connection["contract-{}".format(l)] = x
            x = self.contract["pool-{}".format(l)](x)

        x = self.contract["conv-{}".format(self.layer_depth)](x)
        skip_connection["latent-{}".format(self.layer_depth)] = x
        return skip_connection


class ExpansivePart(nn.Module):
    def __init__(self, layer_depth, root_features, dropout=0.0):
        super(ExpansivePart, self).__init__()
        self.layer_depth = layer_depth
        self.root_features = root_features
        self.expansive = torch.nn.ModuleDict()
        in_ch = self.root_features
        out_ch = self.root_features // 2
        for l in range(layer_depth - 1, 0, -1):
            self.expansive["up-{}".format(l)] = UpConv(in_ch, out_ch)
            self.expansive["conv-{}".format(l)] = DoubleConv(in_ch, out_ch, dropout)
            in_ch = out_ch
            out_ch //= 2

    def forward(self, skip_connection):
        x = skip_connection["latent-{}".format(self.layer_depth)]
        for l in range(self.layer_depth - 1, 0, -1):
            x = self.expansive["up-{}".format(l)](x)
            x = self.center_crop_concat(skip_connection["contract-{}".format(l)], x)
            x = self.expansive["conv-{}".format(l)](x)

        return x

    @staticmethod
    def center_crop_concat(contract, up):
        h, w = contract.size(2), contract.size(3)
        t_h, t_w = up.size(2), up.size(3)
        diff_h = (h - t_h) // 2
        diff_w = (w - t_w) // 2
        crop_contract = contract[:, :, diff_h : diff_h + t_h, diff_w : diff_w + t_w]
        return torch.cat([crop_contract, up], dim=1)


class UNet(SegmentationModelTemplate):
    def __init__(
        self,
        num_classes: int,
        layer_depth: int,
        root_features: int,
        dropout: float,
        is_inference: bool,
        criterion_name: str = "",
        **kwargs
    ) -> None:
        super().__init__(num_classes, is_inference, criterion_name, **kwargs)
        self.layer_depth = layer_depth
        expansive_root_features = root_features * (2 ** (self.layer_depth - 1))
        self.contract_part = ContractingPart(self.layer_depth, root_features, dropout)
        self.expansive_part = ExpansivePart(
            self.layer_depth, expansive_root_features, dropout
        )
        self.last_part = nn.Conv2d(root_features, num_classes, 1, 1)

    @staticmethod
    def add_argparser(
        parent_parser: ArgumentParser, is_inference: bool
    ) -> ArgumentParser:
        parser = parent_parser.add_argument_group("UNet")
        parser.add_argument("--layer-depth", default=4, type=int)
        parser.add_argument("--root-features", default=32, type=int)
        parser.add_argument("--dropout", default=0.0, type=float)
        if not is_inference:  # For Training
            parser.add_argument("--criterion-name", required=True, help="")
        return parent_parser

    def forward(self, inputs):
        skip_connection = self.contract_part(inputs)
        x = self.expansive_part(skip_connection)
        out = self.last_part(x)

        return out

    def forward_with_losses(self, inputs, labels) -> dict:
        logits = self.forward(inputs)
        loss = self.criterion(logits, labels)
        return {"logits": logits, "loss": loss}

    def init_weights(self, pretrained=""):
        print("=> init weights from normal distribution")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            print("=> loading pretrained model {}".format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items() if k in model_dict.keys()
            }
            for k, _ in pretrained_dict.items():
                print("=> loading {} pretrained model {}".format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
