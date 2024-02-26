# -*-coding: utf-8 -*-
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, drop_rate=0.0):
        super().__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                middle_channels,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(),
            nn.Dropout2d(p=drop_rate),
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(
                middle_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(p=drop_rate),
        )

    def forward(self, x):
        x = self.conv_layer1(x)
        o = self.conv_layer2(x)
        return o


class UpConv2x(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_conv_layer = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.up_conv_layer(x)
        x = self.bn(x)
        o = self.act(x)
        return o


# https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/deeplabv3.py
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: Sequence[int], out_channels: int = 256) -> None:
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)


# https://paperswithcode.com/method/global-local-attention-module
class LocalChannelAttention(nn.Module):
    def __init__(self, in_channels, kernel_size) -> None:
        super().__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=2),
        )
        self.conv1d = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b = x.size(0)
        o = self.gap(x)
        o = self.conv1d(o)
        o = o.reshape(b, -1, 1, 1)
        return o


class LocalSpatialAttention(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.stem = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.inception = nn.ModuleList()
        self.inception.append(nn.Identity())
        self.inception.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=1, padding="same"))
        self.inception.append(nn.Conv2d(out_channels, out_channels, kernel_size=5, dilation=2, padding="same"))
        self.inception.append(nn.Conv2d(out_channels, out_channels, kernel_size=7, dilation=3, padding="same"))

        self.conv = nn.Conv2d(out_channels * 4, 1, kernel_size=1)

    def forward(self, x):
        o = self.stem(x)

        feature_list = []
        for layer in self.inception:
            tmp_o = layer(o)
            feature_list.append(tmp_o)
        o = torch.cat(feature_list, dim=1)

        o = self.conv(o)
        return o


class GlobalChannelAttention(nn.Module):
    def __init__(self, in_channels, kernel_size) -> None:
        super().__init__()
        self.flatten = nn.Flatten(start_dim=2)

        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=2),
        )
        self.query = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.Sigmoid(),
        )
        self.key = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.Sigmoid(),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.size()
        o = self.gap(x)
        query_o = self.query(o)  # size: 3, 32, 1
        query_k = self.key(o)  # size: 3, 32, 1

        correl_o = torch.einsum("bce,bde->bcd", query_o, query_k)

        soft_o = self.softmax(correl_o)
        flat_o = self.flatten(x).permute(0, 2, 1)

        o = torch.matmul(flat_o, soft_o).permute(0, 2, 1).reshape(b, c, h, w)

        return o


class GlobalSpatialAttention(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.qeury = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Flatten(start_dim=2),
        )
        self.key = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Flatten(start_dim=2),
        )
        self.value = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Flatten(start_dim=2),
        )

        self.softmax = nn.Softmax(dim=-1)

        self.conv = nn.Conv2d(out_channels, in_channels, kernel_size=1)

    def forward(self, x):
        b, f, h, w = x.size()
        query_o = self.qeury(x)
        key_o = self.key(x)

        correl_o = torch.einsum("bcd,bef->bdf", query_o, key_o)
        correl_o = self.softmax(correl_o)

        value_o = self.value(x)
        o = torch.matmul(value_o, correl_o)
        o = o.reshape(b, -1, h, w)
        o = self.conv(o)

        return o


class GLAM(nn.Module):
    def __init__(self, in_channels, kernel_size=5) -> None:
        super().__init__()
        self.local_channel_attn = LocalChannelAttention(in_channels, kernel_size=kernel_size)
        self.local_spatial_attn = LocalSpatialAttention(in_channels, in_channels)

        self.global_channel_attn = GlobalChannelAttention(in_channels, kernel_size=kernel_size)
        self.global_spatial_attn = GlobalSpatialAttention(in_channels, in_channels)

        self.weight = nn.Parameter(torch.rand(size=[3]), requires_grad=True)
        self.eps = 1e-5

    def forward(self, x):
        lca_o = self.local_channel_attn(x)
        lsa_o = self.local_spatial_attn(x)
        local_attn_o = lca_o * lsa_o + lca_o

        gca_o = self.global_channel_attn(x)
        gsa_o = self.global_spatial_attn(x)
        global_attn_o = gca_o * gsa_o + gca_o

        o = local_attn_o * self.weight[0] + global_attn_o * self.weight[1] + x * self.weight[2]
        o = o / (self.weight[0] + self.weight[1] + self.weight[2] + self.eps)

        return o
