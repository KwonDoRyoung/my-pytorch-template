# -*-coding: utf-8 -*-
from typing import Union
from argparse import ArgumentParser
from collections import OrderedDict

from torch import nn

from ..classification import get_cls_model
from .template import SegmentationModelTemplate

fcn32_target_layer = {
    "vgg": {"pool5": "pool5"},
    "resnet": {"layer4": "pool5"},
    "efficientnet": {"layer9": "pool5"},
    "convnext": {"layer8": "pool5"},
}

fcn16_target_layer = {
    "vgg": {"pool4": "pool4", "pool5": "pool5"},
    "resnet": {"layer3": "pool4", "layer4": "pool5"},
    "efficientnet": {"layer6": "pool4", "layer9": "pool5"},
    "convnext": {"layer6": "pool4", "layer8": "pool5"},
}

fcn8_target_layer = {
    "vgg": {"pool3": "pool3", "pool4": "pool4", "pool5": "pool5"},
    "resnet": {"layer2": "pool3", "layer3": "pool4", "layer4": "pool5"},
    "efficientnet": {"layer4": "pool3", "layer6": "pool4", "layer9": "pool5"},
    "convnext": {"layer4": "pool3", "layer6": "pool4", "layer8": "pool5"},
}


class FCN(SegmentationModelTemplate):
    backbone: nn.Module
    score: nn.ModuleDict
    target_layer: Union[str, list]

    def __init__(
        self,
        backbone_name: str,
        backbone_pretrained: bool,
        num_classes: int,
        is_inference: bool,
        criterion_name: str = "",
        **kwargs,
    ) -> None:
        super().__init__(
            num_classes,
            is_inference,
            criterion_name,
            **kwargs,
        )
        self.backbone = get_cls_model(
            model_name=backbone_name,
            pretrained=backbone_pretrained,
            num_classes=0,
            is_inference=is_inference,
            criterion_name="",
        )

    def __str__(self) -> str:
        msg = super().__str__()
        msg += "Backbone Mdoel\n"
        msg += f"{str(self.backbone)}"
        return msg

    @staticmethod
    def add_argparser(
        parent_parser: ArgumentParser, is_inference: bool = True
    ) -> ArgumentParser:
        parser = parent_parser.add_argument_group("FCN")
        parser.add_argument("--backbone-name", type=str, required=True)
        parser.add_argument("--backbone-pretrained", action="store_true")
        if not is_inference:  # For Training
            parser.add_argument("--criterion-name", required=True, help="")
        return parent_parser


class FCN32(FCN):
    def __init__(
        self,
        backbone_name: str,
        backbone_pretrained: bool,
        num_classes: int,
        is_inference: bool,
        criterion_name: str = "",
        **kwargs,
    ) -> None:
        super().__init__(
            backbone_name,
            backbone_pretrained,
            num_classes,
            is_inference,
            criterion_name,
            **kwargs,
        )
        backbone_name = str(self.backbone.__class__.__name__).lower()
        self.target_layer = fcn32_target_layer[backbone_name]

        features_info = self.backbone.features_info
        new_feature_info = OrderedDict()
        for ori_name, f in features_info.items():
            if ori_name in self.target_layer.keys():
                new_feature_info[self.target_layer[ori_name]] = f

        self.score = nn.ModuleDict()
        in_channels = new_feature_info["pool5"]
        if backbone_name == "vgg":
            self.score["pool5"] = nn.Sequential(
                        nn.Conv2d(in_channels, 4096, 1),
                        nn.ReLU(inplace=True),
                        nn.Dropout2d(0.1),
                        nn.Conv2d(4096, 4096, 1),
                        nn.ReLU(inplace=True),
                        nn.Dropout2d(0.1),
                        nn.Conv2d(4096, self.num_classes, 1))
        else:
            inter_channels = in_channels // 4
            self.score["pool5"] = nn.Sequential(
                nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(inter_channels, self.num_classes, 1),)
            
        self.score["last_upsample"] = nn.Upsample(scale_factor=32, mode="bilinear")
            
    def forward(self, inputs):
        h,w = inputs.size()[2:]
        features = self.backbone.get_features(inputs, target_layer=self.target_layer)
        outputs = self.score["pool5"](features["pool5"])
        outputs = self.score["last_upsample"](outputs)
        return outputs

class FCN16(FCN):
    def __init__(
        self,
        backbone_name: str,
        backbone_pretrained: bool,
        num_classes: int,
        is_inference: bool,
        criterion_name: str = "",
        **kwargs,
    ) -> None:
        super().__init__(
            backbone_name,
            backbone_pretrained,
            num_classes,
            is_inference,
            criterion_name,
            **kwargs,
        )
        backbone_name = str(self.backbone.__class__.__name__).lower()
        self.target_layer = fcn16_target_layer[backbone_name]

        features_info = self.backbone.features_info
        new_feature_info = OrderedDict()
        for ori_name, f in features_info.items():
            if ori_name in self.target_layer.keys():
                new_feature_info[self.target_layer[ori_name]] = f

        print(new_feature_info)
        self.score = nn.ModuleDict()
        if backbone_name == "vgg":
            in_channels = new_feature_info["pool5"]
            self.score["pool5"] = nn.Sequential(
                    nn.Conv2d(in_channels, 4096, 1),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(0.1),
                    nn.Conv2d(4096, 4096, 1),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(0.1),
                    nn.Conv2d(4096, self.num_classes, 1),
                    nn.Upsample(scale_factor=2, mode="bilinear"))
            
            in_channels = new_feature_info["pool4"]
            self.score["pool4"] = nn.Conv2d(in_channels, self.num_classes, 1)
        else:
            in_channels = new_feature_info["pool5"]
            inter_channels = in_channels // 4
            self.score["pool5"] = nn.Sequential(
                nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(inter_channels, self.num_classes, 1),
                nn.Upsample(scale_factor=2, mode="bilinear"))
            
            in_channels = new_feature_info["pool4"]
            self.score["pool4"] = nn.Conv2d(in_channels, self.num_classes, 1)
                
        self.score["last_upsample"] = nn.Upsample(scale_factor=16, mode="bilinear")
            
    def forward(self, inputs):
        h,w = inputs.size()[2:]
        features = self.backbone.get_features(inputs, target_layer=self.target_layer)
        pool5_output = self.score["pool5"](features["pool5"])
        pool4_output = self.score["pool4"](features["pool4"])
        outputs = pool4_output + pool5_output
        outputs = self.score["last_upsample"](outputs)
        return outputs


class FCN8(FCN):
    def __init__(
        self,
        backbone_name: str,
        backbone_pretrained: bool,
        num_classes: int,
        is_inference: bool,
        criterion_name: str = "",
        **kwargs,
    ) -> None:
        super().__init__(
            backbone_name,
            backbone_pretrained,
            num_classes,
            is_inference,
            criterion_name,
            **kwargs,
        )
        backbone_name = str(self.backbone.__class__.__name__).lower()
        self.target_layer = fcn8_target_layer[backbone_name]

        features_info = self.backbone.features_info
        new_feature_info = OrderedDict()
        for ori_name, f in features_info.items():
            if ori_name in self.target_layer.keys():
                new_feature_info[self.target_layer[ori_name]] = f

        print(new_feature_info)
        self.score = nn.ModuleDict()
        if backbone_name == "vgg":
            in_channels = new_feature_info["pool5"]
            self.score["pool5"] = nn.Sequential(
                    nn.Conv2d(in_channels, 4096, 1),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(0.1),
                    nn.Conv2d(4096, 4096, 1),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(0.1),
                    nn.Conv2d(4096, self.num_classes, 1),
                    nn.Upsample(scale_factor=2, mode="bilinear"))
            
            in_channels = new_feature_info["pool4"]
            self.score["pool4"] = nn.Conv2d(in_channels, self.num_classes, 1)

            in_channels = new_feature_info["pool3"]
            self.score["pool3"] = nn.Conv2d(in_channels, self.num_classes, 1)
        else:
            in_channels = new_feature_info["pool5"]
            inter_channels = in_channels // 4
            self.score["pool5"] = nn.Sequential(
                nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(inter_channels, self.num_classes, 1),
                nn.Upsample(scale_factor=2, mode="bilinear"))
            
            in_channels = new_feature_info["pool4"]
            self.score["pool4"] = nn.Conv2d(in_channels, self.num_classes, 1)

            in_channels = new_feature_info["pool3"]
            self.score["pool3"] = nn.Conv2d(in_channels, self.num_classes, 1)
        
        self.score["upsample4_5"] = nn.Upsample(scale_factor=2, mode="bilinear")
        self.score["last_upsample"] = nn.Upsample(scale_factor=8, mode="bilinear")
            
    def forward(self, inputs):
        h,w = inputs.size()[2:]
        features = self.backbone.get_features(inputs, target_layer=self.target_layer)
        pool5_output = self.score["pool5"](features["pool5"])
        pool4_output = self.score["pool4"](features["pool4"])
        pool3_output = self.score["pool3"](features["pool3"])
        outputs = pool4_output + pool5_output
        outputs = self.score["upsample4_5"](outputs)
        outputs = pool3_output + outputs
        outputs = self.score["last_upsample"](outputs)
        return outputs
