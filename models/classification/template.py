# -*-coding: utf-8
from fractions import Fraction
from collections import OrderedDict

import torch
import torch.nn as nn

from ..template import BaseModelTemplate


class ClassificationModelTemplate(BaseModelTemplate):
    num_classes: int
    feature_extractor: nn.ModuleDict
    last_pooling: nn.Module
    classifier: nn.Module

    model_name: str
    pretrained: bool

    features_info: OrderedDict
    size_info: OrderedDict

    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        num_classes: int,
        is_inference: bool,
        criterion_name: str = "",
        **kwargs,
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.num_classes = num_classes
        super().__init__(is_inference, criterion_name, **kwargs)

    def _set_model(self):
        raise NotImplementedError("")

    def __str__(self) -> str:
        msg = super().__str__()
        msg += f"  - {self.model_name} | pretrained: {self.pretrained}\n"
        if self.num_classes != 0:
            msg += f"  - Num of classes: {self.num_classes}\n"
        else:
            msg += f"!! Feature extractor model\n"
            msg += f"  - Feature & Size Info(input size = [1,3,512,512])\n"
            for layer_name, f in self.features_info.items():
                h_ratio, w_ratio = self.size_info[layer_name]
                msg += f"    > {layer_name}: [{f}, {h_ratio}, {w_ratio}]\n"
        return msg

    def set_criterion(self, criterion_name, **kwargs):
        if self.num_classes == 0:
            self.criterion = None
        else:
            if criterion_name == "ce":
                self.criterion = nn.CrossEntropyLoss()
            elif criterion_name == "weighted-ce":
                weight = kwargs.get("weight")
                self.criterion = nn.CrossEntropyLoss(weight=weight)
            else:
                raise NotImplementedError(f"{criterion_name} is not supported")

    def _get_features_info(self):
        H = 512
        W = 512
        x = torch.rand(1, 3, H, W)
        feature_info = OrderedDict()
        size_info = OrderedDict()
        for layer_name, module in self.feature_extractor.items():
            x = module(x)
            f, h, w = x.size()[1:]
            feature_info[layer_name] = f
            size_info[layer_name] = [str(Fraction(h, H)), str(Fraction(w, W))]

        return feature_info, size_info

    def get_features(self, inputs, target_layer:dict):
        features = OrderedDict()

        x = inputs
        for layer_name, module in self.feature_extractor.items():
            x = module(x)
            if layer_name in target_layer.keys():
                features[target_layer[layer_name]] = x

        return features
