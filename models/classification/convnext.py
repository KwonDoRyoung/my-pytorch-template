# -*-coding: utf-8 -*-
import torchvision
import torch.nn as nn
from argparse import ArgumentParser
from collections import OrderedDict
from .template import ClassificationModel


model_dict = {
    "convnext_tiny": torchvision.models.convnext_tiny,
    "convnext_small": torchvision.models.convnext_small,
    "convnext_base": torchvision.models.convnext_base,
}

weight_dict = {
    "convnext_tiny": torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT,
    "convnext_small": torchvision.models.ConvNeXt_Small_Weights.DEFAULT,
    "convnext_base": torchvision.models.ConvNeXt_Base_Weights.DEFAULT,
}


class ConvNext(ClassificationModel):
    def __init__(
        self, model, num_classes, pretrained=False, fixed_features_ext=False
    ) -> None:
        super().__init__()

        assert model in model_dict.keys(), f"{model} is not supported"
        self.feature_extractor = None
        self.avgpool = None
        self.classifier = None
        self.features_layer = ["layer2", "layer4", "layer6", "layer8"]

        self.num_classes = num_classes

        self._get_basemodel(model, pretrained)

        if fixed_features_ext:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

    @staticmethod
    def add_argparser(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("ConvNext")
        parser.add_argument("--pretrained", action="store_true")
        parser.add_argument("--fixed-features-ext", default=False, action="store_true")
        return parent_parser

    def get_features(self, inputs):
        features = []
        x = inputs
        for layer in self.feature_extractor.keys():
            x = self.feature_extractor[layer](x)
            if layer in self.feature_layer:
                features.append(x)
        return features

    def forward(self, inputs):
        x = inputs
        for layer in self.feature_extractor.keys():
            x = self.feature_extractor[layer](x)
        x = self.avgpool(x)
        logits = self.classifier(x)
        return logits

    def _get_basemodel(self, model_name, pretrained):
        weights = weight_dict[model_name] if pretrained else None
        print("pretrained weight: ", weights)
        basemodel = model_dict[model_name](weights=weights, progress=True)

        feature_extractor = OrderedDict()
        avgpool = None
        classifier = []
        nth = 1

        for name, module in basemodel.named_children():
            if name == "features":
                for _, sub_module in module.named_children():
                    feature_extractor[f"layer{nth}"] = sub_module
                    nth += 1
            elif name == "avgpool":
                avgpool = module
            elif name == "classifier":
                classifier = list(basemodel.classifier.children())
                in_features = classifier[-1].in_features
                classifier[-1] = nn.Linear(in_features, self.num_classes)  # type: ignore

        self.feature_extractor = nn.ModuleDict(feature_extractor)
        self.avgpool = avgpool
        self.classifier = nn.Sequential(*classifier)

        return basemodel
