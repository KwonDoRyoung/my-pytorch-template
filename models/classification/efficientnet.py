# -*-coding: utf-8 -*-
import torchvision
import torch.nn as nn
from argparse import ArgumentParser
from collections import OrderedDict
from .template import ClassificationModel


model_dict = {
    "efficientnet_b0": torchvision.models.efficientnet_b0,
    "efficientnet_b1": torchvision.models.efficientnet_b1,
    "efficientnet_b2": torchvision.models.efficientnet_b2,
}

weight_dict = {
    "efficientnet_b0": torchvision.models.EfficientNet_B0_Weights.DEFAULT,
    "efficientnet_b1": torchvision.models.EfficientNet_B1_Weights.DEFAULT,
    "efficientnet_b2": torchvision.models.EfficientNet_B2_Weights.DEFAULT,
}


class EfficientNet(ClassificationModel):
    def __init__(
        self,
        model,
        num_classes,
        pretrained=False,
        fixed_features_ext=False,
        pooling_size=None,
    ) -> None:
        super().__init__()
        assert model in model_dict.keys(), f"{model} is not supported"
        self.feature_extractor = None
        self.avgpool = None
        self.classifier = None
        self.features_layer = ["layer2", "layer3", "layer4", "layer6", "layer8"]

        self.num_classes = num_classes

        self._get_basemodel(model, pretrained, pooling_size)

        if fixed_features_ext:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

    @staticmethod
    def add_argparser(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("EfficientNet")
        parser.add_argument("--pretrained", default=False, action="store_true")
        parser.add_argument("--fixed-features-ext", default=False, action="store_true")
        parser.add_argument("--pooling-size", default=None, type=int, nargs="+")
        return parent_parser

    def get_features(self, inputs):
        features = []
        x = inputs
        for layer in self.feature_extractor.keys():
            x = self.feature_extractor[layer](x)
            if layer in self.features_layer:
                features.append(x)
        return features

    def forward(self, inputs):
        x = inputs
        for layer in self.feature_extractor.keys():
            x = self.feature_extractor[layer](x)
        x = self.avgpool(x)
        logits = self.classifier(x)
        return logits

    def _get_basemodel(self, model, pretrained, pooling_size):
        weights = weight_dict[model] if pretrained else None
        print("pretrained weight: ", weights)
        basemodel = model_dict[model](weights=weights, progress=True)

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
                if pooling_size is None:
                    avgpool = module
                else:
                    avgpool = nn.AdaptiveAvgPool2d(pooling_size)
            elif name == "classifier":
                classifier = list(basemodel.classifier.children())
                classifier = [nn.Flatten()] + classifier
                if pooling_size is None:
                    in_features = classifier[-1].in_features
                else:
                    in_features = (
                        classifier[-1].in_features * pooling_size[0] * pooling_size[1]
                    )
                classifier[-1] = nn.Linear(in_features, self.num_classes)  # type: ignore

        self.feature_extractor = nn.ModuleDict(feature_extractor)
        self.avgpool = avgpool
        self.classifier = nn.Sequential(*classifier)

        return basemodel
