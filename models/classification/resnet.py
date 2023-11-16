# -*-coding: utf-8 -*-
import torchvision
import torch.nn as nn
from argparse import ArgumentParser
from collections import OrderedDict
from .template import ClassificationModel


model_dict = {
    "resnet18": torchvision.models.resnet18,
    "resnet34": torchvision.models.resnet34,
    "resnet50": torchvision.models.resnet50,
    "resnext50_32x4d": torchvision.models.resnext50_32x4d,
    "wide_resnet50_2": torchvision.models.wide_resnet50_2,
}

weight_dict = {
    "resnet18": torchvision.models.ResNet18_Weights.DEFAULT,
    "resnet34": torchvision.models.ResNet34_Weights.DEFAULT,
    "resnet50": torchvision.models.ResNet50_Weights.DEFAULT,
    "resnext50_32x4d": torchvision.models.ResNeXt50_32X4D_Weights.DEFAULT,
    "wide_resnet50_2": torchvision.models.Wide_ResNet50_2_Weights.DEFAULT,
}


class ResNet(ClassificationModel):
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
        self.features_layer = ["stem", "layer1", "layer2", "layer3", "layer4"]

        self.num_classes = num_classes

        self._get_basemodel(model, pretrained, pooling_size)

        if fixed_features_ext:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

    @staticmethod
    def add_argparser(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("ResNet|ResNeXt")
        parser.add_argument("--pretrained", default=False, action="store_true")
        parser.add_argument("--fixed-features-ext", default=False, action="store_true")
        parser.add_argument("--pooling-size", default=None, type=int, nargs="+")
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

    def _get_basemodel(self, model, pretrained, pooling_size):
        weights = weight_dict[model] if pretrained else None
        print("pretrained weight: ", weights)
        basemodel = model_dict[model](weights=weights, progress=True)

        feature_extractor = OrderedDict()
        avgpool = None
        classifier = []
        nth = 1

        for name, module in basemodel.named_children():
            if name in ["conv1", "bn1", "relu"]:
                if "stem" not in feature_extractor.keys():
                    feature_extractor["stem"] = []
                    # TODO: conv1 kernel 사이즈 변경 관련 & 여부 argument 추가
                feature_extractor["stem"].append(module)

            elif name == "maxpool":
                feature_extractor["stemdown"] = module
                feature_extractor["stem"] = nn.Sequential(*feature_extractor["stem"])

            elif str(name).startswith("layer"):
                feature_extractor[f"layer{nth}"] = module
                nth += 1
            elif name == "avgpool":
                if pooling_size is None:
                    avgpool = module
                else:
                    avgpool = nn.AdaptiveAvgPool2d(pooling_size)
            elif name == "fc":
                if pooling_size is None:
                    in_features = module.in_features
                else:
                    in_features = module.in_features * pooling_size[0] * pooling_size[1]
                classifier.append(nn.Flatten())
                classifier.append(nn.Linear(in_features, self.num_classes))

        self.feature_extractor = nn.ModuleDict(feature_extractor)
        self.avgpool = avgpool
        self.classifier = nn.Sequential(*classifier)

        return basemodel
