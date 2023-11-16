# -*-coding: utf-8 -*-
import torchvision
import torch.nn as nn
from argparse import ArgumentParser
from collections import OrderedDict
from .template import ClassificationModel


model_dict = {
    "vgg13_bn": torchvision.models.vgg13_bn,
    "vgg16_bn": torchvision.models.vgg16_bn,
    "vgg19_bn": torchvision.models.vgg19_bn,
}

weight_dict = {
    "vgg13_bn": torchvision.models.VGG13_BN_Weights.DEFAULT,
    "vgg16_bn": torchvision.models.VGG16_BN_Weights.DEFAULT,
    "vgg19_bn": torchvision.models.VGG19_BN_Weights.DEFAULT,
}


class VGG(ClassificationModel):
    def __init__(
        self, model, num_classes, pretrained=False, fixed_features_ext=False
    ) -> None:
        super().__init__()

        assert model in model_dict.keys(), f"{model} is not supported"
        self.feature_extractor = None
        self.avgpool = None
        self.classifier = None
        self.features_layer = ["layer1", "layer2", "layer3", "layer4", "layer5"]
        self.num_classes = num_classes

        self._get_basemodel(model, pretrained)

        if fixed_features_ext:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

    @staticmethod
    def add_argparser(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("Vgg")
        parser.add_argument("--pretrained", default=False, action="store_true")
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
                    if isinstance(sub_module, nn.MaxPool2d):
                        feature_extractor[f"pool{nth}"] = sub_module
                        feature_extractor[f"layer{nth}"] = nn.Sequential(
                            *feature_extractor[f"layer{nth}"]
                        )
                        nth += 1
                    else:
                        if f"layer{nth}" not in feature_extractor.keys():
                            feature_extractor[f"layer{nth}"] = []
                        feature_extractor[f"layer{nth}"].append(sub_module)
            elif name == "avgpool":
                avgpool = module
            elif name == "classifier":
                classifier = list(basemodel.classifier.children())
                classifier = [nn.Flatten()] + classifier
                in_features = classifier[-1].in_features
                classifier[-1] = nn.Linear(in_features, self.num_classes)  # type: ignore

        self.feature_extractor = nn.ModuleDict(feature_extractor)
        self.avgpool = avgpool
        self.classifier = nn.Sequential(*classifier)

        return basemodel
