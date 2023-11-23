# -*-coding: utf-8 -*-
import torch.nn as nn
import torchvision
from argparse import ArgumentParser
from collections import OrderedDict

from .template import ClassificationModelTemplate

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


class VGG(ClassificationModelTemplate):
    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        num_classes: int,
        is_inference: bool,
        criterion_name: str = "",
        last_pooling_output_size: list = None,
        **kwargs,
    ) -> None:
        assert model_name in model_dict.keys(), f"{model_name} is not supported"
        super().__init__(
            model_name,
            pretrained,
            num_classes,
            is_inference,
            criterion_name,
            **kwargs,
        )
        self._set_model(last_pooling_output_size)
        self.features_info, self.size_info = self._get_features_info()

    @staticmethod
    def add_argparser(
        parent_parser: ArgumentParser, is_inference: bool
    ) -> ArgumentParser:
        parser = parent_parser.add_argument_group("VGG")
        parser.add_argument("--pretrained", default=False, action="store_true")
        parser.add_argument("--last-pooling-output-size", default=None, type=int, nargs="+")
        if not is_inference:  # For Training
            parser.add_argument("--criterion-name", required=True, help="")
        return parent_parser

    def _set_model(self, last_pooling_output_size):
        weights = weight_dict[self.model_name] if self.pretrained else None
        print("pretrained weight: ", weights)
        basemodel = model_dict[self.model_name](weights=weights, progress=True)

        feature_extractor = OrderedDict()
        last_adaptive_avg_pooling = None
        for_classifier = None
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
                if last_pooling_output_size is None:
                    last_adaptive_avg_pooling = module
                else:
                    h, w = last_pooling_output_size
                    last_adaptive_avg_pooling = nn.AdaptiveAvgPool2d(last_pooling_output_size)
                    for_classifier = h * w * 512
            elif name == "classifier":
                if self.num_classes == 0:
                    classifier = [nn.Identity()]
                else:
                    classifier = list(module)
                    if for_classifier is not None:
                        out_features = classifier[0].out_features
                        classifier[0] = nn.Linear(for_classifier, out_features)
                    in_features = classifier[-1].in_features
                    classifier[-1] = nn.Linear(in_features, self.num_classes)
                    classifier = [nn.Flatten()] + classifier

        self.feature_extractor = nn.ModuleDict(feature_extractor)
        self.last_pooling = last_adaptive_avg_pooling
        self.classifier = nn.Sequential(*classifier)

    def forward(self, inputs):
        x = inputs
        for layer_name, module in self.feature_extractor.items():
            x = module(x)
        x = self.last_pooling(x)
        logits = self.classifier(x)
        return logits
