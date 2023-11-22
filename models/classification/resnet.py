# -*-coding: utf-8 -*-
import torchvision
import torch.nn as nn
from argparse import ArgumentParser
from collections import OrderedDict

from .template import ClassificationModelTemplate


model_dict = {
    "resnet18": torchvision.models.resnet18,
    "resnet34": torchvision.models.resnet34,
    "resnet50": torchvision.models.resnet50,
    "resnet101": torchvision.models.resnet101,
    "resnext50_32x4d": torchvision.models.resnext50_32x4d,
    "wide_resnet50_2": torchvision.models.wide_resnet50_2,
}

weight_dict = {
    "resnet18": torchvision.models.ResNet18_Weights.DEFAULT,
    "resnet34": torchvision.models.ResNet34_Weights.DEFAULT,
    "resnet50": torchvision.models.ResNet50_Weights.DEFAULT,
    "resnet101": torchvision.models.ResNet101_Weights.DEFAULT,
    "resnext50_32x4d": torchvision.models.ResNeXt50_32X4D_Weights.DEFAULT,
    "wide_resnet50_2": torchvision.models.Wide_ResNet50_2_Weights.DEFAULT,
}


class ResNet(ClassificationModelTemplate):
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
    def add_argparser(parent_parser: ArgumentParser, is_inference:bool) -> ArgumentParser:
        parser = parent_parser.add_argument_group("ResNet|ResNeXt")
        parser.add_argument("--pretrained", default=False, action="store_true")
        parser.add_argument("--last-pooling-size", default=None, type=int, nargs="+")
        if not is_inference:  # For Training
            parser.add_argument("--criterion-name", required=True, help="")
        return parent_parser

    def _set_model(self, last_pooling_output_size):
        weights = weight_dict[self.model_name] if self.pretrained else None
        print("pretrained weight: ", weights)
        basemodel = model_dict[self.model_name](weights=weights, progress=True)

        feature_extractor = OrderedDict()
        last_adaptive_avg_pooling = None
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
                if last_pooling_output_size is None:
                    last_adaptive_avg_pooling = module
                else:
                    last_adaptive_avg_pooling = nn.AdaptiveAvgPool2d(last_pooling_output_size)
            elif name == "fc":
                if self.num_classes == 0:
                    classifier = [nn.Identity()]
                else:
                    if last_pooling_output_size is None:
                        in_features = module.in_features
                    else:
                        h, w = last_pooling_output_size
                        in_features = module.in_features * h * w
                    classifier.append(nn.Flatten())
                    classifier.append(nn.Linear(in_features, self.num_classes))

        self.feature_extractor = nn.ModuleDict(feature_extractor)
        self.last_pooling = last_adaptive_avg_pooling
        self.classifier = nn.Sequential(*classifier)

    def forward(self, inputs):
        x = inputs
        for layer in self.feature_extractor.keys():
            x = self.feature_extractor[layer](x)
        x = self.last_pooling(x)
        logits = self.classifier(x)
        return logits