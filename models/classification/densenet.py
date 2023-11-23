# -*-coding: utf-8 -*-
import torchvision
import torch.nn as nn
from argparse import ArgumentParser
from collections import OrderedDict

from .template import ClassificationModelTemplate


model_dict = {
    "densenet121": torchvision.models.densenet121,
    "densenet161": torchvision.models.densenet161,
    "densenet169": torchvision.models.densenet169,
    "densenet201": torchvision.models.densenet201,
}

weight_dict = {
    "densenet121": torchvision.models.DenseNet121_Weights.DEFAULT,
    "densenet161": torchvision.models.DenseNet161_Weights.DEFAULT,
    "densenet169": torchvision.models.DenseNet169_Weights.DEFAULT,
    "densenet201": torchvision.models.DenseNet201_Weights.DEFAULT,
}


class DenseNet(ClassificationModelTemplate):
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
        parser = parent_parser.add_argument_group("DenseNet")
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
        classifier = []
        nth = 1
        
        stem = []
        for name, module in basemodel.named_children():
            if name == "features":
                for sub_name, sub_module in module.named_children():
                    if sub_name in ["conv0", "norm0", "relu0", "pool0"]:
                        stem.append(sub_module)
                    else:
                        feature_extractor[f"layer{nth}"] = sub_module
                        nth += 1
                feature_extractor["stem"] = nn.Sequential(*stem)
                feature_extractor.move_to_end("stem", False)
            elif name == "classifier":
                if self.num_classes == 0:
                    classifier = [nn.Identity()]
                else:
                    classifier = [module]
                    classifier = [nn.Flatten()] + classifier
                    if last_pooling_output_size is None:
                        in_features = classifier[-1].in_features
                    else:
                        h, w = last_pooling_output_size
                        in_features = (classifier[-1].in_features * h * w)
                    classifier[-1] = nn.Linear(in_features, self.num_classes)

        if last_pooling_output_size is None:
            last_adaptive_avg_pooling = nn.AdaptiveAvgPool2d(1)
        else:
            last_adaptive_avg_pooling = nn.AdaptiveAvgPool2d(last_pooling_output_size)
        
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