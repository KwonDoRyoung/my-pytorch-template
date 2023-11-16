# -*-coding: utf-8
from abc import abstractmethod, abstractstaticmethod
import argparse

import torch.nn as nn

from .criterion import CrossEntropy, DiceLoss, CrossEntropyWithDiceLoss

class BaseModelTemplate(nn.Module):
    def __init__(self, is_inference:bool, criterion_name:str="", **kwargs) -> None:
        super().__init__()
        assert is_inference is not None, "Please set the True or False"
        self.is_inference = is_inference
        if not self.is_inference:
            self.set_criterion(criterion_name, **kwargs)
        else:
            self.criterion = None

    def __str__(self) -> str:
        msg = f"• Model: {self.__class__.__name__} - "
        msg += f"{'Inference' if self.is_inference else 'Training'} Mode\n"
        return msg

    @abstractstaticmethod
    def add_argparser(
        parent_parser: argparse.ArgumentParser, is_inference:bool,
    ) -> argparse.ArgumentParser:
        raise NotImplementedError("add_argparser")
    
    @abstractmethod
    def set_criterion(self, criterion_name, **kwargs):
        raise NotImplementedError("set_criterion")
    
    @abstractmethod
    def forward_with_losses(self, inputs, labels) -> dict:
        raise NotImplementedError("")


class ClassificationModelTemplate(BaseModelTemplate):
    num_classes: int
    def __init__(self, num_classes:int, is_inference:bool, criterion_name:str="", **kwargs) -> None:
        assert num_classes > 0, f"num_classes is larger than 0"
        self.num_classes = num_classes
        super().__init__(is_inference, criterion_name, **kwargs)

    def __str__(self) -> str:
        msg = super().__str__()
        msg += f"  - Num of classes: {self.num_classes}"
        return msg

    def set_criterion(self, criterion_name, **kwargs):
        if criterion_name == "ce":
            self.criterion = nn.CrossEntropyLoss()
        elif criterion_name == "weighted-ce":
            weight = kwargs.get("weight")
            self.criterion = nn.CrossEntropyLoss(weight=weight)
        else:
            raise NotImplementedError(f"{criterion_name} is not supported")
        
class SegmentationModelTemplate(BaseModelTemplate):
    num_classes: int
    def __init__(self, num_classes:int, is_inference: bool, criterion_name: str = "", **kwargs) -> None:
        super().__init__(is_inference, criterion_name, **kwargs)
        self.num_classes = num_classes
    
    def __str__(self) -> str:
        msg = super().__str__()
        msg += f"  - Num of classes: {self.num_classes}"
        return msg

    def set_criterion(self, criterion_name, **kwargs):
        if criterion_name == "ce":
            self.criterion = CrossEntropy()
        elif criterion_name == "dice":
            self.criterion = DiceLoss()
        elif criterion_name == "ce-dice":
            self.criterion = CrossEntropyWithDiceLoss()
        else:
            raise NotImplementedError(f"{criterion_name} is not supported")
        
class MultitaskModelTemplate(BaseModelTemplate):
    num_classes: int
    def __init__(self, num_classes) -> None:
        self.num_classes = num_classes
        super().__init__()
    
    def __str__(self) -> str:
        msg = super().__str__()
        msg += f"  - Num of classes: {self.num_classes}"
        return msg

    def set_criterion(self, criterion_name, **kwargs):
        #TODO: update this
        pass