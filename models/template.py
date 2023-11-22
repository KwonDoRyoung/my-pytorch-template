# -*-coding: utf-8
from abc import abstractmethod, abstractstaticmethod
from argparse import ArgumentParser

import torch.nn as nn

class BaseModelTemplate(nn.Module):
    def __init__(self, is_inference: bool, criterion_name: str = "", **kwargs) -> None:
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
        parent_parser: ArgumentParser, is_inference: bool
    ) -> ArgumentParser:
        raise NotImplementedError("add_argparser")

    @abstractmethod
    def set_criterion(self, criterion_name, **kwargs):
        raise NotImplementedError("set_criterion")

    def forward_with_losses(self, inputs, labels) -> dict:
        logits = self.forward(inputs)
        loss = self.criterion(logits, labels)
        return {"logits": logits, "loss": loss}