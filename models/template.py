# -*-coding: utf-8

import argparse

import torch.nn as nn


class ClassificationModel(nn.Module):
    num_classes: int

    @staticmethod
    def add_argparser(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        raise NotImplemented()

    def __str__(self) -> str:
        msg = f"Model: {self.__class__.__name__}\n"
        msg += f"- Num of classes: {self.num_classes}"
        return msg
