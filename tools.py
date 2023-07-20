# -*-coding: utf-8
import argparse

import torch.optim
import torch.nn as nn


def add_argparser_optim(parent_parser: argparse.ArgumentParser, optimizer: str):
    optimizer = str(optimizer).lower()
    parser.add_argument(
        "--lr",
        default=0.001,
        type=float,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        default=0.0,
        type=float,
        help="한 그룹으로 볼 Lesion code list 입력",
    )
    if optimizer == "sgd":
        parser = parent_parser.add_argument_group("SGD")
        parser.add_argument(
            "--momentum",
            default=0.0,
            type=str,
            help="SGD Momentum",
        )

    elif optimizer in ["adam", "adamw"]:
        parser = parent_parser.add_argument_group("ADAM")
        parser.add_argument(
            "--betas",
            nargs="+",
            help="adam betas",
        )
    else:
        raise NotImplementedError(f"{optimizer} is not supported")
    return parent_parser


# def add_argparser_criterion(parent_parser, criterion):
#     criterion = str(criterion).lower()
#     if criterion == "mse":
#         pass
#     elif criterion == "ce":
#         pass
#     else:
#         raise NotImplementedError(f"{criterion} is not supported")
#     return parent_parser


def add_argparser_lr_scheduler(parent_parser, lr_scheduler):
    lr_scheduler = str(lr_scheduler).lower()
    return parent_parser


class Classification:
    epochs: int
    optimizer: torch.optim.Optimizer
    criterion: nn.Module
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler

    def set_optimizer(self, optimizer):
        pass

    def set_criterion(self):
        pass

    def set_lr_scheduler(self):
        raise NotImplemented("LR 스케쥴러를 작성해야됩니다.")

    def train(self, model, train_loader):
        for epoch in range(self.epochs):
            self.train_one_epoch()
            self.validation()

    @staticmethod
    def train_one_epoch(model, train_loader):
        raise NotImplemented("만들어야 됩니다.")

    @staticmethod
    def validation(model, valid_loader):
        raise NotImplemented("만들어야 됩니다.")

    @staticmethod
    def test(model, test_dataset):
        raise NotImplemented("만들어야 됩니다.")

    @staticmethod
    def visualization(model, loader):
        raise NotImplemented("만들어야 됩니다.")


class Segmentation:
    epochs: int
    optimizer: torch.optim.Optimizer
    criterion: nn.Module
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler

    def set_optimizer(self, optimizer):
        pass

    def set_criterion(self):
        pass

    def set_lr_scheduler(self):
        raise NotImplemented("LR 스케쥴러를 작성해야됩니다.")

    def train(self, model, train_loader):
        for epoch in range(self.epochs):
            self.train_one_epoch()
            self.validation()

    @staticmethod
    def train_one_epoch(model, train_loader):
        raise NotImplemented("만들어야 됩니다.")

    @staticmethod
    def validation(model, valid_loader):
        raise NotImplemented("만들어야 됩니다.")

    @staticmethod
    def test(model, test_dataset):
        raise NotImplemented("만들어야 됩니다.")

    @staticmethod
    def visualization(model, loader):
        raise NotImplemented("만들어야 됩니다.")
