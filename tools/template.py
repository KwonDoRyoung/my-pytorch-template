# -*- coding: utf-8 -*-
from abc import abstractmethod
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.optim as optim


class SimpleToolTemplate:    
    distributed: bool
    output_dir: str
    print_freq: int

    model: nn.Module
    optimizer: optim.Optimizer
    device: torch.device

    def __init__(self, model, device, distributed, output_dir, print_freq) -> None:
        self.device = device
        self.print_freq = print_freq
        self.output_dir = output_dir

        self.distributed = distributed

        self.model = model

    # TODO: __str__ 을 추가 해야될까?? 생각해보자

    def set_optimizer(self, optim_name: str, lr: float, **kwargs):
        # Optimizer part
        optim_name = str(optim_name).lower()
        if optim_name == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                momentum=kwargs.get("momentum"),
                lr=lr,
                weight_decay=kwargs.get("weight_decay"),
            )
        elif optim_name == "adam":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                betas=kwargs.get("betas"),
                lr=lr,
                weight_decay=kwargs.get("weight_decay"),
            )
        else:
            return ValueError(f"[{optim_name}] is not supported!")

    @staticmethod
    def add_argparser_optim(
        parent_parser: ArgumentParser, optim_name: str
    ) -> ArgumentParser:
        parent_parser.add_argument(
            "--lr", default=1e-4, type=float, help="Learning rate"
        )
        parent_parser.add_argument(
            "--weight-decay",
            default=0.0,
            type=float,
        )

        optim_name = str(optim_name).lower()
        if optim_name == "sgd":
            parser = parent_parser.add_argument_group("SGD")
            parser.add_argument(
                "--momentum",
                default=0.0,
                type=float,
                help="SGD Momentum",
            )

        elif optim_name in ["adam", "adamw"]:
            parser = parent_parser.add_argument_group("ADAM")
            parser.add_argument(
                "--betas",
                default=[0.9, 0.999],
                nargs="+",
                help="adam betas",
            )
        else:
            raise NotImplementedError(f"{optim_name} is not supported")
        return parent_parser

    @abstractmethod
    def train_one_epoch(self, epoch, train_loader, wandb):
        raise NotImplementedError("train_one_epoch")

    @abstractmethod
    def validation(self, global_step, valid_loader, wandb):
        raise NotImplementedError("validation")

    @abstractmethod
    def train(
        self,
        ith_fold,
        wandb,
        start_epoch,
        epochs,
        train_loader,
        valid_loader,
        train_sampler,
        args,
    ):
        raise NotImplementedError("train")

    @abstractmethod
    def test(model, test_loader, num_classes, classes, device, test_output_dir):
        raise NotImplementedError("test")

    @abstractmethod
    def inference(model, image):
        raise NotImplementedError("inference")
    
    @abstractmethod
    def cam_visualization(model,image, target_layer):
        raise NotImplementedError("cam_visualization")
