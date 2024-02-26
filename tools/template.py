# -*- coding: utf-8 -*-
from abc import abstractmethod
from argparse import ArgumentParser

import os

import torch
import torch.nn as nn
import torch.optim as optim

import utils


class SimpleToolTemplate:
    distributed: bool
    output_dir: str
    print_freq: int

    model: nn.Module
    optimizer: optim.Optimizer
    criterion: nn.Module
    device: torch.device

    def __init__(self, model, device, distributed, output_dir, print_freq) -> None:
        self.device = device
        self.print_freq = print_freq
        self.output_dir = output_dir

        self.distributed = distributed

        self.model = model

    def set_criterion(self, criterion_name: str, **kwargs):
        raise NotImplementedError("set_criterion")
    
    def add_argparser_criterion(parent_parser: ArgumentParser) -> ArgumentParser:
        raise NotImplementedError("add_argparser_criterion")

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
    def add_argparser_optim(parent_parser: ArgumentParser, optim_name: str) -> ArgumentParser:
        parent_parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate")
        parent_parser.add_argument(
            "--weight-decay",
            default=0.0,
            type=float,
        )
        parent_parser.add_argument("--use-orthogonal-l2-reg", action="store_true",help="")

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
    def test(model, test_loader, num_classes, classes, device, test_output_dir):
        raise NotImplementedError("test")

    @abstractmethod
    def inference(model, image):
        raise NotImplementedError("inference")

    @abstractmethod
    def visualization(model: nn.Module, test_dataset, device, test_output_dir):
        raise NotImplementedError("visualization")

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
        best_metric_dict = {}
        fit_val_loss = 99999
        for epoch in range(start_epoch, epochs):
            print("-" * 15, f" {ith_fold} Fold ", "-" * 15)
            if self.distributed:
                train_sampler.set_epoch(epoch)
            train_loss = self.train_one_epoch(epoch, train_loader, wandb)

            global_step = (epoch + 1) * len(train_loader)

            metric_dict, valid_loss = self.validation(global_step, valid_loader, wandb)

            if self.distributed:
                model_without_ddp = self.model.module
            else:
                model_without_ddp = self.model

            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": valid_loss,
                "args": args,
                "ith_fold": ith_fold,
            }

            if best_metric_dict == {}:
                best_metric_dict = metric_dict.copy()

            if fit_val_loss > valid_loss:
                fit_val_loss = valid_loss
                save_path = os.path.join(self.output_dir, f"fold{ith_fold}", f"fit_val_loss.pth")
                utils.save_on_master(checkpoint, save_path)

            for metric_name, metric_value in best_metric_dict.items():
                if metric_value <= metric_dict[metric_name]:
                    best_metric_dict[metric_name] = metric_dict[metric_name]
                    metric_name = metric_name.replace("/", "_")
                    save_path = os.path.join(self.output_dir, f"fold{ith_fold}", f"best_{metric_name}.pth")
                    utils.save_on_master(checkpoint, save_path)

            if epoch >= epochs // 2 and epoch % 5 == 0:
                utils.save_on_master(
                    checkpoint,
                    os.path.join(self.output_dir, f"fold{ith_fold}", f"ckpt_{epoch}.pth"),
                )
            else:
                utils.save_on_master(
                    checkpoint,
                    os.path.join(self.output_dir, f"fold{ith_fold}", f"ckpt_last.pth"),
                )
