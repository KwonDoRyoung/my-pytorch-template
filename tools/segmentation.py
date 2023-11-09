# -*- coding: utf-8 -*-

from argparse import ArgumentParser

import os
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.nn import functional as F

from torchmetrics.classification import Dice

import utils

from .criterion import DiceLoss, CrossEntropy, CrossEntropyWithDiceLoss
from .metrics import BinaryDiceScore

class Segmentation:
    def __init__(
        self,
        model: nn.Module,
        optim_name: str,
        criterion_name: str,
        num_classes: int,
        device,
        distributed: bool,
        output_dir: str,
        print_freq: int,
        **kwargs,
    ) -> None:
        self.distributed = distributed
        self.device = device
        self.print_freq = print_freq
        self.output_dir = output_dir

        self.num_classes = num_classes

        self.model = model

        self.set_criterion(criterion_name, **kwargs)
        self.set_optimizer(optim_name, **kwargs)
        # TODO: set_lr_scheduler()

    @staticmethod
    def add_argparser_optim(parent_parser: ArgumentParser, optim_name: str) -> ArgumentParser:
        parent_parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate")
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
                default=[0.9 ,0.999],
                nargs="+",
                help="adam betas",
            )
        else:
            raise NotImplementedError(f"{optim_name} is not supported")
        return parent_parser

    def set_criterion(self, criterion: str, **kwargs):
        if criterion == "ce":
            self.criterion = CrossEntropy()
        elif criterion == "dice":
            self.criterion = DiceLoss()
        elif criterion == "ce-dice" :
            self.criterion = CrossEntropyWithDiceLoss()
        else:
            raise NotImplementedError(f"{criterion} is not supported")
    
    def set_optimizer(self, optimizer: str, lr: float, **kwargs):
        # Optimizer part
        optimizer = str(optimizer).lower()
        if optimizer == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                momentum=kwargs.get("momentum"),
                lr=lr,
                weight_decay=kwargs.get("weight_decay"),
            )
        elif optimizer == "adam":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                betas=kwargs.get("betas"),
                lr=lr,
                weight_decay=kwargs.get("weight_decay"),
            )
        else:
            return ValueError(f"[{optimizer}] is not supported!")

    def train_one_epoch(self, epoch, train_loader, wandb):
        self.model.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))

        if self.num_classes in [1, 2]: 
            dice = BinaryDiceScore(average=True).to(self.device)
        else:
            dice = None
            raise NotImplementedError("Dice Multiclass Not implemented")

        header = f"Epoch: [{epoch}]"
        step = 0
        for step, (inputs, labels) in enumerate(
            metric_logger.log_every(train_loader, self.print_freq, header), 1
        ):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            logits = self.model(inputs)
            loss = self.criterion(logits, labels)

            self.optimizer.zero_grad()

            loss.backward()
            self.optimizer.step()

            preds = torch.softmax(logits, dim=1).argmax(dim=1)
            dice.update(preds, labels)

            metric_logger.update(
                loss=loss.item(), lr=self.optimizer.param_groups[0]["lr"]
            )
            wandb.log(
                data={"Loss/train": loss.item()}, step=len(train_loader) * epoch + step
            )

        dice_score = dice.compute().item()
        print(f"{header} Dice score: {dice_score:.4f} \n")
        train_loss = metric_logger.meters["loss"].global_avg
        wandb.log(
            data={
                "Dice/train": dice_score,
                "Loss/avgTrain": train_loss,
                "LR": metric_logger.meters["lr"].global_avg,
            },
            step=step * (1 + epoch),
        )
        return train_loss

    def validation(self,global_step, valid_loader, wandb):
        self.model.eval()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = f"Validation"

        if self.num_classes in [1, 2]: 
            dice = BinaryDiceScore(average=True).to(self.device)
        else:
            dice = None
            raise NotImplementedError("Dice Multiclass Not implemented")
        
        with torch.inference_mode():
            for inputs, labels in metric_logger.log_every(
                valid_loader, self.print_freq, header
            ):
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                logits = self.model(inputs)
                loss = self.criterion(logits, labels)
                metric_logger.update(loss=loss.item())

                preds = torch.softmax(logits, dim=1).argmax(dim=1)
                dice.update(preds, labels)

            metric_logger.synchronize_between_processes()

            dice_score = dice.compute().item()
        val_loss = metric_logger.meters["loss"].global_avg
        print(f"{header} Dice score: {dice_score:.4f}")
        wandb.log(
            data={
                "Dice/valid": dice_score,
                "Loss/avgValid": val_loss,
            },
            step=global_step,
        )
        return {"Dice/valid": dice_score}, val_loss

    def train(self,
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

            # TODO: self.lr_scheduler.step()

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

            utils.save_on_master(
                checkpoint,
                os.path.join(self.output_dir, f"fold{ith_fold}", f"ckpt.pth"),
            )
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
                    save_path = os.path.join(
                        self.output_dir, f"fold{ith_fold}", f"best_{metric_name}.pth"
                    )
                    utils.save_on_master(checkpoint, save_path)

    @staticmethod
    def test(model, test_loader, num_classes, classes, device, test_output_dir):
        model.eval()

        if num_classes in [1, 2]: 
            dice = BinaryDiceScore(average=False).to(device)
        else:
            dice = None
            raise NotImplementedError("Dice Multiclass Not implemented")
        file_name_list = []
        with torch.inference_mode():
            for inputs, labels, file_name in test_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(inputs)

                preds = torch.softmax(logits, dim=1).argmax(dim=1)
                dice.update(preds, labels)
                file_name_list.extend(list(file_name))
        
        dice_score = dice.compute().cpu()
        result_data = {
            "file name": file_name_list,
            "Dice score": dice_score,
        }
        print(f"Average Dice score: {dice_score.mean():.4f}")
        df = pd.DataFrame(result_data, columns=result_data.keys())
        print(df)
        with open(os.path.join(test_output_dir, "result.tex"), "w") as f:
            f.write(df.to_latex(index=False))
        with open(os.path.join(test_output_dir, "result.md"), "w") as f:
            f.write(df.to_markdown(index=False))

    @staticmethod
    def visualization(model, input_image):
        model.eval()
        logits = model(input_image.unsqueeze(0))
        pred_prob = torch.softmax(logits, dim=1).argmax(dim=1)
        return pred_prob

        