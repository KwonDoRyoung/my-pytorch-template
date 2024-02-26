# -*- coding: utf-8 -*-
from argparse import ArgumentParser

import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from torchvision.io import write_jpeg
from torchvision.utils import draw_segmentation_masks

import utils

from .template import SimpleToolTemplate
from .metrics import BinaryDiceScore, MulticlassDiceScore
from .losses import (
    BinaryCE,
    BinaryDiceLoss,
    MulticlassCE,
    MulticlassDiceLoss,
    BinaryCEWithDiceLoss,
    MultiClassCEWithDiceLoss,
)


def add_argparser_seg_tool(task: str, parent_parser: ArgumentParser, optim_name: str) -> ArgumentParser:
    if task == "binary":
        parent_parser = BinarySegmentation.add_argparser_optim(parent_parser, optim_name)
        parent_parser = BinarySegmentation.add_argparser_criterion(parent_parser)
        return parent_parser
    elif task == "multiclass":
        parent_parser = MultiSegmentation.add_argparser_optim(parent_parser, optim_name)
        parent_parser = MultiSegmentation.add_argparser_criterion(parent_parser)
        return parent_parser
    else:
        raise NotImplementedError(f"'{task}' tool is not supported.")


def create_seg_tool(
    task: str,
    model: nn.Module,
    num_classes: int,
    optim_name: str,
    criterion_name: str,
    device: torch.device,
    distributed: bool,
    output_dir: str,
    print_freq: int,
    **kwargs,
):
    if task == "binary":
        return BinarySegmentation(
            model,
            optim_name,
            criterion_name,
            device,
            distributed,
            output_dir,
            print_freq,
            **kwargs,
        )
    elif task == "multiclass":
        return MultiSegmentation(
            model,
            num_classes,
            optim_name,
            criterion_name,
            device,
            distributed,
            output_dir,
            print_freq,
            **kwargs,
        )
    else:
        raise NotImplementedError(f"'{task}' tool is not supported.")


class BinarySegmentation(SimpleToolTemplate):
    def __init__(
        self,
        model: nn.Module,
        optim_name: str,
        criterion_name: str,
        device: torch.device,
        distributed: bool,
        output_dir: str,
        print_freq: int,
        **kwargs,
    ) -> None:
        super().__init__(model, device, distributed, output_dir, print_freq)
        self.model = model
        self.num_classes = 2

        self.set_optimizer(optim_name, **kwargs)
        self.set_criterion(criterion_name, **kwargs)

    def set_criterion(self, criterion_name: str, **kwargs):
        if criterion_name == "ce":
            self.criteroin = BinaryCE()
        elif criterion_name == "dice":
            self.criteroin = BinaryDiceLoss()
        elif criterion_name == "ce+dice":
            self.criterion = BinaryCEWithDiceLoss()
        else:
            raise NotImplementedError(f"{criterion_name} is not supported.")

    def add_argparser_criterion(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("Criterion")
        parser.add_argument("--criterion-name", required=True, type=str, help="")
        return parent_parser

    def train_one_epoch(self, epoch, train_loader, wandb):
        self.model.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))

        dice_metrics = BinaryDiceScore()

        header = f"Epoch: [{epoch}]"
        step = 0
        for step, (inputs, targets, _) in enumerate(metric_logger.log_every(train_loader, self.print_freq, header), 1):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)

            loss = self.criterion(outputs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            preds = (torch.sigmoid(outputs["out"].squeeze(dim=1)) > 0.5).to(torch.long)
            dice_metrics.update(preds, targets)

            metric_logger.update(loss=loss.item(), lr=self.optimizer.param_groups[0]["lr"])
            if utils.is_main_process():
                wandb.log(data={"Loss/train": loss.item()}, step=len(train_loader) * epoch + step)

        dice_metrics.reduce_from_all_processes()
        print(dice_metrics)
        dice_score = dice_metrics.compute().item()
        train_loss = metric_logger.meters["loss"].global_avg

        if utils.is_main_process():
            wandb.log(
                data={
                    "Dice/train": dice_score,
                    "Loss/avgTrain": train_loss,
                    "LR": metric_logger.meters["lr"].global_avg,
                },
                step=step * (1 + epoch),
            )
        return train_loss

    def validation(self, global_step, valid_loader, wandb):
        self.model.eval()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = f"Validation"

        dice_metrics = BinaryDiceScore()

        with torch.inference_mode():
            for inputs, targets, _ in metric_logger.log_every(valid_loader, self.print_freq, header):
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                preds = (torch.sigmoid(outputs["out"].squeeze(dim=1)) > 0.5).to(torch.long)
                dice_metrics.update(preds, targets)
                metric_logger.update(loss=loss.item())

            metric_logger.synchronize_between_processes()

        dice_metrics.reduce_from_all_processes()
        print(dice_metrics)
        dice_score = dice_metrics.compute().item()
        val_loss = metric_logger.meters["loss"].global_avg

        if utils.is_main_process():
            wandb.log(
                data={
                    "Dice/valid": dice_score,
                    "Loss/avgValid": val_loss,
                },
                step=global_step,
            )
        return {"Dice/valid": dice_score}, val_loss

    @staticmethod
    def test(model, test_dataset, device, batch_size=8):
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        dice_metrics = BinaryDiceScore()

        model.eval()
        with torch.inference_mode():
            for inputs, targets, _ in test_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                outputs = model(inputs)
                preds = (torch.sigmoid(outputs["out"].squeeze(dim=1)) > 0.5).to(torch.long)
                dice_metrics.update(preds, targets)

        print(dice_metrics)
        dice_score = dice_metrics.compute().item()
        result_data = {
            "mean Dice": dice_score,
        }

        return result_data

    @staticmethod
    def visualization(model: nn.Module, test_dataset, device, test_output_dir, batch_size=4):
        colors = test_dataset.color_space

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        dice_metrics = BinaryDiceScore()

        model.eval()
        with torch.inference_mode():
            for inputs, targets, filenames in test_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                outputs = model(inputs)
                preds = (torch.sigmoid(outputs["out"].squeeze(dim=1)) > 0.5).to(torch.long)

                dice_scores = dice_metrics(preds, targets)

                for b_idx in range(batch_size):
                    img = inputs.cpu()[b_idx]
                    target_img = targets.cpu()[b_idx]
                    pred_img = preds.cpu()[b_idx]
                    filename = filenames[b_idx]
                    dice_score = dice_scores[b_idx]

                    img = (img - img.max()) / (img.max() - img.min()) * 255.0
                    img = img.type(torch.uint8)

                    one_hot_pred = F.one_hot(pred_img, num_classes=2).type(torch.bool).permute(2, 0, 1)
                    one_hot_target = F.one_hot(target_img, num_classes=2).type(torch.bool).permute(2, 0, 1)

                    img_with_target = draw_segmentation_masks(img, one_hot_target, alpha=0.6, colors=colors)
                    img_with_pred = draw_segmentation_masks(img, one_hot_pred, alpha=0.6, colors=colors)

                    padding = (torch.ones([3, img.shape[1], 10]) * 255).to(img_with_target.dtype)
                    total_img = torch.cat([img, padding, img_with_target, padding, img_with_pred], dim=-1)
                    plt.figure(figsize=(18,10))
                    plt.imshow(total_img.permute(1, 2, 0))
                    plt.title(f"Image | Target | Prediction\nDice score: {dice_score:.4f}")
                    plt.tight_layout()
                    plt.axis("off")
                    plt.savefig(os.path.join(test_output_dir, f"{filename}.jpeg"), bbox_inches="tight", pad_inches=0)
                    plt.close()


class MultiSegmentation(SimpleToolTemplate):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        optim_name: str,
        criterion_name: str,
        device: torch.device,
        distributed: bool,
        output_dir: str,
        print_freq: int,
        **kwargs,
    ) -> None:
        super().__init__(model, device, distributed, output_dir, print_freq)
        self.model = model
        self.num_classes = num_classes

        self.set_optimizer(optim_name, **kwargs)
        self.set_criterion(criterion_name, **kwargs)

    def set_criterion(self, criterion_name: str, **kwargs):
        if criterion_name == "ce":
            self.criteroin = MulticlassCE()
        elif criterion_name == "dice":
            self.criteroin = MulticlassDiceLoss()
        elif criterion_name == "ce+dice":
            self.criterion = MultiClassCEWithDiceLoss()
        else:
            raise NotImplementedError(f"{criterion_name} is not supported.")

    def add_argparser_criterion(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("Criterion")
        parser.add_argument("--criterion-name", required=True, type=str, help="")
        return parent_parser

    def train_one_epoch(self, epoch, train_loader, wandb):
        self.model.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))

        dice_metrics = MulticlassDiceScore(num_classes=self.num_classes)

        header = f"Epoch: [{epoch}]"
        step = 0
        for step, (inputs, targets, _) in enumerate(metric_logger.log_every(train_loader, self.print_freq, header), 1):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)

            loss = self.criterion(outputs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            preds = torch.softmax(outputs["out"], dim=1).argmax(dim=1).to(torch.long)
            dice_metrics.update(preds, targets)

            metric_logger.update(loss=loss.item(), lr=self.optimizer.param_groups[0]["lr"])
            if utils.is_main_process():
                wandb.log(data={"Loss/train": loss.item()}, step=len(train_loader) * epoch + step)

        dice_metrics.reduce_from_all_processes()
        print(dice_metrics)
        dice_score = dice_metrics.compute().item()
        train_loss = metric_logger.meters["loss"].global_avg

        if utils.is_main_process():
            wandb.log(
                data={
                    "Dice/train": dice_score,
                    "Loss/avgTrain": train_loss,
                    "LR": metric_logger.meters["lr"].global_avg,
                },
                step=step * (1 + epoch),
            )
        return train_loss

    def validation(self, global_step, valid_loader, wandb):
        self.model.eval()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = f"Validation"

        dice_metrics = MulticlassDiceScore(num_classes=self.num_classes)

        with torch.inference_mode():
            for inputs, targets, _ in metric_logger.log_every(valid_loader, self.print_freq, header):
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                preds = torch.softmax(outputs["out"], dim=1).argmax(dim=1).to(torch.long)
                dice_metrics.update(preds, targets)
                metric_logger.update(loss=loss.item())

            metric_logger.synchronize_between_processes()

        dice_metrics.reduce_from_all_processes()
        print(dice_metrics)
        dice_score = dice_metrics.compute().item()
        val_loss = metric_logger.meters["loss"].global_avg

        if utils.is_main_process():
            wandb.log(
                data={
                    "Dice/valid": dice_score,
                    "Loss/avgValid": val_loss,
                },
                step=global_step,
            )
        return {"Dice/valid": dice_score}, val_loss

    @staticmethod
    def test(model, test_dataset, device, batch_size=8):
        num_classes = test_dataset.num_classes
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        dice_metrics = MulticlassDiceScore(num_classes=num_classes)

        model.eval()
        with torch.inference_mode():
            for inputs, targets, _ in test_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                outputs = model(inputs)
                preds = torch.softmax(outputs["out"], dim=1).argmax(dim=1).to(torch.long)
                dice_metrics.update(preds, targets)

        print(dice_metrics)
        dice_score = dice_metrics.compute().item()
        result_data = {
            "mean Dice": dice_score,
        }

        return result_data

    @staticmethod
    def visualization(model: nn.Module, test_dataset, device, test_output_dir, batch_size=4):
        num_classes = test_dataset.num_classes
        colors = test_dataset.color_space

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        dice_metrics = MulticlassDiceScore(num_classes=num_classes)

        model.eval()
        with torch.inference_mode():
            for inputs, targets, filenames in test_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                outputs = model(inputs)
                preds = (torch.sigmoid(outputs["out"].squeeze(dim=1)) > 0.5).to(torch.long)

                dice_scores = dice_metrics(preds, targets)

                for b_idx in range(batch_size):
                    img = inputs.cpu()[b_idx]
                    target_img = targets.cpu()[b_idx]
                    pred_img = preds.cpu()[b_idx]
                    filename = filenames[b_idx]
                    dice_score = dice_scores[b_idx]

                    img = (img - img.max()) / (img.max() - img.min()) * 255.0
                    img = img.type(torch.uint8)

                    one_hot_pred = F.one_hot(pred_img, num_classes=2).type(torch.bool).permute(2, 0, 1)
                    one_hot_target = F.one_hot(target_img, num_classes=2).type(torch.bool).permute(2, 0, 1)

                    img_with_target = draw_segmentation_masks(img, one_hot_target, alpha=0.6, colors=colors)
                    img_with_pred = draw_segmentation_masks(img, one_hot_pred, alpha=0.6, colors=colors)

                    padding = (torch.ones([3, img.shape[1], 10]) * 255).to(img_with_target.dtype)
                    total_img = torch.cat([img, padding, img_with_target, padding, img_with_pred], dim=-1)
                    plt.figure(figsize=(18,10))
                    plt.imshow(total_img.permute(1, 2, 0))
                    msg = ""
                    for c_idx in range(1, num_classes):
                        msg += f"{c_idx}: {dice_score[c_idx]:.4f}"
                    plt.title(f"Image | Target | Prediction\n{msg}")
                    plt.tight_layout()
                    plt.axis("off")
                    plt.savefig(os.path.join(test_output_dir, f"{filename}.jpeg"), bbox_inches="tight", pad_inches=0)
                    plt.close()
