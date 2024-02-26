# -*- coding: utf-8 -*-
from typing import Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import reduce_across_processes_tensor_list

__all__ = ["BinaryDiceScore", "MulticlassDiceScore"]


class BinaryDiceScore:
    def __init__(self, is_background: bool = False, reduction: str = "mean") -> None:
        assert reduction in ["mean", "none"], "reduction 은 mean or none"
        self.reduction = reduction
        self.is_background = is_background
        self.dice_score_list = None

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        if self.dice_score_list is None:
            self.dice_score_list = []
        with torch.inference_mode():
            preds = preds.squeeze(dim=1)  # [N, 1, H, W] -> [N, H, W]
            preds_flat = torch.flatten(preds, start_dim=-2, end_dim=-1)  # [N, H, W] -> [N, H*W]
            targets_flat = torch.flatten(targets, start_dim=-2, end_dim=-1)  # [N, H, W] -> [N, H*W]

            preds_flat = F.one_hot(preds_flat.to(torch.long), num_classes=2)  # [N, H*W, 2]
            preds_flat = preds_flat.permute(0, 2, 1).type(preds_flat.type())  # [N, 2, H*W]

            targets_flat = F.one_hot(targets_flat.to(torch.long), num_classes=2)  # [N, H*W, 2]
            targets_flat = targets_flat.permute(0, 2, 1).type(preds_flat.type())  # [N, 2, H*W]

            numerator = torch.sum(preds_flat * targets_flat, dim=-1)  # [N, 2]
            denominator = torch.sum(preds_flat + targets_flat, dim=-1)  # [N, 2]
            batch_dice = torch.div((2.0 * numerator), denominator)  # [N, 2]
            # 정답에도 전경(1)에 대한 segmentation mask 가 없으며, 예측 또한 예측한 전경(1) segmentation 없을 경우 dice = 0 이 아닌 nan 로 하여 평균 계산에서 배제
            # print(batch_dice)
            batch_dice[denominator == 0] = torch.nan
            if not self.is_background:
                batch_dice[:, 0] = torch.nan
            # print(batch_dice)
        self.dice_score_list.append(batch_dice)

    def reset(self):
        self.dice_score_list = []

    def compute(self):
        dice_score = torch.cat(self.dice_score_list)
        if self.reduction == "mean":
            return dice_score.nanmean()
        else:
            return dice_score.tolist()

    def reduce_from_all_processes(self):
        self.dice_score_list = reduce_across_processes_tensor_list(self.dice_score_list)

    def __str__(self):
        dice_score = self.compute()
        if self.reduction == "mean":
            msg = f"Mean Dice score: {dice_score:.4f}\n"
        else:
            msg = f"{dice_score}\n"
        return msg
    
    def __call__(self, preds: torch.Tensor, targets: torch.Tensor):
        with torch.inference_mode():
            preds = preds.squeeze(dim=1)  # [N, 1, H, W] -> [N, H, W]
            preds_flat = torch.flatten(preds, start_dim=-2, end_dim=-1)  # [N, H, W] -> [N, H*W]
            targets_flat = torch.flatten(targets, start_dim=-2, end_dim=-1)  # [N, H, W] -> [N, H*W]

            preds_flat = F.one_hot(preds_flat.to(torch.long), num_classes=2)  # [N, H*W, 2]
            preds_flat = preds_flat.permute(0, 2, 1).type(preds_flat.type())  # [N, 2, H*W]

            targets_flat = F.one_hot(targets_flat.to(torch.long), num_classes=2)  # [N, H*W, 2]
            targets_flat = targets_flat.permute(0, 2, 1).type(preds_flat.type())  # [N, 2, H*W]

            numerator = torch.sum(preds_flat * targets_flat, dim=-1)  # [N, 2]
            denominator = torch.sum(preds_flat + targets_flat, dim=-1)  # [N, 2]
            batch_dice = torch.div((2.0 * numerator), denominator)  # [N, 2]
            return batch_dice[:, 1]  # background 제외
        


class MulticlassDiceScore:
    def __init__(self, num_classes: int, is_background: bool = False, reduction: str = "mean") -> None:
        assert reduction in ["mean", "none"], "reduction 은 mean or none"
        self.num_classes = num_classes
        self.is_background = is_background
        self.reduction = reduction
        self.dice_score_list = None

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        if self.dice_score_list is None:
            self.dice_score_list = []
        with torch.inference_mode():
            preds_flat = torch.flatten(preds, start_dim=-2, end_dim=-1)  # [N, H, W] -> [N, C, H*W]
            targets_flat = torch.flatten(targets, start_dim=-2, end_dim=-1)  # [N, H, W] -> [N, H*W]

            preds_flat = F.one_hot(preds_flat.to(torch.long), num_classes=self.num_classes)  # [N, H*W, C]
            preds_flat = preds_flat.permute(0, 2, 1).type(preds_flat.type())  # [N, C, H*W]

            targets_flat = F.one_hot(targets_flat.to(torch.long), num_classes=self.num_classes)  # [N, H*W, C]
            targets_flat = targets_flat.permute(0, 2, 1).type(preds_flat.type())  # [N, C, H*W]

            numerator = torch.sum(preds_flat * targets_flat, dim=-1)  # [N, C]
            denominator = torch.sum(preds_flat + targets_flat, dim=-1)  # [N, C]
            batch_dice = torch.div((2.0 * numerator), denominator)  # [N, C]
            # print(batch_dice)
            batch_dice[denominator == 0] = torch.nan
            # print(batch_dice)
            if not self.is_background:
                batch_dice[:, 0] = torch.nan
            # print(batch_dice)
        self.dice_score_list.append(batch_dice)

    def reset(self):
        self.dice_score_list = []

    def compute(self):
        dice_score = torch.cat(self.dice_score_list)
        if self.reduction == "mean":
            return dice_score.nanmean()
        else:
            return dice_score.tolist()

    def reduce_from_all_processes(self):
        self.dice_score_list = reduce_across_processes_tensor_list(self.dice_score_list)

    def __str__(self):
        dice_score = self.compute()
        if self.reduction == "mean":
            msg = f"Mean Dice score: {dice_score}\n"
        else:
            msg = f"{dice_score}\n"
        return msg

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor):
        with torch.inference_mode():
            preds_flat = torch.flatten(preds, start_dim=-2, end_dim=-1)  # [N, H, W] -> [N, C, H*W]
            targets_flat = torch.flatten(targets, start_dim=-2, end_dim=-1)  # [N, H, W] -> [N, H*W]

            preds_flat = F.one_hot(preds_flat.to(torch.long), num_classes=self.num_classes)  # [N, H*W, C]
            preds_flat = preds_flat.permute(0, 2, 1).type(preds_flat.type())  # [N, C, H*W]

            targets_flat = F.one_hot(targets_flat.to(torch.long), num_classes=self.num_classes)  # [N, H*W, C]
            targets_flat = targets_flat.permute(0, 2, 1).type(preds_flat.type())  # [N, C, H*W]

            numerator = torch.sum(preds_flat * targets_flat, dim=-1)  # [N, C]
            denominator = torch.sum(preds_flat + targets_flat, dim=-1)  # [N, C]
            batch_dice = torch.div((2.0 * numerator), denominator)  # [N, C]
            return batch_dice[:, 1:]  # background 제외
