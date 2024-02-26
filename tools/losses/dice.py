# -*- coding: utf-8 -*-
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["BinaryDiceLoss", "MulticlassDiceLoss", "_binary_dice_loss", "_multiclass_dice_loss"]


class BinaryDiceLoss(nn.Module):
    def __init__(
        self,
        ignore_index: Optional[int] = None,
        reduction: str = "mean",
        smooth: float = 0.0,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.smooth = smooth
        self.eps = eps

    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        losses = {}
        for name, x in outputs.items():
            losses[name] = loss = _binary_dice_loss(
                outputs=x,
                targets=targets,
                ignore_index=self.ignore_index,
                reduction=self.reduction,
                smooth=self.smooth,
                eps=self.eps,
            )

        if len(losses) == 1:
            return losses["out"]

        return losses["out"]  # FIXME: auxil 또는 deep supervision 용 추가할 것. weight값추가 하면 좋을 듯


class MulticlassDiceLoss(nn.Module):
    def __init__(
        self,
        ignore_index: Optional[int] = None,
        reduction: str = "mean",
        smooth: float = 0.0,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.smooth = smooth
        self.eps = eps

    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        losses = {}
        for name, x in outputs.items():
            losses[name] = _multiclass_dice_loss(
                outputs=x,
                targets=targets,
                ignore_index=self.ignore_index,
                reduction=self.reduction,
                smooth=self.smooth,
                eps=self.eps,
            )
        if len(losses) == 1:
            return losses["out"]

        return losses["out"]  # FIXME: auxil 또는 deep supervision 용 추가할 것. weight값추가 하면 좋을 듯


def _binary_dice_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: Optional[int] = None,
    reduction: str = "mean",
    smooth: float = 0.0,
    eps: float = 1e-7,
) -> torch.Tensor:

    outputs = torch.sigmoid(outputs)
    outputs = torch.squeeze(outputs, dim=1)  # [N, 1, H, W] -> [N, H, W]
    outputs_flat = torch.flatten(outputs, start_dim=-2, end_dim=-1)  # [N, H, W] -> [N, H*W]
    targets_flat = torch.flatten(targets, start_dim=-2, end_dim=-1)  # [N, H, W] -> [N, H*W]

    if ignore_index is not None:
        mask = targets_flat != ignore_index
        outputs_flat = outputs_flat * mask
        targets_flat = targets_flat * mask

    numerator = torch.sum(outputs_flat * targets_flat, dim=-1)  # [N]
    denominator = torch.sum(outputs_flat + targets_flat, dim=-1)  # [N]
    batch_dice = torch.div((2.0 * numerator + smooth), (denominator + smooth)).clamp_min(eps)  # [N]
    batch_dice_loss = 1.0 - batch_dice

    if reduction == "mean":
        return torch.mean(batch_dice_loss)  # []
    elif reduction == "sum":
        return torch.sum(batch_dice_loss)  # []
    else:
        return batch_dice_loss  # [N]


def _multiclass_dice_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: Optional[int] = None,
    reduction: str = "mean",
    smooth: float = 0.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    num_classes = outputs.size()[1]
    outputs = torch.softmax(outputs, dim=1)
    outputs_flat = torch.flatten(outputs, start_dim=-2, end_dim=-1)  # [N, C, H, W] -> [N, C, H*W]
    targets_flat = torch.flatten(targets, start_dim=-2, end_dim=-1)  # [N, H, W] -> [N, H*W]

    if ignore_index is not None:
        mask = targets_flat != ignore_index
        outputs_flat = outputs_flat * mask.unsqueeze(1)
        targets_flat = targets_flat * mask

    targets_flat = F.one_hot(targets_flat.to(torch.long), num_classes=num_classes)  # [N, H*W, C]
    targets_flat = targets_flat.permute(0, 2, 1).type(outputs_flat.type())  # [N, C, H*W]

    numerator = torch.sum(outputs_flat * targets_flat, dim=-1)  # [N, C-1]
    denominator = torch.sum(outputs_flat + targets_flat, dim=-1)  # [N, C-1]
    batch_dice = torch.div((2.0 * numerator + smooth), (denominator + smooth)).clamp_min(eps)  # [N, C-1]
    batch_dice_loss = 1.0 - batch_dice  # [N, C-1]

    if reduction == "mean":
        return torch.mean(batch_dice_loss)  # []
    elif reduction == "sum":
        return torch.sum(batch_dice_loss)  # []
    else:
        return batch_dice_loss  # [N, C-1]
