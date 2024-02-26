# -*- coding: utf-8 -*-
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ce import _binary_ce_loss, _multiclass_ce_loss
from .dice import _binary_dice_loss, _multiclass_dice_loss


__all__ = ["BinaryCEWithDiceLoss", "MultiClassCEWithDiceLoss"]


class BinaryCEWithDiceLoss(nn.Module):
    def __init__(
        self,
        ignore_index: Optional[int] = None,
        smooth: float = 0.0,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.eps = eps

    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        losses = {}
        for name, x in outputs.items():
            ce_loss = _binary_ce_loss(
                x,
                targets,
            )
            dice_loss = _binary_dice_loss(
                outputs=x,
                targets=targets,
                ignore_index=self.ignore_index,
                smooth=self.smooth,
                eps=self.eps,
            )
            losses[name] = ce_loss + dice_loss

        if len(losses) == 1:
            return losses["out"]

        return losses["out"]  # FIXME: auxil 또는 deep supervision 용 추가할 것. weight값추가 하면 좋을 듯


class MultiClassCEWithDiceLoss(nn.Module):
    def __init__(
        self,
        ignore_index: Optional[int] = None,
        smooth: float = 0.0,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.eps = eps

    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        losses = {}
        for name, x in outputs.items():
            ce_loss = _multiclass_ce_loss(
                x,
                targets,
                ignore_index=self.ignore_index,
            )

            dice_loss = _multiclass_dice_loss(
                outputs=x,
                targets=targets,
                ignore_index=self.ignore_index,
                smooth=self.smooth,
                eps=self.eps,
            )
            losses[name] = ce_loss + dice_loss

        if len(losses) == 1:
            return losses["out"]

        return losses["out"]  # FIXME: auxil 또는 deep supervision 용 추가할 것. weight값추가 하면 좋을 듯
