# -*- coding: utf-8 -*-
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["BinaryCE", "MulticlassCE", "_binary_ce_loss", "_multiclass_ce_loss"]


class BinaryCE(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        losses = {}
        for name, x in outputs.items():
            losses[name] = _binary_ce_loss(x, targets, self.reduction)

        if len(losses) == 1:
            return losses["out"]

        return losses["out"] # FIXME: auxil 또는 deep supervision 용 추가할 것. weight값추가 하면 좋을 듯


class MulticlassCE(nn.Module):
    def __init__(
        self,
        reduction: str = "mean",
        ignore_index: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        losses = {}
        for name, x in outputs.items():
            losses[name] = _multiclass_ce_loss(x, targets, self.reduction, self.ignore_index)

        if len(losses) == 1:
            return losses["out"]

        return losses["out"] # FIXME: auxil 또는 deep supervision 용 추가할 것. weight값추가 하면 좋을 듯


def _binary_ce_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    outputs = torch.squeeze(outputs, dim=1)  # [N, 1, H, W] -> [N, H, W]
    outputs_flat = torch.flatten(outputs, start_dim=-2, end_dim=-1)  # [N, H, W] -> [N, H*W]
    targets_flat = torch.flatten(targets, start_dim=-2, end_dim=-1).to(outputs_flat.dtype)  # [N, H, W] -> [N, H*W]

    loss = F.binary_cross_entropy_with_logits(
        outputs_flat,
        targets_flat,
        reduction=reduction,
    )

    return loss


def _multiclass_ce_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
    ignore_index: Optional[int] = None,
) -> torch.Tensor:
    if ignore_index is None:
        ignore_index = -100

    loss = F.cross_entropy(
        outputs,
        targets,
        reduction=reduction,
        ignore_index=ignore_index,
    )

    return loss
