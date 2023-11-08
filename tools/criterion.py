# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import functional as F

class CrossEntropyWithDiceLoss(nn.Module):
    def __init__(self, ignore_label=-1, weight=None, reduction="mean"):
        super(CrossEntropyWithDiceLoss, self).__init__()
        self.ce_loss = CrossEntropy(ignore_label, weight)
        self.dice_loss = DiceLoss(reduction)
        
    def forward(self, score, target):
        loss1 = self.ce_loss(score, target)
        loss2 = self.dice_loss(score, target)
        return loss1 + loss2

class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label)

    def forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(
                input=score, size=(h, w), mode='bilinear')

        loss = self.criterion(score, target)

        return loss


class DiceLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(DiceLoss, self).__init__()
        self.reduction = reduction

    def forward(self, score, target):
        assert score.size()[0] == target.size()[0]  # batch size
        assert score.size()[-2] == target.size()[-2]  # height
        assert score.size()[-1] == target.size()[-1]  # width

        if score.size()[1] == 1:  # binary
            return _binary_dice_loss(score, target, reduction=self.reduction)
        else:
            assert score.size()[1] > target.max()
            return _dice_loss(score, target, reduction=self.reduction)


def _binary_dice_loss(outputs, targets, reduction="mean"):
    """

    :param outputs: [N, 1, H, W]
    :param targets: [N, H, W]
    :param reduction:
    :return:
    """
    epsilon = 1e-7
    outputs = torch.sigmoid(outputs)
    outputs = torch.squeeze(outputs)  # [N, 1, H, W] -> [N, H, W]
    outputs_flat = torch.flatten(outputs, start_dim=-2, end_dim=-1)  # [N, H, W] -> [N, H*W]
    targets_flat = torch.flatten(targets, start_dim=-2, end_dim=-1)  # [N, H, W] -> [N, H*W]

    numerator = torch.sum(outputs_flat * targets_flat, dim=-1)  # [N]
    denominator = torch.sum(outputs_flat + targets_flat, dim=-1)  # [N]
    batch_dice = torch.div((2. * numerator + epsilon), (denominator + epsilon))  # [N]
    batch_dice_loss = torch.tensor(1.) - batch_dice
    if reduction == "mean":
        return torch.mean(batch_dice_loss)  # []
    elif reduction == "sum":
        return torch.sum(batch_dice_loss)  # []
    else:
        return batch_dice_loss  # [N]


def _dice_loss(outputs, targets, reduction="mean"):
    """

    :return:
    """
    epsilon = 1e-7
    outputs = torch.softmax(outputs, dim=1)
    outputs_flat = torch.flatten(outputs, start_dim=-2, end_dim=-1)  # [N, C, H, W] -> [N, C, H*W]
    targets_flat = torch.flatten(targets, start_dim=-2, end_dim=-1)  # [N, H, W] -> [N, H*W]
    targets_flat = torch.nn.functional.one_hot(targets_flat.to(torch.int64), num_classes=outputs_flat.size()[1])
    targets_flat = targets_flat.permute(0, 2, 1).type(outputs_flat.type())  # [N, C, H*W]

    outputs_flat = outputs_flat[:, 1:, :]  # [N, C-1, H*W] <- exclude background
    targets_flat = targets_flat[:, 1:, :]  # [N, C-1, H*W] <- exclude background

    numerator = torch.sum(outputs_flat * targets_flat, dim=-1)  # [N, C-1]
    denominator = torch.sum(outputs_flat + targets_flat, dim=-1)  # [N, C-1]
    batch_dice = torch.div((2. * numerator + epsilon), (denominator + epsilon))  # [N, C-1]
    batch_dice_loss = torch.tensor(1.) - batch_dice  # [N, C-1]

    if reduction == "mean":
        return torch.mean(batch_dice_loss)  # []
    elif reduction == "sum":
        return torch.sum(batch_dice_loss)  # []
    else:
        return batch_dice_loss  # [N, C-1]