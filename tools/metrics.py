import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat

from sklearn.metrics import confusion_matrix

# TODO: Multiclass Dice score 정의 필요
# TODO: Binary Dice score 다시 확인할 것

class BinaryDiceScore(Metric):
    def __init__(self, average = True) -> None:
        super().__init__()
        self.eps = 1e-6
        self.average = average
        self.add_state("tp_list", default=[], dist_reduce_fx="cat")
        self.add_state("fp_list", default=[], dist_reduce_fx="cat")
        self.add_state("fn_list", default=[], dist_reduce_fx="cat")

        self.add_state("dice_list", default=[], dist_reduce_fx="cat")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds:Tensor, target: Tensor) -> None:
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")
        b, h, w = preds.shape
        preds = preds.view(-1, h*w)
        target = target.view(-1, h*w)
        self.total += b
        for i in range(b):
            tp = torch.sum((preds[i] == 1) & (target[i] == 1))
            fp = torch.sum((preds[i] == 1) & (target[i] == 0))
            fn = torch.sum((preds[i] == 0) & (target[i] == 1))
            self.tp_list.append(tp)
            self.fp_list.append(fp)
            self.fn_list.append(fn)

    def compute(self) -> Tensor:
        tp_list = dim_zero_cat(self.tp_list)
        fp_list = dim_zero_cat(self.fp_list)
        fn_list = dim_zero_cat(self.fn_list)
        dice_list = torch.divide(2*tp_list,(2*tp_list + fp_list + fn_list + self.eps))
        if self.average:
            return dice_list.mean()
        else:
            return dice_list


# TODO: FIX it
import numpy as np
def get_confusion_matrix(pred, label, size, num_classes, ignore=-1):
    output = pred.cpu().detach().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_gt = np.asarray(label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int16)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_classes + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_classes, num_classes))

    for i_label in range(num_classes):
        for i_pred in range(num_classes):
            cur_index = i_label * num_classes + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred] = label_count[cur_index]
    return confusion_matrix