# -*-coding: utf-8

from ..template import BaseModelTemplate
from ..criterion import CrossEntropy, DiceLoss, CrossEntropyWithDiceLoss

class SegmentationModelTemplate(BaseModelTemplate):
    num_classes: int

    def __init__(
        self, num_classes: int, is_inference: bool, criterion_name: str = "", **kwargs
    ) -> None:
        super().__init__(is_inference, criterion_name, **kwargs)
        self.num_classes = num_classes

    def __str__(self) -> str:
        msg = super().__str__()
        msg += f"  - Num of classes: {self.num_classes}\n"
        return msg

    def set_criterion(self, criterion_name, **kwargs):
        if criterion_name == "ce":
            self.criterion = CrossEntropy()
        elif criterion_name == "dice":
            self.criterion = DiceLoss()
        elif criterion_name == "ce-dice":
            self.criterion = CrossEntropyWithDiceLoss()
        else:
            raise NotImplementedError(f"{criterion_name} is not supported")

