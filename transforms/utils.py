# -*- coding: utf-8 -*-
import numpy as np

import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform

def flip(hflip_prob, vflip_prob):
    temp_list = []
    if 0 < hflip_prob < 1:
        temp_list.append(A.HorizontalFlip(p=hflip_prob))
    elif hflip_prob >= 1:
        temp_list.append(A.HorizontalFlip(p=1))

    if 0 < vflip_prob < 1:
        temp_list.append(A.VerticalFlip(p=vflip_prob))
    elif vflip_prob >= 1:
        temp_list.append(A.VerticalFlip(p=1))

    return temp_list


def converter_ch1_to_ch3(img: np.ndarray):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
        return np.concatenate([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[-1] == 3:
        return img
    else:
        raise RuntimeError(f"{img.shape, type(img)}")


class ConverterCh1toCh3(ImageOnlyTransform):
    def __init__(self, always_apply: bool = False):
        super().__init__(always_apply, 1.0)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return converter_ch1_to_ch3(img)

    def get_params_dependent_on_targets(self, params):
        return params

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return ""
