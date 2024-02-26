# -*- coding: utf-8 -*-
from argparse import ArgumentParser
from typing import Tuple, Callable, Optional

import os
import numpy as np
from PIL import Image
from glob import glob

import torch

from .template import ClassificationVisionDataset


class CatDogDataset(ClassificationVisionDataset):
    # https://www.kaggle.com/datasets/tongpython/cat-and-dog?rvi=1
    def __init__(self, root: str, is_train: bool, transforms: Optional[Callable] = None):
        super().__init__(root, is_train, transforms)
        self.classes = {"cat": 0, "dog": 1}
        self.inv_classes = {0: "cat", "dog": 1}
        self.load_database()

    @staticmethod
    def max_pixel_value():
        return 255.0

    @staticmethod
    def add_argparser(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("Cat Dog Binary Classification")
        parser.add_argument("--root", required=True, type=str, help="데이터 존재하는 상위폴더 경로")
        return parent_parser

    def load_database(self) -> None:
        phase = "train" if self.is_train else "test"
        for label, cls in self.classes.items():
            for file_path in glob(os.path.join(self.root, phase, f"{label}/*.jpg")):
                self.inputs.append(file_path)
                self.labels.append(cls)

        self.distribution = torch.zeros(size=[len(self.classes)])
        for label in self.classes.values():
            self.distribution[label] = self.labels.count(label)

    def read_image(self, file_path):
        image = Image.open(file_path)
        image = np.asarray(image)
        return image
