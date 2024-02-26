# -*- coding: utf-8 -*-
from argparse import ArgumentParser
from typing import Tuple, Callable, Optional

import os
import numpy as np
from PIL import Image

import torch
from torchvision.datasets import MNIST

from .template import ClassificationVisionDataset


class MNISTDataset(ClassificationVisionDataset):
    def __init__(self, root: str, is_train: bool, transforms: Optional[Callable] = None):
        super().__init__(root, is_train, transforms)
        self.classes = {str(i): i for i in range(10)}
        self.inv_classes = {i: str(i) for i in range(10)}
        self.load_database()

    @staticmethod
    def max_pixel_value():
        return 255.0

    @staticmethod
    def add_argparser(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("MNIST Classification")
        parser.add_argument("--root", required=True, type=str, help="데이터 존재하는 상위폴더 경로")
        return parent_parser

    def load_database(self) -> None:
        mnist = MNIST(root=self.root, train=self.is_train, download=True)
        save_path = os.path.join(self.root, "MNIST/image")
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for idx, (image, label) in enumerate(mnist):
            file_path = os.path.join(save_path, f"{label}_{idx}.jpeg")
            if not os.path.exists(file_path):
                image.save(file_path)
            self.inputs.append(file_path)
            self.labels.append(label)

        self.distribution = torch.zeros(size=[len(self.classes)])
        for label in self.classes.values():
            self.distribution[label] = self.labels.count(label)

    def read_image(self, file_path):
        image = Image.open(file_path)
        image = np.asarray(image)
        return image
