# -*-coding: utf-8
import argparse

import torch.nn as nn

from .classification import add_argparser_cls_model, create_cls_model
from .segmentation import add_argparser_seg_model, create_seg_model

