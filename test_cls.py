# -*- coding: utf-8 -*-
import os
import argparse

import torch

from transforms import create_cls_transform
from datasets import get_dataset
from models import create_cls_model

from tools import BinaryClassification, MultiClassification

import utils


def main(args):
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    device = torch.device("cuda")

    train_args = ckpt["args"]
    print(train_args)
    ith_fold = ckpt["ith_fold"]

    print(f"{train_args.output_dir = }")
    print(f"{ith_fold = }")
    print(f"{ckpt['epoch'] = }")

    test_output_dir = os.path.join(os.path.dirname(args.ckpt_path), "test")
    if not os.path.exists(test_output_dir):
        utils.mkdir(test_output_dir)
        
    model = create_cls_model(**vars(train_args))
    model.load_state_dict(ckpt["model"])
    model.to(device)

    test_transforms = create_cls_transform(is_train=False, **vars(train_args))
    test_dataset = get_dataset(is_train=False, **vars(train_args), transforms=test_transforms)
    print(test_dataset)

    if train_args.task == "binary":
        BinaryClassification.test(model, test_dataset, device=device, test_output_dir=test_output_dir)
        if args.visualization:
            BinaryClassification.visualization(model, test_dataset, device=device, test_output_dir=test_output_dir)
    elif train_args.task == "multiclass":
        MultiClassification.test(model, test_dataset, device=device, test_output_dir=test_output_dir)
        if args.visualization:
            MultiClassification.visualization(model, test_dataset, device=device, test_output_dir=test_output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="학습 모델 테스트 하기 위한 파라미터")
    parser.add_argument("--ckpt-path", default="", type=str, help="학습된 모델의 파라미터 및 가중치 파일")
    parser.add_argument("--visualization", action="store_true", help="")

    args = parser.parse_args()
    
    main(args)