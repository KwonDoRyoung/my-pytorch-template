# -*- coding: utf-8 -*-
import os
import argparse

import torch

from transforms import create_seg_transform
from datasets import get_dataset
from models import create_seg_model

from tools import BinarySegmentation, MultiSegmentation

import utils


def main(args):
    device = torch.device("cuda")
    ckpt = torch.load(args.ckpt_path, map_location="cpu")

    train_args = ckpt["args"]
    ith_fold = ckpt["ith_fold"]
    print(train_args)

    print(f"save path: {train_args.output_dir}")
    print(f"fold {ith_fold}")
    print(f"{ckpt['epoch']}")

    test_output_dir = os.path.join(os.path.dirname(args.ckpt_path), "test")
    if not os.path.exists(test_output_dir):
        utils.mkdir(test_output_dir)

    model = create_seg_model(**vars(train_args))
    model.load_state_dict(ckpt["model"])
    model.to(device)

    test_transforms = create_seg_transform(is_train=False, **vars(train_args))
    test_dataset = get_dataset(is_train=False, **vars(train_args), transforms=test_transforms)
    print(test_dataset)

    if train_args.task == "binary":
        BinarySegmentation.test(model, test_dataset, device=device)
        if args.visualization:
            BinarySegmentation.visualization(model, test_dataset, device=device, test_output_dir=test_output_dir)
    elif train_args.task == "multiclass":
        MultiSegmentation.test(model, test_dataset, device=device)
        if args.visualization:
            MultiSegmentation.visualization(model, test_dataset, device=device, test_output_dir=test_output_dir)
    else:
        raise RuntimeError(f"{train_args.task} is not supported!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="학습 모델 테스트 하기 위한 파라미터")
    parser.add_argument("--ckpt-path", required=True, type=str, help="학습된 모델의 파라미터 및 가중치 파일")
    parser.add_argument("--visualization", action="store_true", help="")

    args = parser.parse_args()

    main(args)
