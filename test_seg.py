
import os
import argparse

import torch

from torch.utils.data import DataLoader

from transforms import get_transform
from datasets import get_dataset
from models import get_model

from tools import Segmentation

import utils


def main(args):
    device = torch.device("cuda:1")
    ckpt = torch.load(args.ckpt_path, map_location="cpu")

    train_args = ckpt["args"]
    ith_fold = ckpt["ith_fold"]

    print(f"save path: {train_args.output_dir}")
    print(f"fold {ith_fold}")
    print(f"{ckpt['epoch']}")

    test_output_dir = os.path.join(train_args.output_dir, f"fold{ith_fold}", "test")
    if not os.path.exists(test_output_dir):
        utils.mkdir(test_output_dir)
        
    model = get_model(**vars(train_args))
    model.load_state_dict(ckpt["model"])
    model.to(device)

    test_transforms = get_transform(task="seg", is_train=False, **vars(train_args))
    test_dataset = get_dataset(is_train=False, **vars(train_args), transforms=test_transforms)
    print(test_dataset)

    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    print(test_dataset.classes)
    Segmentation.test(model, test_loader, num_classes=test_dataset.num_classes, classes=test_dataset.classes, device=device, test_output_dir=test_output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="학습 모델 테스트 하기 위한 파라미터")
    parser.add_argument("--ckpt-path", default="", type=str, help="학습된 모델의 파라미터 및 가중치 파일")

    args = parser.parse_args()
    
    main(args)