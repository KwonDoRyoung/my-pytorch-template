# -*-coding: utf-8
import os
import time
import argparse
import datetime
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

import torch
import torch.nn as nn

from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import utils
from datasets import add_argparser_dataset, get_dataset
from transforms import add_argparser_transform, get_transform
from models import add_argparser_mdoel, get_model
from tools import (
    add_argparser_optim,
    # add_argparser_criterion,
    add_argparser_lr_scheduler,
    Classification,
    Segmentation,
)


def main(args):
    if args.output_suffix is None:
        # 기본 경로: {output_dir}/{task}-{model_name}-{data_name}/
        temp = f"{args.task}-{args.model}-{args.dataset}"
    else:
        # 기본 경로: {output_dir}/{task}-{model_name}-{data_name}-{suffix}/
        temp = f"{args.task}-{args.model}-{args.dataset}-{args.output_suffix}"

    args.output_dir = os.path.join(args.output_dir, temp)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    # 최종 경로 1: # 기본 경로: {output_dir}/{task}-{model_name}-{data_name}/{current_time}/
    # 최종 경로 2: # 기본 경로: {output_dir}/{task}-{model_name}-{data_name}-{suffix}/{current_time}/
    args.output_dir = os.path.join(args.output_dir, current_time)

    print(f"\nCreate the output directory: [{args.output_dir}]", end="\n\n")
    utils.mkdir(args.output_dir)

    # For Reproducibility
    utils.set_seed(args.seed)

    # Set the DDP GPU or Single GPU
    utils.init_distributed_mode(args)
    print(args, end="\n\n")

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        cudnn.benchmark = True

    train_transforms, valid_transforms = get_transform(
        task=args.task,
        dataset=args.dataset,
        **vars(args),
    )

    train_dataset = get_dataset(
        args.dataset,
        phase="train",
        transforms=train_transforms,
        **vars(args),
    )
    valid_dataset = get_dataset(
        args.dataset,
        phase="valid",
        transforms=valid_transforms,
        **vars(args),
    )

    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
        valid_sampler = SequentialSampler(valid_dataset)
    else:
        train_sampler = RandomSampler(train_dataset)
        valid_sampler = SequentialSampler(valid_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        sampler=valid_sampler,
    )

    print("Creating Model")
    model = get_model(**vars(args))
    model.to(device)
    print(model, end="\n\n")

    if args.distributed and args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.task == "cls":
        trainer = Classification()
    elif args.task == "seg":
        trainer = Segmentation()
    else:
        raise RuntimeError(f"{args.task} is not supported")

    start_time = time.time()

    trainer.train()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="모델 학습을 위한 모든 종류의 파라미터를 선언합니다.")
    parser.add_argument("--seed", default=42, type=int, help="Model 재현을 위한 랜덤 시드 고정")

    parser.add_argument(
        "--task", default="cls", type=str, help="cls(classification), seg(segmentation)"
    )
    parser.add_argument(
        "--output_dir", default="./results", type=str, help="모델의 학습 결과 및 가중치 저장"
    )
    parser.add_argument(
        "--output_suffix",
        default=None,
        type=str,
        help="프로그램을 실행할 때 자동으로 [output_dir]/[task]-[model_name]-[data_name]-[output_suffix]} 하위 폴더 생성",
    )
    parser.add_argument("--print-freq", default=10, type=int, help="print 주기")

    parser.add_argument("--device", default="cuda", type=str, help="cuda or cpu")
    parser.add_argument(
        "--num-workers", default=8, type=int, help="학습 시 Dataloader가 활용하는 CPU 개수를 뜻함"
    )
    parser.add_argument(
        "--use-deterministic-algorithms",
        action="store_true",
        help="Forces the use of deterministic algorithms only.",
    )
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )

    parser.add_argument("--batch-size", default=8, type=int, help="train batch size")
    parser.add_argument("--dataset", required=True, type=str, help="데이터 선택하기")

    parser.add_argument("--model", required=True, type=str, help="모델 선택하기")

    parser.add_argument("--resume", type=str)
    parser.add_argument(
        "--start-epoch", default=0, type=int, metavar="N", help="start epoch"
    )

    parser.add_argument("--epochs", default=50, type=int, help="Training epoch size")
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate")

    parser.add_argument(
        "--optimizer",
        required=True,
        type=str,
        help="최적화 함수 선택",
    )

    parser.add_argument(
        "--criterion",
        required=True,
        type=str,
        help="손실 함수 선택",
    )

    parser.add_argument(
        "--lr-scheduler",
        default=None,
        type=str,
        help="스케쥴러 선택",
    )

    temp_args, _ = parser.parse_known_args()

    parser = add_argparser_dataset(parser, temp_args.dataset)
    parser = add_argparser_transform(parser, temp_args.task)
    parser = add_argparser_mdoel(parser, temp_args.model)
    parser = add_argparser_optim(parser, temp_args.optimizer)
    # parser = add_argparser_criterion(parser, temp_args.criterion)
    parser = add_argparser_lr_scheduler(parser, temp_args.lr_scheduler)

    args = parser.parse_args()

    main(args)
