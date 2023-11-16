# -*- coding: utf-8 -*-

# Classification 전용 train.py
import os
import json
import copy
import time
import wandb
import datetime
import argparse
import numpy as np
from sklearn.model_selection import ShuffleSplit, StratifiedKFold

import torch
import torch.nn as nn

from torch.backends import cudnn
from torch.utils.data import Subset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import utils
from transforms import add_argparser_transform, get_transform
from datasets import add_argparser_dataset, get_dataset
from models import add_argparser_mdoel, get_model
from tools import Segmentation


def main(args):
    if args.output_suffix is None:
        # 기본 경로: {output_dir}/{model_name}-{data_name}/
        output_dir_temp = f"{args.model_name}-{args.dataset_name}"
    else:
        # 기본 경로: {output_dir}/{model_name}-{data_name}-{suffix}/
        output_dir_temp = f"{args.model_name}-{args.dataset_name}-{args.output_suffix}"

    args.output_dir = os.path.join(args.output_dir, output_dir_temp)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M-%S")
    # 최종 경로 1: # 기본 경로: {output_dir}/{model_name}-{data_name}/{current_time}/
    # 최종 경로 2: # 기본 경로: {output_dir}/{model_name}-{data_name}-{suffix}/{current_time}/
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

    # Call Train Dataset
    train_transforms, valid_transforms = get_transform(
        task="seg",
        is_train=True,
        **vars(args),
    )

    dataset = get_dataset(
        is_train=True,
        transforms=train_transforms,
        **vars(args),
    )

    args.num_classes = dataset.num_classes
    temp_dataset = copy.deepcopy(dataset)
    print(dataset)

    dataset.set_transforms(train_transforms)
    temp_dataset.set_transforms(valid_transforms)
    if args.criterion_name == "w-ce":
        weight = temp_dataset.class_weight()
        args.weight = weight
    else:
        args.weight = None

    rng = np.random.RandomState(args.seed)

    fold_list = []
    if args.k_fold >= 5:  # TODO: FIX IT
        fold = StratifiedKFold(
            n_splits=args.k_fold,
            shuffle=True,
            random_state=rng,
        )
        for sp in fold.split(dataset.inputs, dataset.labels):
            fold_list.append(sp)
    else:
        n_train = int(len(dataset) * 0.9)
        n_valid = len(dataset) - n_train
        train_valid_fold = ShuffleSplit(n_splits=1, test_size=n_valid, random_state=rng)
        train_index, valid_index = next(
            train_valid_fold.split(np.asarray(dataset.inputs))
        )
        fold_list.append((train_index, valid_index))

    for ith_fold, (train_index, valid_index) in enumerate(fold_list, start=1):
        print(
            f"\nCreate the fold output directory: [{args.output_dir}/fold{ith_fold}]",
            end="\n\n",
        )
        utils.mkdir(os.path.join(args.output_dir, f"fold{ith_fold}"))
        train_dataset = Subset(dataset, train_index)
        valid_dataset = Subset(temp_dataset, valid_index)

        train_inputs = np.asarray(dataset.inputs)[train_index].tolist()
        valid_inputs = np.asarray(dataset.inputs)[valid_index].tolist()

        if set(train_inputs).intersection(set(valid_inputs)):
            raise RuntimeError(
                "두 데이터에 간섭이 존재함."
                f"[{set(train_inputs).intersection(set(valid_inputs))}]"
            )

        with open(
            os.path.join(args.output_dir, f"fold{ith_fold}", f"train.yaml"), "w"
        ) as f:
            json.dump(train_inputs, f)

        with open(
            os.path.join(args.output_dir, f"fold{ith_fold}", f"valid.yaml"), "w"
        ) as f:
            json.dump(valid_inputs, f)

        if args.distributed:
            train_sampler = DistributedSampler(train_dataset)
            valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
        else:
            train_sampler = RandomSampler(train_dataset)
            valid_sampler = SequentialSampler(valid_dataset)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            sampler=valid_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        print("Creating Model")
        model = get_model(**vars(args))
        model.to(device)
        print(model, end="\n\n")

        if args.distributed:
            if args.sync_bn:
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

        trainer = Segmentation(model=model, **vars(args))

        start_time = time.time()
        wandb.init(
            project=f"{args.project_name}",  # knuh-ear-ct
            group=f"{output_dir_temp}",
            name=f"fold{ith_fold}",
            config=vars(args),
        )

        trainer.train(
            ith_fold=ith_fold,
            wandb=wandb,
            start_epoch=args.start_epoch,
            epochs=args.epochs,
            train_loader=train_loader,
            valid_loader=valid_loader,
            train_sampler=train_sampler,
            args=args,
        )

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"Training time {total_time_str}")
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="모델 학습을 위한 모든 종류의 파라미터를 선언합니다.")
    parser.add_argument("--project-name", required=True, type=str)
    parser.add_argument("--seed", default=42, type=int, help="Model 재현을 위한 랜덤 시드 고정")

    parser.add_argument(
        "--output-dir", default="./results-cls", type=str, help="모델의 학습 결과 및 가중치 저장"
    )
    parser.add_argument(
        "--output-suffix",
        default=None,
        type=str,
        help="프로그램을 실행할 때 자동으로 [output_dir]/[model_name]-[data_name]-[output_suffix]} 하위 폴더 생성",
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

    parser.add_argument(
        "--k-fold",
        default=0,
        type=int,
        help="None 또는 5 이하일 경우, Hold-out / 정수이며 5 이상일 경우 K fold cross validatoin 동작 ",
    )

    parser.add_argument("--batch-size", default=8, type=int, help="train batch size")
    parser.add_argument("--dataset-name", required=True, type=str, help="데이터 선택하기")
    parser.add_argument("--model-name", required=True, type=str, help="모델 선택하기")
    parser.add_argument("--resume", type=str)
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="start epoch",
    )
    parser.add_argument("--epochs", default=50, type=int, help="Training epoch size")

    parser.add_argument(
        "--optim-name",
        required=True,
        type=str,
        help="최적화 함수 선택",
    )

    # TODO: parser.add_argument(
    #     "--lr-scheduler",
    #     default=None,
    #     type=str,
    #     help="스케쥴러 선택",
    # )

    temp_args, _ = parser.parse_known_args()

    parser, max_pixel_value = add_argparser_dataset(parser, temp_args.dataset_name)
    parser = add_argparser_transform(parser, task="seg", is_train=True)
    parser = add_argparser_mdoel(parser, temp_args.model_name, is_inference=False)
    parser = Segmentation.add_argparser_optim(parser, temp_args.optim_name)
    # TODO: ADD lr_scheduler

    args = parser.parse_args()
    args.max_pixel_value = max_pixel_value

    main(args)
