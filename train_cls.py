# -*- coding: utf-8 -*-
import os
import json
import copy
import time
import wandb
import datetime
import argparse
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

import torch
import torch.nn as nn

from torch.backends import cudnn
from torch.utils.data import Subset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import utils
from datasets import add_argparser_dataset, get_dataset
from transforms import add_argparser_cls_transform, create_cls_transform
from models import add_argparser_cls_model, create_cls_model
from tools import add_argparser_cls_tool, create_cls_tool


def main(args):
    # 결과 저장소 생성
    args.output_dir, output_dir_temp = utils.create_output_dir(
        args.model_name, args.dataset_name, args.output_suffix, args.output_dir
    )

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
    train_transforms, valid_transforms = create_cls_transform(
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
    if args.k_fold >= 5:
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
        train_valid_fold = StratifiedShuffleSplit(
            train_size=n_train, test_size=n_valid, random_state=rng
        )
        train_index, valid_index = next(
            train_valid_fold.split(np.asarray(dataset.inputs), dataset.labels)
        )
        fold_list.append((train_index, valid_index))

    for ith_fold, (train_index, valid_index) in enumerate(fold_list, start=1):
        print(f"\nCreate the fold output directory: [{args.output_dir}/fold{ith_fold}]", end="\n\n")
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

        with open(os.path.join(args.output_dir, f"fold{ith_fold}", f"train.yaml"), "w") as f:
            json.dump(train_inputs, f)

        with open(os.path.join(args.output_dir, f"fold{ith_fold}", f"valid.yaml"), "w") as f:
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
        model = create_cls_model(**vars(args))
        model.to(device)
        print(model, end="\n\n")

        if args.distributed:
            if args.sync_bn:
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

        trainer = create_cls_tool(model=model, **vars(args))

        
        start_time = time.time()
        wandb.init(
            project=f"{args.project_name}",
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
    parser.add_argument("--task", required=True, type=str, help="Choice ['binary', 'multiclass']")

    parser = utils.create_base_parser(parser=parser)
    temp_args, _ = parser.parse_known_args()

    parser, max_pixel_value = add_argparser_dataset(parser, temp_args.dataset_name)
    parser = add_argparser_cls_transform(parser, is_train=True)
    parser = add_argparser_cls_model(parser, temp_args.model_name)
    parser = add_argparser_cls_tool(temp_args.task, parser, temp_args.optim_name)

    args = parser.parse_args()
    args.max_pixel_value = max_pixel_value
    
    main(args)
