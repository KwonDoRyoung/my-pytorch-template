# -*-coding: utf-8
import os
import time
import errno
import random
import datetime
import numpy as np
from collections import defaultdict, deque

import torch
import torch.nn.functional as F
import torch.distributed as dist


def create_base_parser(parser):
    parser.add_argument("--project-name", required=True, type=str)
    parser.add_argument("--seed", default=42, type=int, help="Model 재현을 위한 랜덤 시드 고정")

    parser.add_argument("--output-dir", default="./results-cls", type=str, help="모델의 학습 결과 및 가중치 저장")
    parser.add_argument(
        "--output-suffix",
        default=None,
        type=str,
        help="프로그램을 실행할 때 자동으로 [output_dir]/[model_name]-[data_name]-[output_suffix]} 하위 폴더 생성",
    )
    parser.add_argument("--print-freq", default=10, type=int, help="print 주기")

    parser.add_argument("--device", default="cuda", type=str, help="cuda or cpu")
    parser.add_argument("--num-workers", default=0, type=int, help="학습 시 Dataloader가 활용하는 CPU 개수를 뜻함")
    parser.add_argument("--sync-bn", action="store_true")
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
    return parser


def create_output_dir(model_name, dataset_name, output_suffix, output_dir):
    if output_suffix is None:
        # 기본 경로: {output_dir}/{model_name}-{data_name}/
        output_dir_temp = f"{model_name}-{dataset_name}"
    else:
        # 기본 경로: {output_dir}/{model_name}-{data_name}-{suffix}/
        output_dir_temp = f"{model_name}-{dataset_name}-{output_suffix}"

    output_dir = os.path.join(output_dir, output_dir_temp)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M-%S")
    # 최종 경로 1: # 기본 경로: {output_dir}/{model_name}-{data_name}/{current_time}/
    # 최종 경로 2: # 기본 경로: {output_dir}/{model_name}-{data_name}-{suffix}/{current_time}/
    output_dir = os.path.join(output_dir, current_time)

    print(f"\nCreate the output directory: [{output_dir}]", end="\n\n")
    mkdir(output_dir)
    return output_dir, output_dir_temp


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    # SLURM 관련 부분에서 에러나는 듯 확인해 볼 것
    # 참고: https://github.com/pytorch/vision/tree/main/references
    # elif "SLURM_PROCID" in os.environ:
    #     args.rank = int(os.environ["SLURM_PROCID"])
    #     args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print("Not using distributed mode", end="\n\n")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        f"| distributed init (rank {args.rank}): {args.dist_url}",
        flush=True,
        end="\n\n",
    )
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    dist.barrier()
    setup_for_distributed(args.rank == 0)


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        t = reduce_across_processes([self.count, self.total])
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str}")


def reduce_across_processes(val):
    if not is_dist_avail_and_initialized():
        # nothing to sync, but we still convert to tensor for consistency with the distributed case.
        return torch.tensor(val)

    t = torch.tensor(val, device="cuda")
    dist.barrier()
    dist.all_reduce(t)
    return t


def reduce_across_processes_tensor_list(tensor_list):
    if not is_dist_avail_and_initialized():
        return torch.cat(tensor_list, dim=0)

    local_tensor = torch.cat(tensor_list, dim=0).to("cuda")  # Concatenate tensor list locally.

    # Calculate the total number of elements in the local tensor.
    local_num_elements = local_tensor.numel()

    # Share the number of elements with all processes.
    num_elements_tensor = torch.tensor([local_num_elements], dtype=torch.int64, device="cuda")
    all_num_elements = [torch.tensor([0], dtype=torch.int64, device="cuda") for _ in range(dist.get_world_size())]
    dist.all_gather(all_num_elements, num_elements_tensor)

    # Find the maximum number of elements across all processes.
    max_num_elements = max(all_num_elements).item()

    # Ensure the local tensor has the same number of elements as the max by padding if necessary.
    if local_num_elements < max_num_elements:
        padding = max_num_elements - local_num_elements
        # Assume padding at the end. Adjust padding logic based on the actual tensor shape and requirements.
        local_tensor = F.pad(local_tensor, (0, padding), "constant", 0)

    # Create a tensor that will hold the gathered data from all processes.
    # Here, we assume that the first dimension is batch size which we will gather across.
    gathered_tensors = [
        torch.zeros(max_num_elements, dtype=local_tensor.dtype, device="cuda") for _ in range(dist.get_world_size())
    ]

    # All-gather across all processes.
    dist.barrier()
    dist.all_gather(gathered_tensors, local_tensor)
    # Concatenating the gathered tensors may not be necessary since all_gather already places the tensors
    # in tensors_gathered next to each other in the list. We might simply want to combine these into a single tensor,
    # depending on how you plan to use the result.
    # The output tensor already contains the data gathered from all processes, so we just return it.
    return gathered_tensors
