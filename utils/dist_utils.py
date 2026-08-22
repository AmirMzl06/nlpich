import os
import random
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather as dist_all_gather
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1

def get_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=True
    )

def is_distributed() -> bool:
    """Return True when torch.distributed is available and initialized."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Return process rank or 0 for single-process runs."""
    return dist.get_rank() if is_distributed() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def setup_ddp(backend: str = "nccl") -> int:
    """
    Initialize torch.distributed from torchrun env variables.
    Returns the local rank so callers can set the CUDA device.
    """
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass


def pad_dim1(x, target_len, value=0):
    """
    Pads tensor x along dimension 1 up to target_len.
    Works for [B, T], [B, T, C], [B, T, ...].
    """
    curr_len = x.shape[1]
    if curr_len >= target_len:
        return x

    pad_amount = target_len - curr_len

    # F.pad pad tuple starts from the last dimension.
    # For dim=1, we need to construct pads for all trailing dims.
    pad = [0, 0] * (x.dim() - 2) + [0, pad_amount]

    return torch.nn.functional.pad(x, tuple(pad), value=value)


def gather_with_grad_seq(x):
    if not is_distributed():
        return x

    local_len = torch.tensor([x.shape[1]], device=x.device, dtype=torch.long)

    all_lens = [
        torch.zeros_like(local_len)
        for _ in range(get_world_size())
    ]

    dist.all_gather(all_lens, local_len)

    max_len = int(torch.stack(all_lens).max().item())

    x = pad_dim1(x, max_len, value=0)

    parts = dist_all_gather(x)
    return torch.cat(parts, dim=0)




def gather_with_grad(x: Any):
    """All-gather helper that preserves gradients for the local shard."""
    if not is_distributed():
        return x
    parts = dist_all_gather(x)
    return torch.cat(parts, dim=0)
