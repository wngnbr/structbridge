"""
Helpers for distributed training.
"""

import os

import torch
import torch.distributed as dist

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))


def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("Launch with python -m torch.distributed.run --standalone --nproc_per_node=1")
    if torch.cuda.is_available():
        torch.cuda.set_device(LOCAL_RANK)
    backend = "gloo" if not torch.cuda.is_available() else "nccl"
    dist.init_process_group(backend)


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
