"""Distributed training utilities for DDP."""

import os
import torch
import torch.distributed as dist
from typing import Optional


def setup_distributed(backend: str = 'nccl') -> tuple[int, int, int]:
    """
    Initialize distributed training.
    
    Returns:
        rank: Process rank
        world_size: Total number of processes
        local_rank: Local rank within node
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print("Not using distributed mode")
        return 0, 1, 0
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method='env://')
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if this is the main process."""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank() -> int:
    """Get current process rank."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Get total number of processes."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def reduce_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """
    Reduce tensor across all processes.
    
    Args:
        tensor: Tensor to reduce
        world_size: Number of processes
    Returns:
        Reduced tensor
    """
    if world_size == 1:
        return tensor
    
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def gather_tensor(tensor: torch.Tensor) -> Optional[list]:
    """
    Gather tensors from all processes to rank 0.
    
    Args:
        tensor: Tensor to gather
    Returns:
        List of tensors if rank 0, None otherwise
    """
    if not dist.is_initialized():
        return [tensor]
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    if rank == 0:
        gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    else:
        gather_list = None
    
    dist.gather(tensor, gather_list, dst=0)
    
    return gather_list


def barrier():
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


class DistributedLogger:
    """Logger that only prints on main process."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.is_main = is_main_process()
    
    def info(self, message: str):
        """Log info message."""
        if self.is_main and self.verbose:
            print(f"[INFO] {message}")
    
    def warning(self, message: str):
        """Log warning message."""
        if self.is_main and self.verbose:
            print(f"[WARNING] {message}")
    
    def error(self, message: str):
        """Log error message."""
        if self.is_main:
            print(f"[ERROR] {message}")
    
    def debug(self, message: str):
        """Log debug message."""
        if self.is_main and self.verbose:
            print(f"[DEBUG] {message}")


def save_on_main(obj, path: str):
    """Save object only on main process."""
    if is_main_process():
        torch.save(obj, path)


def print_on_main(message: str):
    """Print only on main process."""
    if is_main_process():
        print(message)

