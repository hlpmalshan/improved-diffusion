# """
# Helpers for distributed training.
# """

# import io
# import os
# import socket

# import blobfile as bf
# from mpi4py import MPI
# import torch as th
# import torch.distributed as dist

# # Change this to reflect your cluster layout.
# # The GPU for a given rank is (rank % GPUS_PER_NODE).
# GPUS_PER_NODE = 8

# SETUP_RETRY_COUNT = 3


# def setup_dist():
#     """
#     Setup a distributed process group.
#     """
#     if dist.is_initialized():
#         return

#     comm = MPI.COMM_WORLD
#     backend = "gloo" if not th.cuda.is_available() else "nccl"

#     if backend == "gloo":
#         hostname = "localhost"
#     else:
#         hostname = socket.gethostbyname(socket.getfqdn())
#     os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
#     os.environ["RANK"] = str(comm.rank)
#     os.environ["WORLD_SIZE"] = str(comm.size)

#     port = comm.bcast(_find_free_port(), root=0)
#     os.environ["MASTER_PORT"] = str(port)
#     dist.init_process_group(backend=backend, init_method="env://")


# def dev():
#     """
#     Get the device to use for torch.distributed.
#     """
#     if th.cuda.is_available():
#         return th.device(f"cuda:{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}")
#     return th.device("cpu")


# def load_state_dict(path, **kwargs):
#     """
#     Load a PyTorch file without redundant fetches across MPI ranks.
#     """
#     if MPI.COMM_WORLD.Get_rank() == 0:
#         with bf.BlobFile(path, "rb") as f:
#             data = f.read()
#     else:
#         data = None
#     data = MPI.COMM_WORLD.bcast(data)
#     return th.load(io.BytesIO(data), **kwargs)


# def sync_params(params):
#     """
#     Synchronize a sequence of Tensors across ranks from rank 0.
#     """
#     for p in params:
#         with th.no_grad():
#             dist.broadcast(p, 0)


# def _find_free_port():
#     try:
#         s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         s.bind(("", 0))
#         s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#         return s.getsockname()[1]
#     finally:
#         s.close()


"""
Helpers for distributed training without MPI.
Use with: torchrun ... train.py
"""

import io
import os
import socket
from datetime import timedelta
from typing import Iterable

import blobfile as bf
import torch as th
import torch.distributed as dist


# ---------- core setup ----------

def setup_dist(timeout_sec: int = 600) -> None:
    """
    Initialize torch.distributed from torchrun-provided env vars.
    Works single-node or multi-node. No MPI required.
    """
    if dist.is_initialized():
        return

    # torchrun sets: RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    backend = "nccl" if th.cuda.is_available() else "gloo"

    # If you insist on standalone without torchrun, we can self-assign a port.
    # For normal use with torchrun these are already set.
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = _get_hostname_ip()
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = str(_find_free_port())

    dist.init_process_group(
        backend=backend,
        init_method="env://",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=timeout_sec),
    )

    if th.cuda.is_available():
        th.cuda.set_device(local_rank)


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def dev() -> th.device:
    """
    Device for this rank. Mirrors your original `dev()` but without MPI.
    """
    if th.cuda.is_available():
        return th.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
    return th.device("cpu")


# ---------- utilities ----------

def sync_params(params: Iterable[th.Tensor], src: int = 0) -> None:
    """
    Broadcast parameters from rank 0 to all ranks.
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return
    for p in params:
        with th.no_grad():
            dist.broadcast(p.data, src)


def barrier() -> None:
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier()


def load_state_dict_dist(path: str, map_location=None, src: int = 0, use_broadcast=True):
    """
    Rank-0 reads a checkpoint once, then broadcasts bytes to others.
    Avoids NFS storms. No MPI.
    """
    if not dist.is_initialized() or dist.get_world_size() == 1 or not use_broadcast:
        with bf.BlobFile(path, "rb") as f:
            return th.load(f, map_location=map_location)

    rank = dist.get_rank()
    if rank == src:
        with bf.BlobFile(path, "rb") as f:
            raw = f.read()
        length = th.tensor([len(raw)], dtype=th.long, device=dev())
    else:
        raw = None
        length = th.tensor([0], dtype=th.long, device=dev())

    # broadcast length
    dist.broadcast(length, src)
    n = int(length.item())

    # allocate and broadcast payload
    if rank != src:
        buf = th.empty(n, dtype=th.uint8, device=dev())
    else:
        buf = th.as_tensor(bytearray(raw), dtype=th.uint8, device=dev())

    dist.broadcast(buf, src)

    # move to CPU BytesIO for torch.load
    if rank != src:
        raw_bytes = bytes(buf.cpu().tolist())
    else:
        raw_bytes = raw

    return th.load(io.BytesIO(raw_bytes), map_location=map_location)


# ---------- internals ----------

def _find_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    port = s.getsockname()[1]
    s.close()
    return port


def _get_hostname_ip() -> str:
    return socket.gethostbyname(socket.getfqdn())
