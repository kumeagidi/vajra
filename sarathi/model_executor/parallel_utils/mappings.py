# Copyright 2023 The Sarathi team.
# Adapted from https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/mappings.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import torch

from sarathi.metrics.constants import OperationMetrics
from sarathi.metrics.cuda_timer import CudaTimer
from sarathi.model_executor.parallel_utils.parallel_state import (
    get_cache_model_parallel_group,
    get_cache_model_parallel_rank,
    get_cache_model_parallel_sub_group,
    get_cache_model_parallel_world_size,
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_next_rank,
    get_pipeline_model_parallel_prev_rank,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

from .utils import split_tensor_along_last_dim


def reduce_from_cache_model_parallel_region(input_):
    """All-reduce the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if get_cache_model_parallel_world_size() == 1:
        return input_

    # All-reduce.
    torch.distributed.all_reduce(input_, group=get_cache_model_parallel_group())

    return input_


def reduce_from_tensor_model_parallel_region(input_):
    """All-reduce the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if get_tensor_model_parallel_world_size() == 1:
        return input_

    # All-reduce.
    torch.distributed.all_reduce(input_, group=get_tensor_model_parallel_group())

    return input_


def reduce_from_cache_model_parallel_region(input_, group_ids):
    world_size = len(group_ids)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    group = get_cache_model_parallel_sub_group(group_ids)

    torch.distributed.all_reduce(input_, group=group)

    return input_


def scatter_to_tensor_model_parallel_region(input_):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""

    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along last dimension.
    input_list = split_tensor_along_last_dim(input_, world_size)

    # Note: torch.split does not create contiguous tensors by default.
    rank = get_tensor_model_parallel_rank()
    output = input_list[rank].contiguous()

    return output


def gather_from_group(input_, world_size, rank, group, concat_dim):
    # Bypass the function if we are using only 1 GPU.
    assert world_size > 1

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(tensor_list, input_, group=group)

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=concat_dim).contiguous()
    return output


def gather_from_tensor_model_parallel_region(input_):
    """Gather tensors and concatinate along the last dimension."""

    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = get_tensor_model_parallel_rank()
    group = get_tensor_model_parallel_group()

    return gather_from_group(input_, world_size, rank, group, last_dim)


def gather_from_cache_model_parallel_region(input_, group_ids):
    world_size = len(group_ids)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    rank = group_ids.index(get_cache_model_parallel_rank())
    group = get_cache_model_parallel_sub_group(group_ids)

    return gather_from_group(input_, world_size, rank, group, concat_dim=1)


def send_to_next_pipeline_stage(hidden_states: torch.tensor):
    """Send hidden states to the next pipeline stage."""
    # Bypass the function if we are using only 1 stage.
    if get_pipeline_model_parallel_group().size() == 1:
        return hidden_states

    with CudaTimer(OperationMetrics.NCCL_SEND):
        # Send the tensor.
        torch.distributed.send(
            tensor=hidden_states,
            dst=get_pipeline_model_parallel_next_rank(),
            group=get_pipeline_model_parallel_group(),
        )


def recv_from_last_pipeline_stage(hidden_states: torch.tensor):
    """Receive hidden states from the previous pipeline stage."""
    # Bypass the function if we are using only 1 stage.
    if get_pipeline_model_parallel_group().size() == 1:
        return hidden_states

    # Receive the tensor.
    with CudaTimer(OperationMetrics.NCCL_RECV):
        torch.distributed.recv(
            tensor=hidden_states,
            src=get_pipeline_model_parallel_prev_rank(),
            group=get_pipeline_model_parallel_group(),
        )

    return hidden_states
