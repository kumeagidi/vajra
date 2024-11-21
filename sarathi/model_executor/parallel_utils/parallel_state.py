# Copyright 2023 The Sarathi team.
# Adapted from https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Model and cache parallel groups."""

from itertools import combinations

import torch

# Intra-layer model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP = None
# Inter-layer model parallel group that the current rank belongs to.
_PIPELINE_MODEL_PARALLEL_GROUP = None
# Intra-layer cache model parallel group that the current rank belongs to.
_CACHE_MODEL_PARALLEL_GROUP = None
_CACHE_MODEL_PARALLEL_SUB_GROUPS = None
_CACHE_MODEL_PARALLEL_SUB_GROUPS_GLOBAL_RANKS = None

# A list of global ranks for each pipeline group to ease calculation of the source
# rank when broadcasting from the first or last pipeline stage.
_PIPELINE_GLOBAL_RANKS = None

_PROCESS_GROUP_WRAPPER = None

from sarathi._model_executor_C.model_executor import ProcessGroupWrapper


def get_power_set(world_size):
    power_set = []
    for i in range(1, world_size + 1):
        power_set.extend([tuple(sorted(x)) for x in combinations(range(world_size), i)])
    return power_set


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    cache_model_parallel_size: int = 1,
) -> None:
    """
    Initialize model cache parallel groups.

    Arguments:
        tensor_model_parallel_size: number of GPUs used for tensor model parallelism.
        pipeline_model_parallel_size: number of GPUs used for pipeline model parallelism.
        cache_model_parallel_size: number of GPUs used for cache model parallelism.

    Let's say we have a total of 16 GPUs denoted by g0 ... g15 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 8 tensor model-parallel groups, 4 pipeline model-parallel groups
    and 8 cache-parallel groups as:
        8 cache_parallel groups:
            [g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]
        8 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]
        4 pipeline model-parallel groups:
            [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()

    assert (
        world_size
        == tensor_model_parallel_size
        * pipeline_model_parallel_size
        * cache_model_parallel_size
    )

    num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size

    rank = torch.distributed.get_rank()

    # Build the cache-parallel groups.
    global _CACHE_MODEL_PARALLEL_GROUP
    global _CACHE_MODEL_PARALLEL_SUB_GROUPS
    global _CACHE_MODEL_PARALLEL_SUB_GROUPS_GLOBAL_RANKS

    assert (
        _CACHE_MODEL_PARALLEL_GROUP is None
    ), "cache parallel group is already initialized"
    assert (
        _CACHE_MODEL_PARALLEL_SUB_GROUPS is None
    ), "cache parallel sub groups are already initialized"

    _CACHE_MODEL_PARALLEL_SUB_GROUPS = {}
    _CACHE_MODEL_PARALLEL_SUB_GROUPS_GLOBAL_RANKS = {}

    for i in range(pipeline_model_parallel_size):
        start_rank = i * num_pipeline_model_parallel_groups
        end_rank = (i + 1) * num_pipeline_model_parallel_groups
        for j in range(tensor_model_parallel_size):
            ranks = range(start_rank + j, end_rank, tensor_model_parallel_size)
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                _CACHE_MODEL_PARALLEL_GROUP = group

            # create all subgroups
            # power_set = get_power_set(len(ranks))
            # TODO(amey): Creating power set doesn't scale so we will just use linearly increasing set
            power_set = [tuple(range(i)) for i in range(1, len(ranks) + 1)]
            for rank_indices in power_set:
                subgroup_ranks = [ranks[i] for i in rank_indices]
                subgroup = torch.distributed.new_group(subgroup_ranks)
                if rank in subgroup_ranks:
                    _CACHE_MODEL_PARALLEL_SUB_GROUPS[frozenset(rank_indices)] = subgroup
                    _CACHE_MODEL_PARALLEL_SUB_GROUPS_GLOBAL_RANKS[subgroup] = (
                        subgroup_ranks
                    )

    # Build the tensor model-parallel groups.
    global _TENSOR_MODEL_PARALLEL_GROUP
    assert (
        _TENSOR_MODEL_PARALLEL_GROUP is None
    ), "tensor model parallel group is already initialized"
    for i in range(num_tensor_model_parallel_groups):
        ranks = range(
            i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size
        )
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP = group

    # Build the pipeline model-parallel groups and embedding groups
    # (first and last rank in each pipeline model-parallel group).
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _PIPELINE_GLOBAL_RANKS
    assert (
        _PIPELINE_MODEL_PARALLEL_GROUP is None
    ), "pipeline model parallel group is already initialized"

    for i in range(num_pipeline_model_parallel_groups):
        ranks = range(i, world_size, num_pipeline_model_parallel_groups)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _PIPELINE_MODEL_PARALLEL_GROUP = group
            _PIPELINE_GLOBAL_RANKS = ranks

    global _PROCESS_GROUP_WRAPPER
    _PROCESS_GROUP_WRAPPER = ProcessGroupWrapper(
        _TENSOR_MODEL_PARALLEL_GROUP,
        _PIPELINE_MODEL_PARALLEL_GROUP,
        _CACHE_MODEL_PARALLEL_SUB_GROUPS,
    )


def model_parallel_is_initialized():
    """Check if model and cache parallel groups are initialized."""
    if (
        _TENSOR_MODEL_PARALLEL_GROUP is None
        or _PIPELINE_MODEL_PARALLEL_GROUP is None
        or _CACHE_MODEL_PARALLEL_GROUP is None
    ):
        return False
    return True


def get_tensor_model_parallel_group():
    """Get the tensor model parallel group the caller rank belongs to."""
    assert (
        _TENSOR_MODEL_PARALLEL_GROUP is not None
    ), "intra_layer_model parallel group is not initialized"
    return _TENSOR_MODEL_PARALLEL_GROUP


def get_pipeline_model_parallel_group():
    """Get the pipeline model parallel group the caller rank belongs to."""
    assert (
        _PIPELINE_MODEL_PARALLEL_GROUP is not None
    ), "pipeline_model parallel group is not initialized"
    return _PIPELINE_MODEL_PARALLEL_GROUP


def get_cache_model_parallel_group():
    """Get the cache parallel group the caller rank belongs to."""
    assert (
        _CACHE_MODEL_PARALLEL_GROUP is not None
    ), "cache parallel group is not initialized"
    return _CACHE_MODEL_PARALLEL_GROUP


def get_cache_model_parallel_sub_group(rank_indices):
    """Get the cache parallel group the caller rank belongs to."""
    assert (
        _CACHE_MODEL_PARALLEL_SUB_GROUPS is not None
    ), "cache parallel sub groups are not initialized"
    return _CACHE_MODEL_PARALLEL_SUB_GROUPS[frozenset(rank_indices)]


def get_cache_model_parallel_sub_group_global_ranks(group):
    """Get the cache parallel group the caller rank belongs to."""
    assert (
        _CACHE_MODEL_PARALLEL_SUB_GROUPS_GLOBAL_RANKS is not None
    ), "cache parallel sub groups global ranks are not initialized"
    return _CACHE_MODEL_PARALLEL_SUB_GROUPS_GLOBAL_RANKS[group]


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    return torch.distributed.get_world_size(group=get_tensor_model_parallel_group())


def get_pipeline_model_parallel_world_size():
    """Return world size for the pipeline model parallel group."""
    return torch.distributed.get_world_size(group=get_pipeline_model_parallel_group())


def get_cache_model_parallel_world_size():
    """Return world size for the pipeline model parallel group."""
    return torch.distributed.get_world_size(group=get_cache_model_parallel_group())


def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group())


def get_pipeline_model_parallel_rank():
    """Return my rank for the pipeline model parallel group."""
    return torch.distributed.get_rank(group=get_pipeline_model_parallel_group())


def get_cache_model_parallel_rank():
    """Return my rank for the cache parallel group."""
    return torch.distributed.get_rank(group=get_cache_model_parallel_group())


def is_pipeline_first_stage():
    """Return True if in the first pipeline model-parallel stage, False otherwise."""
    return get_pipeline_model_parallel_rank() == 0


def is_pipeline_last_stage():
    """Return True if in the last pipeline model-parallel stage, False otherwise."""
    return get_pipeline_model_parallel_rank() == (
        get_pipeline_model_parallel_world_size() - 1
    )


def get_pipeline_model_parallel_next_rank():
    """Return the global rank that follows the caller in the pipeline"""
    assert (
        _PIPELINE_GLOBAL_RANKS is not None
    ), "Pipeline parallel group is not initialized"
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline + 1) % world_size]


def get_pipeline_model_parallel_prev_rank():
    """Return the global rank that preceeds the caller in the pipeline"""
    assert (
        _PIPELINE_GLOBAL_RANKS is not None
    ), "Pipeline parallel group is not initialized"
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline - 1) % world_size]


def get_rank():
    return torch.distributed.get_rank()


def get_process_group_wrapper():
    return _PROCESS_GROUP_WRAPPER


def destroy_model_parallel():
    """Set the groups to none."""
    global _TENSOR_MODEL_PARALLEL_GROUP
    _TENSOR_MODEL_PARALLEL_GROUP = None
    global _PIPELINE_MODEL_PARALLEL_GROUP
    _PIPELINE_MODEL_PARALLEL_GROUP = None
    global _CACHE_MODEL_PARALLEL_GROUP
    _CACHE_MODEL_PARALLEL_GROUP = None
