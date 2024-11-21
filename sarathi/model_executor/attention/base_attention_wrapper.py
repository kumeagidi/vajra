from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import torch

from sarathi.config import (
    BaseSchedulerConfig,
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    WorkerConfig,
)
from sarathi.core.datatypes.sequence import SequenceMetadata
from sarathi.metrics.constants import OperationMetrics
from sarathi.metrics.cuda_timer import CudaTimer


class BaseAttentionWrapper(ABC):
    _inst = None

    def __init__(
        self,
        scheduler_config: BaseSchedulerConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        worker_config: WorkerConfig,
        block_size: int,
        device: torch.device,
    ):
        self.device = device
        self.num_q_heads = model_config.get_num_q_heads(parallel_config)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        self.head_dim = model_config.get_head_size()
        self.dtype = model_config.dtype
        self.block_size = block_size
        self.timers = {}

    """
    For a given model, all layers same the same AttentionWrapper instance.
    However, we cannot have a single timer for all layers because the same timer cannot be turned on/off dynamically.
    So, we have timers for each layer separately.
    """

    def get_timer(self, operation: OperationMetrics, layer_id: Optional[int] = None):
        if self.timers.get((operation, layer_id)) is None:
            self.timers[(operation, layer_id)] = CudaTimer(operation, layer_id)
        return self.timers.get((operation, layer_id))

    @abstractmethod
    def begin_forward(
        self,
        seq_metadata_list: List[SequenceMetadata],
    ) -> None:
        pass

    @classmethod
    def get_instance(cls):
        assert (
            cls._inst is not None
        ), "Instance not created. Call create_instance() first."
        return cls._inst

    @classmethod
    def create_instance(cls, *args, **kwargs):
        cls._inst = cls(*args, **kwargs)
        return cls._inst

    @abstractmethod
    def end_forward(self):
        pass

    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        softmax_scale: float = 1.0,
        layer_id: Optional[int] = None,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def set_cache_config(self, cache_config: CacheConfig) -> None:
        pass
