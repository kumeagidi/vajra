from abc import ABC, abstractmethod
from typing import List
from queue import PriorityQueue

from sarathi.config import BaseSchedulerConfig, CacheConfig, ModelConfig, ParallelConfig
from sarathi.core.block_space_manager.block_space_manager_registry import (
    BlockSpaceManagerRegistry,
)
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import Sequence, SequenceStatus, SequenceWithPriority
from sarathi.core.sequence_manager.engine_sequence_manager import EngineSequenceManager
from sarathi.core.policy import PolicyFactory
from sarathi.logger import init_logger
from sarathi.metrics.metrics_store import MetricsStore
from sarathi.utils.threading_utils import synchronized




logger = init_logger(__name__)


class BaseScheduler(ABC):

    def __init__(
        self,
        model_config: ModelConfig,
        scheduler_config: BaseSchedulerConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        waiting_queue : PriorityQueue,
        replica_seq_manager : EngineSequenceManager,
        metric_store : MetricsStore,
    ) -> None:
        self.metrics_store = MetricsStore.get_instance()
        self.model_config = model_config
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.parallel_config = parallel_config

        # we maintain this just for logging purposes
        self._iteration_id = -1

        # Instantiate the scheduling policy.
        self.policy = PolicyFactory.get_policy(policy_name="fcfs")
        # Create the block space manager.
        self.block_manager = BlockSpaceManagerRegistry.get(
            scheduler_config.get_type(),
            cache_config.block_size,
            cache_config.num_gpu_blocks,
            model_config.max_model_len,
        )
        self.prompt_limit = model_config.max_model_len
        self.replica_seq_manager = replica_seq_manager
        self.new_seqs: List[Sequence] = []
        self.metrics_store = metric_store
        self.seq_seen = set()

        # number of running batches should be less than or equal to the number of pipeline stages
        self.num_running_batches = 0

        # TODO(zhuohan): Use deque instead of list for better performance.
        # Sequence groups in the WAITING state.
        self.waiting : PriorityQueue = PriorityQueue()
        # self.waiting : PriorityQueue = waiting_queue
        # Sequence groups in the RUNNING state.
        self.running: List[Sequence] = []

    def reset_state(self) -> None:
        self._iteration_id = -1

    def add_seq(self, seq: Sequence) -> None:
        # Add sequence groups to the waiting queue. 
        wrapped_seq = SequenceWithPriority(seq.arrived_at, seq)

        self.waiting.put(wrapped_seq)

    def has_unfinished_seqs(self) -> bool:
        return self.waiting.qsize() > 0 or self.running

    def get_num_unfinished_seqs(self) -> int:
        return self.waiting.qsize() + len(self.running)

    @abstractmethod
    def _schedule(self) -> SchedulerOutputs:
        pass

    @synchronized
    def add_to_new_seqs(self, seq: Sequence) -> None:
        self.new_seqs.append(seq)

    @synchronized
    def get_new_seqs(
        self,
    ) -> List[Sequence]:
        new_seqs = self.new_seqs
        self.new_seqs = []
        return new_seqs
    
    @synchronized
    def add_seq_to_seq_manager(self, seq: Sequence) -> None:
        self.replica_seq_manager.add_seq(seq)

    def schedule(self) -> SchedulerOutputs:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running and self.waiting.
        self._iteration_id += 1

        if self.num_running_batches >= self.parallel_config.pipeline_parallel_size:
            return SchedulerOutputs(
                self._iteration_id,
                ignored_seq_ids=[],
                preempted_seq_ids=[],
                scheduled_seq_metadata_list=[],
            )
        
        scheduler_outputs = self._schedule()

        if not scheduler_outputs.is_empty():
            self.num_running_batches += 1

        return scheduler_outputs

    def free_finished_seqs(self) -> None:
        for seq in self.running:
            if seq.is_finished():
                self._free_seq(seq)
        self.running = [seq for seq in self.running if not seq.is_finished()]

    def on_step_completed(self) -> None:
        self.free_finished_seqs()
        self.num_running_batches -= 1

    def _allocate(self, seq: Sequence) -> None:
        self.block_manager.allocate(seq)

    def _free_seq(self, seq: Sequence) -> None:
        self.block_manager.free(seq)

    def _append_slot(
        self,
        seq: Sequence,
    ) -> None:
        assert seq.is_executing()
        self.block_manager.append_slot(seq)

    def _preempt(
        self,
        seq: Sequence,
    ) -> None:
        assert seq.is_executing()
        self._free_seq(seq)

        wrapped_seq = SequenceWithPriority(seq.arrived_at, seq)
        
        self.waiting.put(wrapped_seq)

    def _check_request_prompt_length(self, seq: Sequence) -> bool:
        if seq.get_len() > self.prompt_limit:
            logger.warning(
                f"Input prompt ({seq.get_len()} tokens) is too long"
                f" and exceeds limit of {self.prompt_limit}"
            )
            seq.set_status(SequenceStatus.FINISHED_IGNORED)
            self.waiting.get(block=False)
            return False

        return True