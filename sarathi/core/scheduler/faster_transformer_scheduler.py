import time
from typing import List
from queue import PriorityQueue

from sarathi.config import (
    CacheConfig,
    FasterTransformerSchedulerConfig,
    ModelConfig,
    ParallelConfig,
)
from sarathi.core.block_space_manager.faster_transformer_block_space_manager import (
    FasterTransformerBlockSpaceManager,
)
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import SequenceScheduleMetadata
from sarathi.core.sequence_manager.engine_sequence_manager import EngineSequenceManager
from sarathi.core.scheduler.base_scheduler import BaseScheduler
from sarathi.logger import init_logger
from sarathi.metrics.metrics_store import MetricsStore


logger = init_logger(__name__)


class FasterTransformerScheduler(BaseScheduler):

    def __init__(
        self,
        model_config: ModelConfig,
        scheduler_config: FasterTransformerSchedulerConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        waiting_queue : PriorityQueue,
        replica_seq_manager : EngineSequenceManager,
        metric_store : MetricsStore,
    ) -> None:
        super().__init__(model_config, scheduler_config, cache_config, parallel_config, waiting_queue, replica_seq_manager, metric_store)

    def get_block_space_manager_class(self):
        return FasterTransformerBlockSpaceManager

    def _schedule(self) -> SchedulerOutputs:
        scheduled_seq_metadata_list: List[SequenceScheduleMetadata] = []

        now = time.monotonic()

        for seq in self.running:
            if not seq.is_paused():
                continue

            assert seq.prompt_stage_processing_finished
            scheduled_seq_metadata_list.append(
                SequenceScheduleMetadata.from_sequence(seq)
            )

        if scheduled_seq_metadata_list:
            return SchedulerOutputs(
                id=self._iteration_id,
                ignored_seq_ids=[],
                preempted_seq_ids=[],
                scheduled_seq_metadata_list=scheduled_seq_metadata_list,
            )

        ignored_seq_ids: List[int] = []
        # Optimization: We do not sort the waiting queue since the preempted
        # sequences are added to the front and the new sequences
        # are added to the back.
        while self.waiting.qsize() > 0:
            seq_wrapped = self.waiting.queue[0]
            seq = seq_wrapped.seq
            # This is required to handle benchmarking where
            # we set request arrival time ahead of time
            if seq.arrival_time > now:
                break

            if not self._check_request_prompt_length(seq):
                ignored_seq_ids.append(seq.seq_id)
                continue

            # If the sequence cannot be allocated, stop.
            if not self.block_manager.can_allocate(seq):
                break

            if len(self.running) + 1 > self.scheduler_config.max_num_seqs:
                break

            seq_wrapped = self.waiting.get()
            seq = seq_wrapped.seq
            self._allocate(seq)
            self.running.append(seq)
            scheduled_seq_metadata_list.append(
                SequenceScheduleMetadata.from_sequence(seq)
            )

        scheduler_outputs = SchedulerOutputs(
            id=self._iteration_id,
            ignored_seq_ids=ignored_seq_ids,
            preempted_seq_ids=[],
            scheduled_seq_metadata_list=scheduled_seq_metadata_list,
        )
        return scheduler_outputs