from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from sarathi.config import SystemConfig
from sarathi.core.datatypes.scheduler_output import SchedulerOutput
from sarathi.core.datatypes.sequence import (
    MnemosyneSequenceMetadata,
    MnemosyneSequenceScheduleMetadata,
    SamplerOutputs,
    Sequence,
    SequenceScheduleMetadata,
)
from sarathi.core.sequence_manager.worker_sequence_manager import WorkerSequenceManager
from sarathi.logger import init_logger
from sarathi.model_executor.parallel_utils.parallel_state import (
    get_cache_model_parallel_rank,
    get_cache_model_parallel_world_size,
    get_rank,
)
from sarathi.types import SchedulerType
from sarathi.utils.threading_utils import synchronized

logger = init_logger(__name__)


class MnemosyneWorkerSequenceManager(WorkerSequenceManager):

    def __init__(
        self,
        config: SystemConfig,
    ):
        super().__init__(config)
        self.rank = get_rank()
        self.group_id = get_cache_model_parallel_rank()

        assert config.scheduler_config.get_type() == SchedulerType.MNEMOSYNE

        if get_cache_model_parallel_world_size() == 1:
            self.max_seq_cache_occupancy = config.model_config.max_model_len
        else:
            self.max_seq_cache_occupancy = (
                config.parallel_config.max_seq_cache_occupancy
            )

        self.seq_num_processed_tokens_map: Dict[int, int] = defaultdict(int)

    def _on_seq_scheduled(
        self,
        seq_sched_metadata: MnemosyneSequenceScheduleMetadata,
    ) -> None:
        assert seq_sched_metadata.seq_id in self.seq_map
        self._resume_seq(seq_sched_metadata.seq_id)

        seq = self.seq_map[seq_sched_metadata.seq_id]
        num_total_blocks = seq_sched_metadata.group_block_mapping[self.group_id]
        logger.debug(
            f"Allocating {num_total_blocks} blocks for seq {seq.seq_id} in group {self.group_id}"
        )
        self.block_manager.allocate_delta(seq, num_total_blocks)

    def _update_seq_num_processed_tokens_map(
        self,
        seq: Sequence,
        seq_sched_metadata: MnemosyneSequenceScheduleMetadata,
    ) -> None:
        if self.group_id != seq_sched_metadata.active_group_ids[-1]:
            return

        if not seq.prompt_stage_processing_finished:
            self.seq_num_processed_tokens_map[
                seq.seq_id
            ] += seq_sched_metadata.prompt_chunk_len
            assert (
                self.seq_num_processed_tokens_map[seq.seq_id]
                <= self.max_seq_cache_occupancy
            ), (
                f"seq_id: {seq.seq_id}, "
                f"num_processed_tokens: {self.seq_num_processed_tokens_map[seq.seq_id]}, "
                f"max_seq_cache_occupancy: {self.max_seq_cache_occupancy}"
            )
            if (
                seq.get_num_prompt_tokens_stage_processed()
                + seq_sched_metadata.prompt_chunk_len
                == seq.get_prompt_len()
            ):
                self.seq_num_processed_tokens_map[seq.seq_id] += 1
        else:
            self.seq_num_processed_tokens_map[seq.seq_id] += 1

    @synchronized
    def on_stage_completed(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        """
        This gets called only when pipeline parallel is enabled.
        The engine calls this when the first pipeline stage completed (engine-side) + each worker will
        call this method separately.
        """

        if not self.enable_sequence_pipeline_parallel:
            return

        for scheduled_seq_metadata in scheduler_output.scheduled_seq_metadata_list:
            seq = self.seq_map[scheduled_seq_metadata.seq_id]
            assert not seq.is_finished()

            if seq.is_waiting_preempted():
                # seq is preempted
                # this can happen with pipeline parallel -- if the system
                # runs out of memory, it will preempt the last arrived request
                # this request might still be executing when the next stage scheduling
                # triggers the preemption
                continue

            if seq.prompt_stage_processing_finished:
                continue

            self._update_seq_num_processed_tokens_map(seq, scheduled_seq_metadata)

            seq.update_prompt_tokens_stage_processed(
                scheduled_seq_metadata.prompt_chunk_len
            )

            if (
                self.group_id in scheduled_seq_metadata.active_group_ids
                and not seq.prompt_stage_processing_finished
            ):
                self._pause_seq(scheduled_seq_metadata.seq_id)

    def on_step_completed(
        self,
        scheduled_seq_metadata_list: List[SequenceScheduleMetadata],
        sampler_outputs: Optional[SamplerOutputs],
    ) -> None:
        filtered_seq_metadata_list = []

        for seq_sched_metadata in scheduled_seq_metadata_list:
            seq = self.seq_map[seq_sched_metadata.seq_id]

            assert not seq.is_finished()

            if (
                not self.group_id in seq_sched_metadata.active_group_ids
                and not seq.prompt_processing_finished
            ):
                if not self.enable_sequence_pipeline_parallel:
                    # In case of sequence pipeline parallel, the stage token cursor is
                    # already updated in the on_stage_completed method
                    seq.update_prompt_tokens_stage_processed(
                        seq_sched_metadata.prompt_chunk_len
                    )
                seq.update_prompt_tokens_processed(seq_sched_metadata.prompt_chunk_len)
                continue

            if (
                not self.enable_sequence_pipeline_parallel
                or seq.prompt_processing_finished
            ):
                # in case of sequence pipeline parallel, the stage token cursor is
                # already updated in the on_stage_completed method
                self._update_seq_num_processed_tokens_map(seq, seq_sched_metadata)

            filtered_seq_metadata_list.append(seq_sched_metadata)

        super().on_step_completed(filtered_seq_metadata_list, sampler_outputs)

    @synchronized
    def on_schedule(
        self,
        scheduler_output: SchedulerOutput,
    ) -> Tuple[List[Sequence], List[MnemosyneSequenceMetadata]]:
        ignored_seqs: List[Sequence] = []
        for seq_id in scheduler_output.ignored_seq_ids:
            assert seq_id in self.seq_map
            seq = self.seq_map[seq_id]
            ignored_seqs.append(seq)
            self._free_seq(seq_id)

        for seq_id in scheduler_output.preempted_seq_ids:
            self._preempt_seq(seq_id)

        seq_metadata_list: List[MnemosyneSequenceMetadata] = []
        multi_group_seq_metadata: Optional[MnemosyneSequenceMetadata] = None

        for seq_sched_metadata in scheduler_output.scheduled_seq_metadata_list:
            assert seq_sched_metadata.seq_id in self.seq_map, (
                f"seq_id {seq_sched_metadata.seq_id} not found in seq_map. "
                f"seq_map: {self.seq_map} for rank {self.rank}"
            )

            seq = self.seq_map[seq_sched_metadata.seq_id]

            if not self.group_id in seq_sched_metadata.active_group_ids:
                continue

            is_multi_group_seq = len(seq_sched_metadata.active_group_ids) > 1

            self._on_seq_scheduled(seq_sched_metadata)

            kv_cache_len = self.seq_num_processed_tokens_map[seq.seq_id]
            save_kv_cache = self.group_id == seq_sched_metadata.active_group_ids[-1]

            seq_metadata = MnemosyneSequenceMetadata(
                # seq_sched_metadata.schedule_id,
                seq,
                self._get_block_table(seq),
                seq_sched_metadata.num_prompt_tokens,
                kv_cache_len,
                save_kv_cache,
                seq_sched_metadata.active_group_ids,
            )

            if is_multi_group_seq:
                assert (
                    multi_group_seq_metadata is None
                ), "Currently only one multi-group sequence can run in a single step."
                multi_group_seq_metadata = seq_metadata
            else:
                seq_metadata_list.append(seq_metadata)

        if multi_group_seq_metadata is not None:
            if multi_group_seq_metadata.is_prompt:
                # add at the beginning
                seq_metadata_list.insert(0, multi_group_seq_metadata)
            else:
                seq_metadata_list.append(multi_group_seq_metadata)

        return ignored_seqs, seq_metadata_list
