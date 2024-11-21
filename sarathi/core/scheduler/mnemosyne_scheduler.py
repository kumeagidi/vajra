import time
from collections import OrderedDict, defaultdict
from math import ceil
from typing import Dict, List, Optional

import numpy as np
from vidur.config import RandomForrestExecutionTimePredictorConfig, ReplicaConfig
from vidur.entities import Batch as VidurBatch
from vidur.entities import Request as VidurRequest
from vidur.execution_time_predictor import ExecutionTimePredictorRegistry

from sarathi.config import (
    CacheConfig,
    MnemosyneSchedulerConfig,
    ModelConfig,
    ParallelConfig,
)
from sarathi.core.block_space_manager.block_space_manager_registry import (
    BlockSpaceManagerRegistry,
)
from sarathi.core.block_space_manager.mnemosyne_block_space_manager import (
    MnemosyneBlockSpaceManager,
)
from sarathi.core.datatypes.scheduler_output import SchedulerOutput
from sarathi.core.datatypes.sequence import MnemosyneSequenceScheduleMetadata, Sequence
from sarathi.core.scheduler.sarathi_scheduler import SarathiScheduler
from sarathi.logger import init_logger

logger = init_logger(__name__)

PREDICTION_MAX_CHUNK_SIZE = 4 * 1024
MAX_TOKENS_PER_REQUEST = 8 * 1024 * 1024
PREDICTION_MAX_BATCH_SIZE = 128
PREDICTION_DEVICE = "h100"
PREDICTION_NETWORK_DEVICE = "h100_dgx"
KV_CACHE_PREDICTION_GRANULARITY = 512
BATCH_TIME_ERROR_MARGIN = 0.01
CP_OVERHEAD = 0.1


class MnemosyneScheduler(SarathiScheduler):

    def __init__(
        self,
        model_config: ModelConfig,
        scheduler_config: MnemosyneSchedulerConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
    ) -> None:
        super().__init__(model_config, scheduler_config, cache_config, parallel_config)

        # TODO(amey): we only support greedy sampling for long context requests for now

        if scheduler_config.enable_predictive_schedule:
            assert scheduler_config.predictive_schedule_max_chunk_size % 32 == 0
            assert (
                PREDICTION_MAX_CHUNK_SIZE
                >= scheduler_config.predictive_schedule_max_chunk_size
            )
            assert MAX_TOKENS_PER_REQUEST >= model_config.max_model_len
            assert PREDICTION_MAX_BATCH_SIZE >= scheduler_config.max_num_seqs

            execution_time_predictor_config = RandomForrestExecutionTimePredictorConfig(
                prediction_max_prefill_chunk_size=PREDICTION_MAX_CHUNK_SIZE,
                prediction_max_tokens_per_request=MAX_TOKENS_PER_REQUEST,
                prediction_max_batch_size=PREDICTION_MAX_BATCH_SIZE,
                kv_cache_prediction_granularity=KV_CACHE_PREDICTION_GRANULARITY,
            )

            replica_config = ReplicaConfig(
                model_name=model_config.model,
                num_pipeline_stages=1,  # doesn't matter for now
                tensor_parallel_size=parallel_config.tensor_parallel_size,
                device=PREDICTION_DEVICE,
                network_device=PREDICTION_NETWORK_DEVICE,
                block_size=cache_config.block_size,
            )

            self.execution_time_predictor = ExecutionTimePredictorRegistry.get(
                execution_time_predictor_config.get_type(),
                predictor_config=execution_time_predictor_config,
                replica_config=replica_config,
            )

        self.cache_parallel_size = parallel_config.cache_parallel_size

        if self.cache_parallel_size == 1:
            self.max_seq_cache_occupancy = model_config.max_model_len
        else:
            assert (
                self.parallel_config.max_seq_cache_occupancy is not None
            ), "max_seq_cache_occupancy must be specified when using cache parallelism."

            # TODO(amey): add token schedule for multi-node case and use that to compute exact schedule
            assert parallel_config.max_seq_cache_occupancy > cache_config.block_size
            assert (
                parallel_config.max_seq_cache_occupancy % cache_config.block_size == 0
            )

            self.max_seq_cache_occupancy = parallel_config.max_seq_cache_occupancy

        self.max_seq_cache_occupancy_blocks = ceil(
            self.max_seq_cache_occupancy / cache_config.block_size
        )
        self.block_managers_map: Dict[int, MnemosyneBlockSpaceManager] = {}

        for i in range(self.cache_parallel_size):
            self.block_managers_map[i] = BlockSpaceManagerRegistry.get(
                scheduler_config.get_type(),
                cache_config.block_size,
                cache_config.num_gpu_blocks,
                model_config.max_model_len,
            )

        self.seq_cache_worker_block_counter: Dict[int, OrderedDict[int, int]] = (
            defaultdict(OrderedDict)
        )
        self.seq_block_counter: Dict[int, int] = {}

        self.last_batch_num_prefill_tokens: Optional[int] = None

        self.skipped_steps = 0

    def reset_state(self) -> None:
        super().reset_state()
        self.last_batch_num_prefill_tokens = None
        self.skipped_steps = 0

    def _get_seq_next_num_prefill_tokens_predictive_schedule(
        self, seq: Sequence
    ) -> int:
        # TODO(amey): expand the scope of this and consider other requests
        # we assuming that this is only request in the batch for simplicity
        if self.last_batch_num_prefill_tokens is None:
            chunk_size = self.scheduler_config.predictive_schedule_max_chunk_size
        else:
            chunk_size = self.last_batch_num_prefill_tokens

        target_latency = (
            self.scheduler_config.predictive_schedule_target_batch_execution_latency
        )

        is_multi_group_seq = len(self.seq_cache_worker_block_counter[seq.seq_id]) > 1

        if is_multi_group_seq:
            num_processed_tokens = self.max_seq_cache_occupancy
            cp_overhead = 1 + CP_OVERHEAD
        else:
            num_processed_tokens = seq.get_num_prompt_tokens_stage_processed()
            cp_overhead = 1

        vidur_request = VidurRequest(
            arrived_at=0,
            num_prefill_tokens=seq.get_prompt_len(),
            num_decode_tokens=1,
            num_processed_tokens=num_processed_tokens,
        )
        vidur_batch = VidurBatch(
            replica_id=0,
            requests=[vidur_request],
            num_tokens=[chunk_size],
        )

        execution_time = (
            self.execution_time_predictor.get_execution_time(
                vidur_batch, pipeline_stage=0
            ).total_time
            * cp_overhead
        )

        # reduce the chunk size by 32 till we find a chunk size that fits in the batch target latency
        while execution_time > target_latency + BATCH_TIME_ERROR_MARGIN:
            chunk_size -= 32

            if chunk_size < 32:
                chunk_size = 32
                break

            vidur_batch = VidurBatch(
                replica_id=0,
                requests=[vidur_request],
                num_tokens=[chunk_size],
            )
            execution_time = (
                self.execution_time_predictor.get_execution_time(
                    vidur_batch, pipeline_stage=0
                ).total_time
                * cp_overhead
            )

        self.last_batch_num_prefill_tokens = chunk_size

        return chunk_size

    def _get_seq_next_num_prefill_tokens(
        self, seq: Sequence, num_batched_tokens: int
    ) -> int:
        assert not seq.is_finished()

        if self.scheduler_config.enable_predictive_schedule:
            chunk_size = self._get_seq_next_num_prefill_tokens_predictive_schedule(seq)
        else:
            chunk_size = self.chunk_size

        next_num_tokens = min(
            seq.get_prompt_len() - seq.get_num_prompt_tokens_stage_processed(),
            chunk_size - num_batched_tokens,
        )

        # if it is the last chunk for the cp group, then we need to make sure that we
        # don't exceed the max_seq_cache_occupancy
        if self.cache_parallel_size > 1:
            num_active_groups = np.ceil(
                (seq.get_num_prompt_tokens_stage_processed() + 1)
                / self.max_seq_cache_occupancy
            )
            next_group_occupancy_limit = int(
                num_active_groups * self.max_seq_cache_occupancy
            )

            if (
                seq.get_num_prompt_tokens_stage_processed() + next_num_tokens
                > next_group_occupancy_limit
            ):
                next_num_tokens = (
                    next_group_occupancy_limit
                    - seq.get_num_prompt_tokens_stage_processed()
                )

        return next_num_tokens

    def get_num_initial_blocks(self, seq: Sequence) -> int:
        return len(seq.logical_token_blocks)

    def _allocate(self, seq: Sequence) -> bool:
        num_blocks = self.get_num_initial_blocks(seq)

        if num_blocks < self.max_seq_cache_occupancy_blocks:
            for group_id in range(self.cache_parallel_size):
                if self.block_managers_map[group_id].can_allocate_blocks(num_blocks):
                    self.block_managers_map[group_id].allocate(seq, num_blocks)
                    self.seq_cache_worker_block_counter[seq.seq_id][
                        group_id
                    ] = num_blocks
                    self.seq_block_counter[seq.seq_id] = num_blocks
                    return True
            return False

        # more than one group is required
        num_cache_parallel_groups = ceil(
            num_blocks / self.max_seq_cache_occupancy_blocks
        )
        last_group_num_blocks = num_blocks % self.max_seq_cache_occupancy_blocks

        num_groups_found = 0
        last_group_found = False

        group_ids: List[int] = []
        last_group_id: Optional[int] = None

        # check if any combination of workers can accommodate this sequence
        for group_id in range(self.cache_parallel_size):
            block_manager = self.block_managers_map[group_id]

            if block_manager.can_allocate_blocks(self.max_seq_cache_occupancy_blocks):
                num_groups_found += 1
                group_ids.append(group_id)
            elif (
                last_group_num_blocks
                and not last_group_found
                and block_manager.can_allocate_blocks(last_group_num_blocks)
            ):
                last_group_found = True
                num_groups_found += 1
                last_group_id = group_id

            if num_groups_found == num_cache_parallel_groups:
                break

        if num_groups_found != num_cache_parallel_groups:
            return False

        if last_group_id:
            group_ids.append(last_group_id)
        else:
            last_group_id = group_ids[-1]

        for group_id in group_ids:
            if group_id == last_group_id:
                self.block_managers_map[group_id].allocate(seq, last_group_num_blocks)
                self.seq_cache_worker_block_counter[seq.seq_id][
                    group_id
                ] = last_group_num_blocks
            else:
                self.block_managers_map[group_id].allocate(
                    seq, self.max_seq_cache_occupancy_blocks
                )
                self.seq_cache_worker_block_counter[seq.seq_id][
                    group_id
                ] = self.max_seq_cache_occupancy_blocks

        self.seq_block_counter[seq.seq_id] = num_blocks

        # print(
        #     f"Allocated {num_blocks} blocks for seq {seq.seq_id} in {num_cache_parallel_groups} groups",
        #     flush=True,
        # )

        return True

    def _free_seq(self, seq: Sequence) -> None:
        for group_id in self.seq_cache_worker_block_counter[seq.seq_id]:
            self.block_managers_map[group_id].free(seq)

        del self.seq_cache_worker_block_counter[seq.seq_id]
        del self.seq_block_counter[seq.seq_id]

    def _can_append_slot(self, seq: Sequence) -> bool:
        # TODO(amey): This is a naive implementation.
        # We need to handle the case where decode tokens are being appended
        # beyond the group limit or the scale when the last group is out of memory
        last_group_id = next(reversed(self.seq_cache_worker_block_counter[seq.seq_id]))
        return self.block_managers_map[last_group_id].can_append_slot()

    def _append_slot(
        self,
        seq: Sequence,
    ) -> None:
        assert seq.is_executing()
        assert seq.prompt_processing_finished

        last_group_id = next(reversed(self.seq_cache_worker_block_counter[seq.seq_id]))
        num_total_blocks = self.seq_block_counter[seq.seq_id]
        has_appended = self.block_managers_map[last_group_id].append_slot(
            seq, num_total_blocks
        )
        self.seq_cache_worker_block_counter[seq.seq_id][last_group_id] += int(
            has_appended
        )
        self.seq_block_counter[seq.seq_id] += int(has_appended)

    def _get_active_group_ids(self, seq: Sequence) -> List[int]:
        group_ids = list(self.seq_cache_worker_block_counter[seq.seq_id])

        if seq.prompt_processing_finished:
            return group_ids

        num_processed_tokens = seq.get_num_prompt_tokens_stage_processed()
        num_groups = num_processed_tokens // self.max_seq_cache_occupancy + 1
        return group_ids[:num_groups]

    def _schedule(self) -> SchedulerOutput:
        # Fix the current time.
        now = time.time()

        running: List[Sequence] = []
        ignored_seq_ids: List[int] = []
        preempted_seq_ids: List[int] = []
        scheduled_seq_metadata_list: List[MnemosyneSequenceScheduleMetadata] = []

        num_batched_tokens: int = 0

        contains_multi_group_seq = False

        contains_prefill = False
        contains_decode = False

        ######################################################################
        # Phase 1: Add existing running sequence groups to the batch.
        # There are two cases:
        # 1. The sequence group has incomplete prefill. The routine
        # remains identical to the one in sarathi scheduler for such sequences.
        # 2. The sequence group has completed prefill. In this case, we need to
        # check for memory availability for the next chunk of decode tokens, and preempt
        # some sequence groups if necessary. Note that, the preempted sequence groups
        # might belong to either of the two categories.
        ######################################################################

        self.running = self.policy.sort_by_priority(now, self.running)

        # in first pass process all the requests with prefill completed
        # this allows us to accurately account for the number of decode tokens
        running_prefills: List[Sequence] = []

        skip_decode = False
        if (
            self.running or self.waiting
        ) and self.skipped_steps < self.scheduler_config.skip_decode_for_iterations:
            self.skipped_steps += 1
            skip_decode = True

        while self.running:
            seq = self.running.pop(0)

            if not seq.is_paused():
                running.append(seq)
                continue

            is_multi_group_seq = (
                len(self.seq_cache_worker_block_counter[seq.seq_id]) > 1
            )

            if contains_multi_group_seq and is_multi_group_seq:
                # TODO(amey): Currently, if there is more than one parallel multi-node request,
                # it could lead to a deadlock. So we limit the batching to one multi-node request.
                running.append(seq)
                break

            if not seq.prompt_processing_finished:
                running_prefills.append(seq)
                continue

            if skip_decode:
                running.append(seq)
                continue

            while not self._can_append_slot(seq):
                if self.running:
                    # Preempt the lowest-priority sequence groups.
                    victim_seq = self.running.pop(-1)
                    self._preempt(victim_seq)
                    preempted_seq_ids.append(victim_seq.seq_id)
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    self._preempt(seq)
                    preempted_seq_ids.append(seq.seq_id)
                    break
            else:
                # Append new slots to the sequence group.
                self._append_slot(seq)
                running.append(seq)
                num_batched_tokens += 1
                group_block_mapping = self.seq_cache_worker_block_counter[seq.seq_id]
                active_group_ids = self._get_active_group_ids(seq)
                if is_multi_group_seq:
                    contains_multi_group_seq = True

                contains_decode = True
                scheduled_seq_metadata_list.append(
                    MnemosyneSequenceScheduleMetadata(
                        self._iteration_id,
                        seq.seq_id,
                        prompt_chunk_len=0,
                        group_block_mapping=group_block_mapping,
                        active_group_ids=active_group_ids,
                    )
                )

        # now add the requests with prefill incomplete
        # the memory for all these prefills has already been allocated
        # so we should be able to run all of them
        for seq in running_prefills:
            assert not seq.prompt_processing_finished

            next_num_prefill_tokens = self._get_seq_next_num_prefill_tokens(
                seq, num_batched_tokens
            )

            # as long as the request could fit in the batch previously
            # it should be able to fit in the batch now
            # so in non-pipeline case this condition should always be false
            # however, in pipeline case, the grouping of requests can change
            # between different microbatches, so this is not guaranteed to be always true
            if contains_multi_group_seq or next_num_prefill_tokens == 0:
                running.append(seq)
                continue

            if is_multi_group_seq:
                contains_multi_group_seq = True

            num_batched_tokens += next_num_prefill_tokens
            group_block_mapping = self.seq_cache_worker_block_counter[seq.seq_id]
            active_group_ids = self._get_active_group_ids(seq)
            contains_prefill = True
            scheduled_seq_metadata_list.append(
                MnemosyneSequenceScheduleMetadata(
                    self._iteration_id,
                    seq.seq_id,
                    prompt_chunk_len=next_num_prefill_tokens,
                    group_block_mapping=group_block_mapping,
                    active_group_ids=active_group_ids,
                )
            )
            running.append(seq)

        ######################################################################
        # Phase 2: Add waiting (new) sequence groups to the batch.
        # This routine is nearly-identical to the one in sarathi scheduler
        ######################################################################
        # Optimization: We do not sort the waiting queue since the preempted
        # sequence groups are added to the front and the new sequence groups
        # are added to the back.
        while self.waiting:
            seq = self.waiting[0]

            # TODO(amey): Currently, when we have any multi-group request we only run one prefill
            if contains_multi_group_seq:
                break

            # This is required to handle benchmarking where we set request arrival time ahead of time
            if seq.arrival_time > now:
                break

            if not self._check_request_prompt_length(seq):
                ignored_seq_ids.append(seq.seq_id)
                self.waiting.pop(0)
                continue

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            if len(running) >= self.scheduler_config.max_num_seqs:
                break

            # check if we can fit the prefill in the batch
            next_num_prefill_tokens = self._get_seq_next_num_prefill_tokens(
                seq, num_batched_tokens
            )

            if next_num_prefill_tokens == 0:
                break

            num_cache_parallel_groups = ceil(
                seq.get_prompt_len() / self.max_seq_cache_occupancy
            )

            if num_cache_parallel_groups > self.cache_parallel_size:
                logger.warning(
                    f"Ignoring seq_id: {seq.seq_id} due to max cache parallel size limit."
                )
                ignored_seq_ids.append(seq.seq_id)
                self.waiting.pop(0)
                continue

            if contains_multi_group_seq and num_cache_parallel_groups > 1:
                # TODO(amey): Currently, if there is more than one parallel multi-node request,
                # it could lead to a deadlock. So we limit the batching to one multi-node request.
                # But this can lead to a lot of head of line blocking.
                break

            # If the sequence group cannot be allocated, stop.
            if not self._allocate(seq):
                break

            self.waiting.pop(0)

            if num_cache_parallel_groups > 1:
                contains_multi_group_seq = True

            num_batched_tokens += next_num_prefill_tokens
            group_block_mapping = self.seq_cache_worker_block_counter[seq.seq_id]
            active_group_ids = self._get_active_group_ids(seq)
            contains_prefill = True
            scheduled_seq_metadata_list.append(
                MnemosyneSequenceScheduleMetadata(
                    self._iteration_id,
                    seq.seq_id,
                    prompt_chunk_len=next_num_prefill_tokens,
                    group_block_mapping=group_block_mapping,
                    active_group_ids=active_group_ids,
                )
            )
            running.append(seq)

        # make sure that prefills are at the start of the batch, so that we don't violate assumptions
        # made in the original vllm codebase
        self.running = running

        skip_model_execution = False
        if self.scheduler_config.skip_execution_till_overlapping:
            skip_model_execution = not (contains_prefill and contains_decode)

        # print(
        #     f"iteration_id: {self._iteration_id}, "
        #     f"skip_decode: {skip_decode}, "
        #     f"skip_model_execution: {skip_model_execution}, "
        #     f"scheduled_seq_metadata_list: {scheduled_seq_metadata_list}",
        # )

        return SchedulerOutput(
            id=self._iteration_id,
            ignored_seq_ids=ignored_seq_ids,
            preempted_seq_ids=preempted_seq_ids,
            scheduled_seq_metadata_list=scheduled_seq_metadata_list,
            skip_model_execution=skip_model_execution,
        )
