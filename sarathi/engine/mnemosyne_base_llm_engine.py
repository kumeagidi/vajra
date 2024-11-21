import math
import time
from typing import List, Optional

from sarathi.core.datatypes.request_output import RequestOutput
from sarathi.core.datatypes.scheduler_output import SchedulerOutput
from sarathi.core.datatypes.sequence import SamplerOutputs, SequenceMetadata
from sarathi.core.datatypes.zmq_protocol import StepInputs
from sarathi.core.sequence_manager.mnemosyne_engine_sequence_manager import (
    MnemosyneEngineSequenceManager,
)
from sarathi.engine.base_llm_engine import BaseLLMEngine


class MnemosyneBaseLLMEngine(BaseLLMEngine):

    def _get_blocks_per_request(self) -> int:
        return math.ceil(
            self.config.parallel_config.max_seq_cache_occupancy
            / self.config.cache_config.block_size
        )

    def _get_worker_impl(self):
        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        from sarathi.worker.mnemosyne_base_worker import MnemosyneBaseWorker

        return MnemosyneBaseWorker

    def _get_seq_manager_impl(self):
        return MnemosyneEngineSequenceManager

    def _combine_sampler_outputs(
        self,
        all_workers_sampler_outputs: List[Optional[SamplerOutputs]],
        seq_metadata_list: List[SequenceMetadata],
    ) -> List[SamplerOutputs]:
        sampler_outputs = sum(filter(None, all_workers_sampler_outputs), [])
        sampler_outputs = set(sampler_outputs)
        sampler_outputs_map = {s.seq_id: s for s in sampler_outputs}
        sampler_outputs = [sampler_outputs_map[s.seq_id] for s in seq_metadata_list]
        return sampler_outputs

    def step(self) -> List[RequestOutput]:
        """Performs one decoding iteration and returns newly generated results.

        This function performs one decoding iteration of the engine. It first
        schedules the sequences to be executed in the next iteration.
        Then, it executes the model and updates the scheduler with the model outputs.
        Finally, it decodes the sequences and returns the newly generated results.
        """
        start_time = time.time()

        with self._scheduler_timer:
            scheduler_output = self.scheduler.schedule()

        if scheduler_output.is_empty():
            return []

        with self._on_schedule_handling_timer:
            ignored_seqs, seq_metadata_list = self.seq_manager.on_schedule(
                scheduler_output
            )

        self.enqueue_socket.send_pyobj(
            StepInputs(
                scheduler_output,
                new_seqs=self._get_new_seqs(),
            )
        )
        all_sampler_outputs: List[Optional[SamplerOutputs]] = []
        for _ in range(self.config.parallel_config.cache_parallel_size):
            step_outputs = self.output_socket.recv_pyobj()
            assert step_outputs.schedule_id == scheduler_output.id
            all_sampler_outputs.append(step_outputs.sampler_outputs)

        combined_sampler_outputs = self._combine_sampler_outputs(
            all_sampler_outputs, seq_metadata_list
        )

        return self._on_step_completed(
            scheduler_output,
            ignored_seqs,
            seq_metadata_list,
            combined_sampler_outputs,
            start_time,
        )
