import math
from collections import defaultdict
from typing import Dict, List, Optional

from sarathi.core.datatypes.sequence import SamplerOutputs, SequenceMetadata
from sarathi.core.datatypes.zmq_protocol import StepMicrobatchOuputs, StepOutputs
from sarathi.core.sequence_manager.mnemosyne_engine_sequence_manager import (
    MnemosyneEngineSequenceManager,
)
from sarathi.engine.pipeline_parallel_llm_engine import PipelineParallelLLMEngine
from sarathi.utils.threading_utils import exit_on_error


class MnemosynePipelineParallelLLMEngine(PipelineParallelLLMEngine):
    def _get_blocks_per_request(self) -> int:
        if self.config.parallel_config.cache_parallel_size == 1:
            return super()._get_blocks_per_request()

        return math.ceil(
            self.config.parallel_config.max_seq_cache_occupancy
            / self.config.cache_config.block_size
        )

    def _get_worker_impl(self):
        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        from sarathi.worker.mnemosyne_pipeline_parallel_worker import (
            MnemosynePipelineParallelWorker,
        )

        return MnemosynePipelineParallelWorker

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

    @exit_on_error
    def _microbatch_watch_loop(self) -> None:
        pending_microbatch_outputs: Dict[int, List[StepMicrobatchOuputs]] = defaultdict(
            list
        )

        while True:
            scheduler_output = self.microbatch_watch_queue.get()
            schedule_id = scheduler_output.id
            num_microbatch_outputs_received = 0

            num_microbatch_outputs_received += len(
                pending_microbatch_outputs[schedule_id]
            )
            del pending_microbatch_outputs[schedule_id]

            while (
                num_microbatch_outputs_received
                < self.config.parallel_config.cache_parallel_size
            ):
                step_microbatch_outputs = self.microbatch_socket.recv_pyobj()
                if step_microbatch_outputs.schedule_id != schedule_id:
                    pending_microbatch_outputs[
                        step_microbatch_outputs.schedule_id
                    ].append(step_microbatch_outputs)
                    continue

                num_microbatch_outputs_received += 1

            self.seq_manager.on_stage_completed(scheduler_output)
            self.schedule_event.set()

    @exit_on_error
    def _output_loop(self) -> None:
        pending_step_outputs: Dict[int, List[StepOutputs]] = defaultdict(list)

        while True:
            scheduler_stage_output = self.scheduler_output_queue.get()
            schedule_id = scheduler_stage_output.scheduler_output.id
            num_step_outputs_received = 0

            all_sampler_outputs: List[Optional[SamplerOutputs]] = []

            all_sampler_outputs.extend(pending_step_outputs[schedule_id])
            num_step_outputs_received += len(pending_step_outputs[schedule_id])
            del pending_step_outputs[schedule_id]

            while (
                num_step_outputs_received
                < self.config.parallel_config.cache_parallel_size
            ):
                step_output = self.output_socket.recv_pyobj()
                if step_output.schedule_id != schedule_id:
                    pending_step_outputs[step_output.schedule_id].append(
                        step_output.sampler_outputs
                    )
                    continue
                all_sampler_outputs.append(step_output.sampler_outputs)
                num_step_outputs_received += 1

            # print(
            #     f"Received {all_sampler_outputs} sampler outputs from workers", flush=True
            # )

            combined_sampler_outputs = self._combine_sampler_outputs(
                all_sampler_outputs, scheduler_stage_output.seq_metadata_list
            )

            self._append_pending_step_output(
                scheduler_stage_output.scheduler_output, combined_sampler_outputs
            )

            all_request_outputs = self._on_step_completed(
                scheduler_stage_output.scheduler_output,
                scheduler_stage_output.ignored_seqs,
                scheduler_stage_output.seq_metadata_list,
                combined_sampler_outputs,
                scheduler_stage_output.start_time,
            )
            self.schedule_event.set()

            self.output_queue.put(all_request_outputs)
