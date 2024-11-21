from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed

from sarathi.config import SchedulerType, SystemConfig
from sarathi.core.datatypes.sampling_params import SamplingParams
from sarathi.core.datatypes.scheduler_output import SchedulerOutput
from sarathi.core.datatypes.sequence import SamplerOutputs, Sequence, SequenceMetadata
from sarathi.logger import init_logger
from sarathi.metrics.constants import CpuOperationMetrics
from sarathi.metrics.cpu_timer import CpuTimer
from sarathi.model_executor import get_model, set_random_seed
from sarathi.model_executor.attention import (
    create_attention_wrapper,
    get_attention_wrapper,
)
from sarathi.model_executor.layers.sampler import Sampler
from sarathi.model_executor.parallel_utils import (
    recv_from_last_pipeline_stage,
    send_to_next_pipeline_stage,
)
from sarathi.model_executor.parallel_utils.parallel_state import (
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)
from sarathi.utils import get_gpu_memory
from sarathi.worker.cache_engine import CacheEngine

logger = init_logger(__name__)

_NUM_WARMUP_ITERS = 5


class ModelRunner:

    def __init__(
        self,
        config: SystemConfig,
        device: torch.device,
        rank: int,
    ):
        self.config = config
        self.device = device
        self.rank = rank

        self.model = get_model(self.config.model_config)
        create_attention_wrapper(
            config.scheduler_config,
            config.model_config,
            config.parallel_config,
            config.worker_config,
            config.cache_config.block_size,
            self.device,
        )

        self.sampler: Optional[Sampler] = None
        if self.model.lm_head:
            self.sampler = Sampler(
                self.model.lm_head.weight, self.model.config.vocab_size
            )

        self.is_pipeline_first_stage = is_pipeline_first_stage()
        self.is_pipeline_last_stage = is_pipeline_last_stage()

        self.cuda_graph_runner_map: Dict[int, CUDAGraphRunner] = {}
        self.graph_memory_pool: Optional[Tuple[int, int]] = None

        self._prepare_inputs_e2e_timer = CpuTimer(
            CpuOperationMetrics.PREPARE_INPUTS_E2E, rank=self.rank
        )
        self._sampler_e2e_timer = CpuTimer(
            CpuOperationMetrics.SAMPLER_E2E, rank=self.rank
        )
        self._model_execution_e2e_decode_timer = CpuTimer(
            CpuOperationMetrics.MODEL_EXECUTION_E2E_DECODE, rank=self.rank
        )
        self._model_execution_e2e_prefill_timer = CpuTimer(
            CpuOperationMetrics.MODEL_EXECUTION_E2E_PREFILL, rank=self.rank
        )
        self._model_execution_e2e_mixed_timer = CpuTimer(
            CpuOperationMetrics.MODEL_EXECUTION_E2E_MIXED, rank=self.rank
        )
        self._attn_begin_forward_timer = CpuTimer(
            CpuOperationMetrics.ATTN_BEGIN_FORWARD, rank=self.rank
        )

        self.dummy_tensor = torch.ones(1, device=self.device)

    def _prepare_inputs(
        self,
        seq_metadata_list: List[SequenceMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_tokens: List[int] = []
        input_positions: List[int] = []
        # need to know prompt chunk sizes for each prompt sequence for sampler
        current_prompt_chunk_lens: List[int] = []

        for seq_metadata in seq_metadata_list:
            if not seq_metadata.is_prompt:
                continue

            prompt_chunk_len = seq_metadata.prompt_chunk_len
            current_prompt_chunk_tokens = (
                seq_metadata.seq.get_next_prompt_chunk_token_ids(prompt_chunk_len)
            )
            current_prompt_chunk_len = len(current_prompt_chunk_tokens)
            current_prompt_chunk_lens.append(current_prompt_chunk_len)
            processed_prompt_len = (
                seq_metadata.seq.get_num_prompt_tokens_stage_processed()
            )

            current_total_len = processed_prompt_len + current_prompt_chunk_len

            input_tokens.extend(current_prompt_chunk_tokens)
            input_positions.extend(range(processed_prompt_len, current_total_len))

        for seq_metadata in seq_metadata_list:
            if seq_metadata.is_prompt:
                continue

            generation_token = seq_metadata.seq.get_last_token_id()
            input_tokens.append(generation_token)

            context_len = seq_metadata.seq.get_len()
            position = context_len - 1
            input_positions.append(position)

        # Optimization: Pad the input length to be a multiple of 8.
        # This is required for utilizing the Tensor Cores in NVIDIA GPUs.
        # input_tokens = pad_to_alignment(input_tokens, multiple_of=8)
        # input_positions = pad_to_alignment(input_positions, multiple_of=8)

        # Convert to tensors.
        tokens_tensor = torch.tensor(input_tokens, dtype=torch.long, device=self.device)
        positions_tensor = torch.tensor(
            input_positions, dtype=torch.long, device=self.device
        )

        return tokens_tensor, positions_tensor

    @torch.inference_mode()
    def profile_num_available_blocks(
        self,
        block_size: int,
        gpu_memory_utilization: float,
    ) -> Tuple[int, int]:
        torch.cuda.set_device(self.device)

        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Enable top-k sampling to reflect the accurate memory usage.
        vocab_size = self.model.config.vocab_size
        sampling_params = SamplingParams(top_p=0.99, top_k=vocab_size - 1)
        max_num_batched_tokens = (
            self.config.scheduler_config.get_max_num_batched_tokens(
                self.config.model_config.max_model_len
            )
        )
        max_num_seqs = self.config.scheduler_config.max_num_seqs

        seq_metadata_list: List[SequenceMetadata] = []

        if (
            self.config.scheduler_config.get_type() == SchedulerType.SARATHI
            or self.config.scheduler_config.get_type() == SchedulerType.SIMPLE_CHUNKING
            or self.config.scheduler_config.get_type() == SchedulerType.MNEMOSYNE
        ):
            # Profile memory usage with a single `chunk_size` chunk
            # which is the last chunk in the longest supported sequence.
            chunk_size = self.config.scheduler_config.get_max_num_batched_tokens(
                self.config.model_config.max_model_len
            )
            seq_len = int(self.config.model_config.max_model_len)
            chunk_size = int(min(chunk_size, seq_len))
            print(
                f"chunk_size: {chunk_size}, seq_len: {seq_len}, block_size: {block_size}",
                flush=True,
            )
            seq = Sequence(
                seq_id=0,
                prompt=None,
                prompt_token_ids=[0] * seq_len,
                block_size=block_size,
                eos_token_id=1,
                arrival_time=None,
                sampling_params=sampling_params,
            )
            seq_metadata = SequenceMetadata(
                schedule_id=0,
                seq=seq,
                block_table=None,
                prompt_chunk_len=chunk_size,
            )
            seq_metadata_list.append(seq_metadata)
        else:
            # Profile memory usage with max_num_sequences sequences and the total
            # number of tokens equal to max_num_batched_tokens.
            for seq_id in range(max_num_seqs):
                seq_len = max_num_batched_tokens // max_num_seqs + (
                    seq_id < max_num_batched_tokens % max_num_seqs
                )

                seq = Sequence(
                    seq_id=str(seq_id),
                    prompt=None,
                    prompt_token_ids=[0] * seq_len,
                    block_size=block_size,
                    eos_token_id=1,
                    arrival_time=None,
                    sampling_params=sampling_params,
                )
                seq_metadata = SequenceMetadata(
                    schedule_id=0,
                    seq=seq,
                    block_table=None,
                    prompt_chunk_len=seq_len,
                )
                seq_metadata_list.append(seq_metadata)

        input_tokens, input_positions = self._prepare_inputs(seq_metadata_list)
        get_attention_wrapper().begin_forward(seq_metadata_list)

        if not self.is_pipeline_first_stage:
            # hidden_states_shape: num_tokens x hidden_size
            input_tokens = torch.empty(
                (input_positions.shape[0], self.model.config.hidden_size),
                dtype=self.model.config.dtype,
                device=self.device,
            )

        # Execute the model.
        num_layers = self.config.model_config.get_num_layers(
            self.config.parallel_config
        )
        self.model(
            hidden_states=input_tokens,
            positions=input_positions,
            kv_caches=[None] * num_layers,
        )

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated()
        total_gpu_memory = get_gpu_memory()
        print(
            f"peak_memory: {peak_memory}, total_gpu_memory: {total_gpu_memory}, gpu_memory_utilization: {gpu_memory_utilization}, block_size: {block_size}",
            flush=True,
        )
        cache_block_size = CacheEngine.get_cache_block_size(
            block_size, self.config.model_config, self.config.parallel_config
        )
        num_gpu_blocks = int(
            (total_gpu_memory * gpu_memory_utilization - peak_memory)
            // cache_block_size
        )
        num_gpu_blocks = max(num_gpu_blocks, 0)
        torch.cuda.empty_cache()

        get_attention_wrapper().end_forward()

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.config.model_config.seed)
        return num_gpu_blocks

    def get_model_timer(
        self,
        seq_metadata_list: List[SequenceMetadata],
    ) -> CpuTimer:
        contains_prefill = any(
            seq_metadata.is_prompt for seq_metadata in seq_metadata_list
        )
        contains_decode = any(
            not seq_metadata.is_prompt for seq_metadata in seq_metadata_list
        )
        if contains_prefill and contains_decode:
            return self._model_execution_e2e_mixed_timer
        elif contains_prefill:
            return self._model_execution_e2e_prefill_timer
        else:
            return self._model_execution_e2e_decode_timer

    def run(
        self,
        scheduler_output: SchedulerOutput,
        seq_metadata_list: List[SequenceMetadata],
        gpu_cache: Optional[List[torch.Tensor]] = None,
    ) -> Optional[SamplerOutputs]:
        if not seq_metadata_list:
            return []

        seq_metadata_list_hash = hash(
            tuple(hash(seq_metadata) for seq_metadata in seq_metadata_list)
        )

        # print(
        #     f"rank {torch.distributed.get_rank()} seq_metadata_list: {seq_metadata_list} seq_metadata_list_hash {seq_metadata_list_hash} {tuple(hash(seq_metadata) for seq_metadata in seq_metadata_list)}", flush=True
        # )

        # Prepare input tensors.
        with self._prepare_inputs_e2e_timer:
            input_tokens, input_positions = self._prepare_inputs(seq_metadata_list)

        is_prefill_batch = all(
            seq_metadata.is_prompt for seq_metadata in seq_metadata_list
        )

        if scheduler_output.skip_model_execution or (
            self.config.worker_config.skip_prefill and is_prefill_batch
        ):
            output = torch.zeros(
                input_tokens.shape[0],
                self.model.config.hidden_size,
                device=self.device,
                dtype=torch.float16,
            )
            if self.sampler is not None:
                with self._sampler_e2e_timer:
                    output = self.sampler(output, seq_metadata_list)

            return output

        if not self.is_pipeline_first_stage:
            # hidden_states_shape: num_tokens x hidden_size
            input_tokens = torch.empty(
                (input_positions.shape[0], self.model.config.hidden_size),
                dtype=self.model.config.dtype,
                device=self.device,
            )
            if not self.config.worker_config.skip_p2p_communication:
                input_tokens = recv_from_last_pipeline_stage(input_tokens)
                # recv_from_last_pipeline_stage(self.dummy_tensor)

        with self._attn_begin_forward_timer:
            get_attention_wrapper().begin_forward(seq_metadata_list)
            # torch.cuda.synchronize()

        if (
            self.config.worker_config.use_cuda_graph
            and seq_metadata_list_hash not in self.cuda_graph_runner_map
        ):
            # print(f"Capturing graph for seq_metadata_list_hash {seq_metadata_list_hash}, input_tokens: {input_tokens.shape}", flush=True)
            cuda_graph_runner = CUDAGraphRunner(self.model)
            cuda_graph_runner.capture(
                input_tokens,
                input_positions,
                gpu_cache,
                memory_pool=self.graph_memory_pool,
            )
            self.cuda_graph_runner_map[seq_metadata_list_hash] = cuda_graph_runner
            if self.graph_memory_pool is None:
                self.graph_memory_pool = cuda_graph_runner.graph.pool()

            # torch.cuda.synchronize()

        with self.get_model_timer(seq_metadata_list):
            # print(f"Running graph for seq_metadata_list_hash {seq_metadata_list_hash}, input_tokens: {input_tokens.shape}", flush=True)
            if self.config.worker_config.use_cuda_graph:
                output = self.cuda_graph_runner_map[seq_metadata_list_hash](
                    input_tokens,
                    input_positions,
                )
            else:
                output = self.model(
                    input_tokens,
                    input_positions,
                    gpu_cache,
                )
            # torch.cuda.synchronize()

        # torch.cuda.synchronize()
        # except RuntimeError as e:
        #     print(
        #         f"Error Rank {self.rank} failed for seq_metadata_list: {seq_metadata_list}, seq_metadata_list_hash {seq_metadata_list_hash}",
        #         flush=True,
        #     )
        #     raise e

        # print(
        #     f"rank {torch.distributed.get_rank()} iter: {seq_metadata_list[0].schedule_id} output checksum {output.sum().item()}", flush=True
        # )

        torch.cuda.synchronize()

        if self.sampler is not None:
            with self._sampler_e2e_timer:
                output = self.sampler(output, seq_metadata_list)
        else:  # is not last stage
            assert not self.is_pipeline_last_stage
            if not self.config.worker_config.skip_p2p_communication:
                send_to_next_pipeline_stage(output)
                # send_to_next_pipeline_stage(self.dummy_tensor)

        get_attention_wrapper().end_forward()

        return output


class CUDAGraphRunner:

    def __init__(self, model: torch.nn.Module):
        self.model = model

        self.input_buffers: Dict[str, torch.Tensor] = {}
        self.output_buffers: Dict[str, torch.Tensor] = {}

        self._graph: Optional[torch.cuda.CUDAGraph] = None

    @property
    def graph(self):
        assert self._graph is not None
        return self._graph

    def capture(
        self,
        input_tokens: torch.Tensor,
        input_positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        memory_pool: Optional[Tuple[int, int]],
    ) -> Optional[torch.Tensor]:
        assert self._graph is None

        self.input_buffers["input_tokens"] = input_tokens
        self.input_buffers["input_positions"] = input_positions

        # Run the model a few times without capturing the graph.
        # This is to make sure that the captured graph does not include the
        # kernel launches for initial benchmarking (e.g., Triton autotune).
        # Note one iteration is not enough for torch.jit.script
        for _ in range(_NUM_WARMUP_ITERS):
            self.model(
                input_tokens,
                input_positions,
                kv_caches,
            )
        torch.cuda.synchronize()

        # Capture the graph.
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph, pool=memory_pool):
            self.output_buffers = self.model(
                input_tokens,
                input_positions,
                kv_caches,
            )
        torch.cuda.synchronize()

        return self.output_buffers

    def forward(
        self,
        input_tokens: torch.Tensor,
        input_positions: torch.Tensor,
    ) -> torch.Tensor:
        # Copy the input tensors to the input buffers.
        self.input_buffers["input_tokens"].copy_(input_tokens, non_blocking=True)
        self.input_buffers["input_positions"].copy_(input_positions, non_blocking=True)

        self.graph.replay()

        return self.output_buffers

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
