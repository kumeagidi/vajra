from typing import Any, List, Optional

import torch
from flashinfer import BatchPrefillWithPagedKVCacheWrapper, append_paged_kv_cache
from flashinfer.cascade import merge_states

from sarathi.config import BaseSchedulerConfig, CacheConfig, ModelConfig, ParallelConfig
from sarathi.core.datatypes.sequence import MnemosyneSequenceMetadata, SequenceMetadata
from sarathi.metrics.constants import OperationMetrics
from sarathi.model_executor.attention.base_attention_wrapper import BaseAttentionWrapper
from sarathi.model_executor.parallel_utils.mappings import (
    gather_from_cache_model_parallel_region,
)
from sarathi.model_executor.parallel_utils.parallel_state import (
    get_cache_model_parallel_rank,
)

USE_FAKE_PAGE_TENSOR = True
SKIP_ATTENTION_REDUCTION = False


class FlashinferAttentionWrapper(BaseAttentionWrapper):
    def __init__(
        self,
        scheduler_config: BaseSchedulerConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        block_size: int,
        device: torch.device,
    ):
        super().__init__(
            scheduler_config, model_config, parallel_config, block_size, device
        )

        self.max_batch_size = scheduler_config.max_num_seqs
        self.max_kv_cache_blocks = (
            scheduler_config.max_num_seqs * model_config.max_model_len // block_size
        )

        self.is_metadata_initialized = False
        self.is_profiling_iteration = False

        self.contains_multi_group_prefill_seq: bool = False
        self.contains_multi_group_decode_seq: bool = False
        self.multi_group_seq_prefill_len: int = 0
        self.multi_group_seq_group_ids: List[int] = []
        self.multi_group_seq_store_kv_cache: bool = True

        self.is_cache_parallel = parallel_config.cache_parallel_size > 1
        if self.is_cache_parallel:
            self.group_id = get_cache_model_parallel_rank()
        else:
            self.group_id = 0

        self.iter_id = 0

    def set_cache_config(self, cache_config: CacheConfig) -> None:
        workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )
        self.page_tensor = torch.arange(
            cache_config.num_gpu_blocks - 1,
            -1,
            -1,
            device=self.device,
            dtype=torch.int32,
        )
        self.wrapper = BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer,
            "NHD",
            use_cuda_graph=True,
            qo_indptr_buf=self._to_int_tensor([0] * (self.max_batch_size + 1)),
            paged_kv_indptr_buf=self._to_int_tensor([0] * (self.max_batch_size + 1)),
            paged_kv_indices_buf=(
                self.page_tensor
                if USE_FAKE_PAGE_TENSOR
                else self._to_int_tensor([0] * self.max_kv_cache_blocks)
            ),
            paged_kv_last_page_len_buf=self._to_int_tensor([0] * self.max_batch_size),
        )

        self.append_qo_indptr_buf = self._to_int_tensor([0] * (self.max_batch_size + 1))
        self.append_kv_page_indptr_buf = self._to_int_tensor(
            [0] * (self.max_batch_size + 1)
        )
        if USE_FAKE_PAGE_TENSOR:
            self.append_kv_page_indices_buf = self.page_tensor
        else:
            self.append_kv_page_indices_buf = self._to_int_tensor(
                [0] * self.max_kv_cache_blocks
            )
        self.append_kv_last_page_len_buf = self._to_int_tensor(
            [0] * self.max_batch_size
        )

    def _to_int_tensor(self, data: List[int], host=False) -> torch.Tensor:
        device = "cpu" if host else "cuda"
        return torch.tensor(data, dtype=torch.int32, device=device)

    def _pad_list(self, data: List[Any], max_len: int, value: Any) -> List[Any]:
        return data + [value] * (max_len - len(data))

    def get_cache_block(self, num_blocks: int, **kwargs) -> torch.Tensor:
        return torch.randn(
            num_blocks,
            2,
            self.block_size,
            self.num_kv_heads,
            self.head_dim,
            **kwargs,
        )

    def begin_forward(
        self,
        seq_metadata_list: List[SequenceMetadata],
    ) -> None:
        # The indptr tensor captures the location query tokens in the input tensor.
        # |<---------------------- num_valid_tokens ----------------------------------------------------->|
        # |<--------------- num_prompt_tokens -------------->||<------- num_generation_tokens (M) ------->|
        # |<--prompt_0-->|<--prompt_1-->|...|<--prompt_N-1-->||<--generation_0-->|...|<--generation_M-1-->|<--padding-->|
        #
        # Flashinfer calls this layout as a raggedtensor. The indptr tensor captures the start of each
        # sequence in the ragged tensor. The length of the indptr tensor is the number of sequences + 1.
        # We perform both prefill and decode attention in a single call to batched prefill kernel.
        # prefill_qo_indptr: [0, prompt_0, prompt_0 + prompt_1, ..., prompt_0 + ... + prompt_N-1, generation_0, generation_0 + 1, ..., generation_0 + ... + M]
        qo_indptr: List[int] = [0]
        # The kv_page_indices tensor captures the pages of the key-value cache that
        # are assigned to each token in the input tensor. Since there is a variable number
        # of pages assigned to each sequence, a ragged tensor to represent this.
        kv_page_indices: List[int] = []
        # the last page might not be full, so we need to keep track of the length of the last page
        kv_last_page_len: List[int] = []
        # Since the prefill_kv_page_indices tensor is a ragged tensor, we also need to keep track of the
        # indptr tensor for the prefill_kv_page_indices tensor. This tensor captures the start of each sequence
        # in the ragged tensor.
        kv_page_indptr: List[int] = [0]

        append_kv_page_indices: List[int] = []
        append_kv_last_page_len: List[int] = []
        append_kv_page_indptr: List[int] = [0]
        append_qo_indptr: List[int] = [0]

        contains_multi_group_seq: bool = False
        contains_multi_group_prefill_seq: bool = False
        contains_multi_group_decode_seq: bool = False
        multi_group_seq_prefill_len: int = 0
        multi_group_seq_group_ids: List[int] = []
        multi_group_seq_store_kv_cache: bool = True

        self.is_profiling_iteration = False
        self.is_metadata_initialized = True

        for i, seq_metadata in enumerate(seq_metadata_list):
            # ONLY used for profiling
            if seq_metadata.block_table is None:
                self.is_profiling_iteration = True
                # During memory profiling, the block tables are not initialized yet.
                #  We will just skip the attention computation for now.
                return

            if seq_metadata.is_prompt:
                assert (
                    not contains_multi_group_prefill_seq
                ), "Only one prefill sequence is allowed when we have multi-group sequence."

            self.iter_id = seq_metadata.schedule_id

            is_multi_group_seq = False

            if self.is_cache_parallel:
                assert type(seq_metadata) == MnemosyneSequenceMetadata
                is_multi_group_seq = len(seq_metadata.group_ids) > 1
                kv_cache_len = seq_metadata.kv_cache_len
            else:
                if seq_metadata.is_prompt:
                    kv_cache_len = (
                        seq_metadata.seq.get_num_prompt_tokens_stage_processed()
                    )
                else:
                    kv_cache_len = seq_metadata.seq.get_len()

            if seq_metadata.is_prompt:
                qo_len = seq_metadata.prompt_chunk_len
            else:
                qo_len = 1

            if is_multi_group_seq:
                assert (
                    not contains_multi_group_seq
                ), "Currently only one multi-group sequence can run in a single step."
                if seq_metadata.is_prompt:
                    assert (
                        i == 0
                    ), "Multi-group prefill sequence should be the first sequence."
                    multi_group_seq_prefill_len = qo_len
                    contains_multi_group_prefill_seq = True
                else:
                    assert (
                        i == len(seq_metadata_list) - 1
                    ), "Multi-group decode sequence should be the last sequence."
                    contains_multi_group_decode_seq = True
                contains_multi_group_seq = True
                multi_group_seq_group_ids = seq_metadata.group_ids
                multi_group_seq_store_kv_cache = seq_metadata.save_kv_cache

            if seq_metadata.is_prompt and (
                not is_multi_group_seq or multi_group_seq_store_kv_cache
            ):
                kv_cache_len = kv_cache_len + qo_len

            # indptr for the prompt tokens in q/o tensor
            qo_indptr.append(qo_indptr[-1] + qo_len)
            # Compute the kv page indices for the prompt tokens.
            num_blocks_in_use = (kv_cache_len + self.block_size - 1) // self.block_size
            num_blocks_in_use = min(num_blocks_in_use, len(seq_metadata.block_table))
            if not USE_FAKE_PAGE_TENSOR:
                kv_page_indices.extend(seq_metadata.block_table[:num_blocks_in_use])
            kv_page_indptr.append(kv_page_indptr[-1] + num_blocks_in_use)
            kv_last_page_len.append(kv_cache_len % self.block_size or self.block_size)

            if is_multi_group_seq and not multi_group_seq_store_kv_cache:
                if seq_metadata.is_prompt:
                    append_qo_indptr[0] = qo_len
                continue

            append_qo_indptr.append(append_qo_indptr[-1] + qo_len)
            if not USE_FAKE_PAGE_TENSOR:
                append_kv_page_indices.extend(
                    seq_metadata.block_table[:num_blocks_in_use]
                )
            append_kv_page_indptr.append(append_kv_page_indptr[-1] + num_blocks_in_use)
            append_kv_last_page_len.append(
                kv_cache_len % self.block_size or self.block_size
            )

        qo_indptr = self._pad_list(qo_indptr, self.max_batch_size + 1, qo_indptr[-1])
        kv_page_indptr = self._pad_list(
            kv_page_indptr, self.max_batch_size + 1, kv_page_indptr[-1]
        )
        kv_last_page_len = self._pad_list(kv_last_page_len, self.max_batch_size, 0)

        if USE_FAKE_PAGE_TENSOR:
            kv_page_indices_tensor = self.page_tensor
        else:
            kv_page_indices = self._pad_list(
                kv_page_indices, self.max_kv_cache_blocks, 0
            )
            kv_page_indices_tensor = self._to_int_tensor(kv_page_indices, host=True)

        self.wrapper.begin_forward(
            self._to_int_tensor(qo_indptr, host=True),
            self._to_int_tensor(kv_page_indptr, host=True),
            kv_page_indices_tensor,
            self._to_int_tensor(kv_last_page_len, host=True),
            self.num_q_heads,
            self.num_kv_heads,
            self.head_dim,
            self.block_size,
        )

        append_qo_indptr = self._pad_list(
            append_qo_indptr, self.max_batch_size + 1, append_qo_indptr[-1]
        )
        append_kv_page_indptr = self._pad_list(
            append_kv_page_indptr, self.max_batch_size + 1, append_kv_page_indptr[-1]
        )
        append_kv_last_page_len = self._pad_list(
            append_kv_last_page_len, self.max_batch_size, 0
        )

        self.append_qo_indptr_buf.copy_(
            self._to_int_tensor(append_qo_indptr, host=True), non_blocking=True
        )
        self.append_kv_page_indptr_buf.copy_(
            self._to_int_tensor(append_kv_page_indptr, host=True), non_blocking=True
        )
        self.append_kv_last_page_len_buf.copy_(
            self._to_int_tensor(append_kv_last_page_len, host=True), non_blocking=True
        )

        if USE_FAKE_PAGE_TENSOR:
            # append_kv_page_indices_tensor = self.page_tensor
            pass
        else:
            append_kv_page_indices = self._pad_list(
                append_kv_page_indices, self.max_kv_cache_blocks, 0
            )
            self.append_kv_page_indices_buf.copy_(
                self._to_int_tensor(append_kv_page_indices, host=True),
                non_blocking=True,
            )

        self.multi_group_seq_group_ids = multi_group_seq_group_ids
        self.multi_group_seq_store_kv_cache = multi_group_seq_store_kv_cache
        self.multi_group_seq_prefill_len = multi_group_seq_prefill_len
        self.contains_multi_group_prefill_seq = contains_multi_group_prefill_seq
        self.contains_multi_group_decode_seq = contains_multi_group_decode_seq

        # print(
        #     f"Initializing prefill wrapper for rank {self.group_id} for {seq_metadata_list} with args:"
        #     f"\n\tqo_indptr: {qo_indptr}"
        #     f"\n\tkv_page_indices: {kv_page_indices}"
        #     f"\n\tkv_page_indptr: {kv_page_indptr}"
        #     f"\n\tkv_last_page_len: {kv_last_page_len}",
        #     f"\n\tappend_qo_indptr: {append_qo_indptr}"
        #     f"\n\tappend_kv_page_indices: {append_kv_page_indices}"
        #     f"\n\tappend_kv_page_indptr: {append_kv_page_indptr}"
        #     f"\n\tappend_kv_last_page_len: {append_kv_last_page_len}",
        #     flush=True,
        # )

    def end_forward(self):
        self.is_metadata_initialized = False

        if self.is_profiling_iteration:
            return

        self.wrapper.end_forward()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        softmax_scale: float = 1.0,
        layer_id: Optional[int] = None,
    ) -> torch.Tensor:
        # assert self.is_metadata_initialized, "Metadata is not initialized."

        if self.is_profiling_iteration:
            # there is no need to call attention in profiling mode
            return torch.zeros_like(query)

        with self.get_timer(OperationMetrics.ATTN_INPUT_RESHAPE, layer_id):
            query = query.contiguous().reshape(-1, self.num_q_heads, self.head_dim)
            key = key.contiguous().reshape(-1, self.num_kv_heads, self.head_dim)
            value = value.contiguous().reshape(-1, self.num_kv_heads, self.head_dim)

        with self.get_timer(OperationMetrics.ATTN_KV_CACHE_SAVE, layer_id):
            if not self.append_qo_indptr_buf.shape[0] == 1:
                append_paged_kv_cache(
                    key,
                    value,
                    self.append_qo_indptr_buf,
                    kv_cache,
                    self.append_kv_page_indices_buf,
                    self.append_kv_page_indptr_buf,
                    self.append_kv_last_page_len_buf,
                    kv_layout="NHD",
                )

        causal = (
            not self.contains_multi_group_prefill_seq
            or self.multi_group_seq_store_kv_cache
        )

        with self.get_timer(OperationMetrics.ATTN, layer_id):
            output, S = self.wrapper.forward_return_lse(
                query,
                kv_cache,
                pos_encoding_mode="NONE",
                sm_scale=softmax_scale,
                causal=causal,
            )

        if self.contains_multi_group_prefill_seq and not SKIP_ATTENTION_REDUCTION:
            multi_group_V = output[: self.multi_group_seq_prefill_len].unsqueeze(1)
            multi_group_S = S[: self.multi_group_seq_prefill_len].unsqueeze(1)
            multi_group_V = gather_from_cache_model_parallel_region(
                multi_group_V, self.multi_group_seq_group_ids
            )
            multi_group_S = gather_from_cache_model_parallel_region(
                multi_group_S, self.multi_group_seq_group_ids
            )
            multi_group_V, _ = merge_states(multi_group_V, multi_group_S)
            output[: self.multi_group_seq_prefill_len] = multi_group_V
        elif self.contains_multi_group_decode_seq and not SKIP_ATTENTION_REDUCTION:
            multi_group_V = output[-1].unsqueeze(0).unsqueeze(0)
            multi_group_S = S[-1].unsqueeze(0).unsqueeze(0)
            multi_group_V = gather_from_cache_model_parallel_region(
                multi_group_V, self.multi_group_seq_group_ids
            )
            multi_group_S = gather_from_cache_model_parallel_region(
                multi_group_S, self.multi_group_seq_group_ids
            )
            multi_group_V, _ = merge_states(multi_group_V, multi_group_S)
            output[-1] = multi_group_V

        with self.get_timer(OperationMetrics.ATTN_OUTPUT_RESHAPE, layer_id):
            output = output.reshape(-1, self.num_q_heads * self.head_dim)

        return output
