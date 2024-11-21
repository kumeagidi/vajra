#pragma once

#include <torch/extension.h>

void silu_and_mul(
  torch::Tensor& out,
  const torch::Tensor& input);

void gelu_new(
  torch::Tensor& out,
  const torch::Tensor& input);

void gelu_fast(
  torch::Tensor& out,
  const torch::Tensor& input);

void rms_norm(
  torch::Tensor& out,
  const torch::Tensor& input,
  const torch::Tensor& weight,
  float epsilon);

void topk_softmax(
    torch::Tensor& topk_weights, 
    torch::Tensor& topk_indices,
    torch::Tensor& token_expert_indices,
    torch::Tensor& gating_output
);

void moe_align_block_size(
    torch::Tensor topk_ids, 
    int64_t num_experts,
    int64_t block_size, 
    torch::Tensor sorted_token_ids,
    torch::Tensor experts_ids,
    torch::Tensor num_tokens_post_pad
);

void rotary_embedding(
  const torch::Tensor& positions,
  torch::Tensor& query,
  torch::Tensor& key,
  int head_size,
  const torch::Tensor& cos_sin_cache,
  bool is_neox);