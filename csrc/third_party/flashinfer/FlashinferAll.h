#include <torch/all.h>
//==============================================================================
// We are copying some class definations from flashinfer so that we don't
// need to have all the header dependencies of flashinfer in this project.
//==============================================================================
class BatchPrefillWithPagedKVCachePyTorchWrapper {
public:
    /*
    Return value:
    if (return_lse) { return {o, lse}; } 
    else { return {o}; } 
    */
    void Plan(
        torch::Tensor float_workspace_buffer, 
        torch::Tensor int_workspace_buffer,
        torch::Tensor qo_indptr, 
        torch::Tensor page_kv_indptr, 
        unsigned int batch_size,
        unsigned int num_qo_heads, 
        unsigned int num_kv_heads, 
        unsigned int head_dim,
        unsigned page_size, 
        torch::Tensor empty_q_data
    );

    bool IsCUDAGraphEnabled() const;

    std::vector<torch::Tensor> Run(
        torch::Tensor q, 
        torch::Tensor qo_indptr,
        std::optional<torch::Tensor> paged_kv_cache,
        std::optional<torch::Tensor> paged_k_cache,
        std::optional<torch::Tensor> paged_v_cache,
        torch::Tensor paged_kv_indptr, 
        torch::Tensor paged_kv_indices,
        torch::Tensor paged_kv_last_page_len, 
        bool causal,
        unsigned int pos_encoding_mode, 
        bool allow_fp16_qk_reduction,
        int window_left, 
        float logits_soft_cap, 
        float sm_scale,
        float rope_scale, 
        float rope_theta, 
        bool return_lse
    );
};
//==============================================================================
void append_paged_kv_cache(
    torch::Tensor append_key, 
    torch::Tensor append_value,
    torch::Tensor append_indptr, 
    std::optional<torch::Tensor> paged_kv_cache,
    std::optional<torch::Tensor> paged_k_cache,
    std::optional<torch::Tensor> paged_v_cache, 
    torch::Tensor kv_indices,
    torch::Tensor kv_indptr, 
    torch::Tensor kv_last_page_len,
    unsigned int layout
);
//==============================================================================
std::vector<torch::Tensor> merge_states(torch::Tensor v, torch::Tensor s);
//==============================================================================
