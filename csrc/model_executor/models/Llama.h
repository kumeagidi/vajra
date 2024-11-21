#pragma once

#include <torch/all.h>

#include "StdCommon.h"
#include "Logging.h"
#include "LinearLayers.h"
#include "NormLayers.h"
#include "RotaryEmbedding.h"
#include "FlashinferAttentionWrapper.h"
//==============================================================================
namespace sarathi
{
class LlamaMLP
{
    public:
        LlamaMLP(
            const std::shared_ptr<ColumnParallelLinear> spGateUpProj,
            const std::shared_ptr<RowParallelLinear> spDownProj
        );

        torch::Tensor Forward(const torch::Tensor& input /*[in]*/) const;

    private:
        const std::shared_ptr<ColumnParallelLinear> m_spGateUpProj;
        const std::shared_ptr<RowParallelLinear> m_spDownProj;
};
//==============================================================================
class LlamaAttention{
public:
    LlamaAttention(
        int nQSize,
        int nKvSize,
        float nScaling,
        const std::shared_ptr<ColumnParallelLinear> spQkvProj,
        const std::shared_ptr<RowParallelLinear> spOProj,
        const std::shared_ptr<RotaryEmbedding> spRotaryEmb,
        const std::shared_ptr<FlashInferAttentionWrapper> spFlashInfer
    );

    torch::Tensor Forward(
        const torch::Tensor& positions, /*[in]*/
        const torch::Tensor& hiddenStates, /*[in]*/
        torch::Tensor& kvCache /*[inout]*/
    ) const;

    private:
        int m_nQSize;
        int m_nKvSize;
        float m_nScaling;
        const std::shared_ptr<ColumnParallelLinear> m_spQkvProj;
        const std::shared_ptr<RowParallelLinear> m_spOProj;
        const std::shared_ptr<RotaryEmbedding> m_spRotaryEmb;
        const std::shared_ptr<FlashInferAttentionWrapper> m_spFlashInfer;
};
//==============================================================================
class LlamaDecoderLayer
{
public:
    LlamaDecoderLayer(
        const std::shared_ptr<LlamaAttention> spSelfAttn,
        const std::shared_ptr<LlamaMLP> spMlp,
        const std::shared_ptr<RMSNorm> spInputLayernorm,
        const std::shared_ptr<RMSNorm> spPostAttentionLayernorm
    );

    torch::Tensor Forward(
        const torch::Tensor& positions, /*[in]*/
        torch::Tensor& hiddenStates, /*[in]*/
        torch::Tensor& kvCache /*[inout]*/
    ) const;

private:
    const std::shared_ptr<LlamaAttention> m_spSelfAttn;
    const std::shared_ptr<LlamaMLP> m_spMlp;
    const std::shared_ptr<RMSNorm> m_spInputLayernorm;
    const std::shared_ptr<RMSNorm> m_spPostAttentionLayernorm;
};
//==============================================================================
class LlamaModel
{
public:
    LlamaModel(
        const std::shared_ptr<VocabParallelEmbedding> spEmbedTokens,
        const std::vector<std::shared_ptr<LlamaDecoderLayer>> vspLayers,
        const std::shared_ptr<RMSNorm> spNorm
    );

    torch::Tensor Forward(
        const torch::Tensor& positions /*[in]*/,
        torch::Tensor& hiddenStates /*[in]*/,
        std::vector<torch::Tensor> vKvCaches /*[inout]*/
    );

private:
    const std::shared_ptr<VocabParallelEmbedding> m_spEmbedTokens;
    const std::vector<std::shared_ptr<LlamaDecoderLayer>> m_vspLayers;
    const std::shared_ptr<RMSNorm> m_spNorm;
};
//==============================================================================
} // namespace sarathi
//==============================================================================
