#include "Llama.h"
#include "ops.h"
//==============================================================================
using namespace sarathi;
//==============================================================================
LlamaMLP::LlamaMLP(
    const std::shared_ptr<ColumnParallelLinear> spGateUpProj,
    const std::shared_ptr<RowParallelLinear> spDownProj
) : 
    m_spGateUpProj(spGateUpProj),
    m_spDownProj(spDownProj)
{
    ASSERT(m_spGateUpProj);
    ASSERT(m_spDownProj);
}
//==============================================================================
torch::Tensor LlamaMLP::Forward(const torch::Tensor& input /*[in]*/) const
{
    auto gateUp = m_spGateUpProj->Forward(input);
    int numTokens = input.size(0);
    int ouputEmbeddingSize = gateUp.size(1) / 2;
    auto activated = torch::empty({numTokens, ouputEmbeddingSize}, input.options());
    silu_and_mul(activated, gateUp);
    return m_spDownProj->Forward(activated);
}
//==============================================================================
LlamaAttention::LlamaAttention(
    int nQSize,
    int nKvSize,
    float nScaling,
    const std::shared_ptr<ColumnParallelLinear> spQkvProj,
    const std::shared_ptr<RowParallelLinear> spOProj,
    const std::shared_ptr<RotaryEmbedding> spRotaryEmb,
    const std::shared_ptr<FlashInferAttentionWrapper> spFlashInfer
) : 
    m_nQSize(nQSize),
    m_nKvSize(nKvSize),
    m_nScaling(nScaling),
    m_spQkvProj(spQkvProj),
    m_spOProj(spOProj),
    m_spRotaryEmb(spRotaryEmb),
    m_spFlashInfer(spFlashInfer)
{
    ASSERT(m_spQkvProj);
    ASSERT(m_spOProj);
    ASSERT(m_spRotaryEmb);
    ASSERT(m_spFlashInfer);
};
// ============================================================================
torch::Tensor LlamaAttention::Forward(
    const torch::Tensor& positions, /*[in]*/
    const torch::Tensor& hiddenStates, /*[in]*/
    torch::Tensor& kvCache /*[inout]*/
) const
{
    auto qkv = m_spQkvProj->Forward(hiddenStates);
    auto qkvSplit = torch::split(qkv, {m_nQSize, m_nKvSize, m_nKvSize}, -1);
    auto q = qkvSplit[0];
    auto k = qkvSplit[1];
    auto v = qkvSplit[2];

    m_spRotaryEmb->Forward(positions, q, k);

    auto attnOutput = m_spFlashInfer->Forward(
        m_nScaling, q, k, v, kvCache
    );

    return m_spOProj->Forward(attnOutput);
}
//==============================================================================
LlamaDecoderLayer::LlamaDecoderLayer(
    const std::shared_ptr<LlamaAttention> spSelfAttn,
    const std::shared_ptr<LlamaMLP> spMlp,
    const std::shared_ptr<RMSNorm> spInputLayernorm,
    const std::shared_ptr<RMSNorm> spPostAttentionLayernorm
) : 
    m_spSelfAttn(spSelfAttn),
    m_spMlp(spMlp),
    m_spInputLayernorm(spInputLayernorm),
    m_spPostAttentionLayernorm(spPostAttentionLayernorm)
{
    ASSERT(m_spSelfAttn);
    ASSERT(m_spMlp);
    ASSERT(m_spInputLayernorm);
    ASSERT(m_spPostAttentionLayernorm);
}
//==============================================================================
torch::Tensor LlamaDecoderLayer::Forward(
    const torch::Tensor& positions, /*[in]*/
    torch::Tensor& hiddenStates, /*[in]*/
    torch::Tensor& kvCache /*[inout]*/
) const
{
    auto residual = hiddenStates;
    hiddenStates = m_spInputLayernorm->Forward(hiddenStates);
    hiddenStates = m_spSelfAttn->Forward(positions, hiddenStates, kvCache);
    hiddenStates = residual + hiddenStates;

    residual = hiddenStates;
    hiddenStates = m_spPostAttentionLayernorm->Forward(hiddenStates);
    hiddenStates = m_spMlp->Forward(hiddenStates);
    hiddenStates = residual + hiddenStates;

    return hiddenStates;
}
//==============================================================================
LlamaModel::LlamaModel(
    const std::shared_ptr<VocabParallelEmbedding> spEmbedTokens,
    const std::vector<std::shared_ptr<LlamaDecoderLayer>> vspLayers,
    const std::shared_ptr<RMSNorm> spNorm
) :
    m_spEmbedTokens(spEmbedTokens),
    m_vspLayers(vspLayers),
    m_spNorm(spNorm)
{
    ASSERT(m_vspLayers.size() > 0);
}
//==============================================================================
torch::Tensor LlamaModel::Forward(
    const torch::Tensor& positions /*[in]*/,
    torch::Tensor& hiddenStates /*[in]*/,
    std::vector<torch::Tensor> vKvCaches /*[inout]*/
)
{
    ASSERT(vKvCaches.size() == m_vspLayers.size());

    if (m_spEmbedTokens)
    {
        hiddenStates = m_spEmbedTokens->Forward(hiddenStates);
    }

    for (int i = 0; i < m_vspLayers.size(); i++)
    {
        hiddenStates = m_vspLayers[i]->Forward(positions, hiddenStates, vKvCaches[i]);
    }

    if (m_spNorm)
    {
        hiddenStates = m_spNorm->Forward(hiddenStates);
    }

    return hiddenStates;
}
//==============================================================================
