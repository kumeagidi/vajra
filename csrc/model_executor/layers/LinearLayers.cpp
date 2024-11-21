#include "LinearLayers.h"
#include "ops.h"
//==============================================================================
using namespace sarathi;
//==============================================================================
ColumnParallelLinear::ColumnParallelLinear(
    int nInputSize,
    int nOutputSize,
    bool bGatherOutput,
    int nWorldSize,
    bool bSkipBiasAdd,
    const torch::Tensor& weight,
    const std::optional<torch::Tensor>& oBias,
    const std::shared_ptr<ProcessGroupWrapper> spProcessGroupWrapper
) :
    m_nInputSize(nInputSize),
    m_nOutputSize(nOutputSize),
    m_bGatherOutput(bGatherOutput),
    m_nOutputSizePerPartition(nOutputSize / nWorldSize),
    m_nWorldSize(nWorldSize),
    m_bSkipBiasAdd(bSkipBiasAdd),
    m_weight(weight),
    m_oBias(oBias),
    m_spProcessGroup(spProcessGroupWrapper->GetTensorModelParallelGroup())
{}
//==============================================================================
torch::Tensor ColumnParallelLinear::Forward(const torch::Tensor& input /*[in]*/) const
{
    auto outputParallel = torch::linear(input, m_weight, m_oBias);

    if (m_bGatherOutput)
    {
        outputParallel = ParallelOps::GatherFromTensorModelParallelRegion(outputParallel, m_spProcessGroup);
    }

    return outputParallel;
}
//==============================================================================
RowParallelLinear::RowParallelLinear(
    int nInputSize,
    int nOutputSize,
    bool bInputIsParallel,
    bool bReduceResults,
    int nWorldSize,
    int nInputSizePerPartition,
    bool bSkipBiasAdd,
    const torch::Tensor& weight,
    const std::optional<torch::Tensor>& oBias,
    const std::shared_ptr<ProcessGroupWrapper> spProcessGroupWrapper
) :
    m_nInputSize(nInputSize),
    m_nOutputSize(nOutputSize),
    m_bInputIsParallel(bInputIsParallel),
    m_bReduceResults(bReduceResults),
    m_nWorldSize(nWorldSize),
    m_nInputSizePerPartition(nInputSizePerPartition),
    m_bSkipBiasAdd(bSkipBiasAdd),
    m_weight(weight),
    m_oBias(oBias),
    m_spProcessGroup(spProcessGroupWrapper->GetTensorModelParallelGroup())
{}
//==============================================================================
torch::Tensor RowParallelLinear::Forward(const torch::Tensor& input /*[in]*/) const
{
    auto inputParallel = input;
    if (!m_bInputIsParallel)
    {
        inputParallel = ParallelOps::ScatterToTensorModelParallelRegion(input, m_spProcessGroup);
    }

    auto outputParallel = torch::matmul(inputParallel, m_weight.t());
    auto output = outputParallel;
    if (m_bReduceResults && m_nWorldSize > 1)
    {
        output = ParallelOps::ReduceFromTensorModelParallelRegion(outputParallel, m_spProcessGroup);
    }

    if (!m_bSkipBiasAdd && m_oBias.has_value())
    {
        output.add_(m_oBias.value());
    }

    return output;
}
//==============================================================================
VocabParallelEmbedding::VocabParallelEmbedding(
    int nNumEmbeddings,
    int nEmbeddingDim,
    int nTensorModelParallelSize,
    int nRank,
    bool bReduceResults,
    int nVocabStartIndex,
    int nVocabEndIndex,
    int nNumEmbeddingsPerPartition,
    const torch::Tensor& weight,
    const std::shared_ptr<ProcessGroupWrapper> spProcessGroupWrapper
) :
    m_nNumEmbeddings(nNumEmbeddings),
    m_nEmbeddingDim(nEmbeddingDim),
    m_nTensorModelParallelSize(nTensorModelParallelSize),
    m_nRank(nRank),
    m_bReduceResults(bReduceResults),
    m_nVocabStartIndex(nVocabStartIndex),
    m_nVocabEndIndex(nVocabEndIndex),
    m_nNumEmbeddingsPerPartition(nNumEmbeddingsPerPartition),
    m_weight(weight),
    m_spProcessGroup(spProcessGroupWrapper->GetTensorModelParallelGroup())
{}
//==============================================================================
torch::Tensor VocabParallelEmbedding::Forward(const torch::Tensor& input /*[in]*/) const
{
    int nWorldSize = m_spProcessGroup->getSize();
    auto maskedInput = input;

    if (nWorldSize > 1)
    {
        auto inputMask = (input < m_nVocabStartIndex) | (input >= m_nVocabEndIndex);
        maskedInput = input.clone() - m_nVocabStartIndex;
        maskedInput = maskedInput.masked_fill_(inputMask, 0);
    }

    auto options = torch::nn::functional::EmbeddingFuncOptions()
        .norm_type(2.0)
        .scale_grad_by_freq(false)
        .sparse(false);


    auto outputParallel = torch::nn::functional::embedding(
        maskedInput,
        m_weight,
        options
    );

    if (nWorldSize > 1)
    {
        auto inputMask = (input < m_nVocabStartIndex) | (input >= m_nVocabEndIndex);
        outputParallel.index_put_({inputMask, torch::indexing::Ellipsis}, 0.0);
    }

    auto output = outputParallel;
    if (m_bReduceResults && nWorldSize > 1)
    {
        output = ParallelOps::ReduceFromTensorModelParallelRegion(outputParallel, m_spProcessGroup);
    }

    return output;
}
//==============================================================================