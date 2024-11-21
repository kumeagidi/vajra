#pragma once

#include <torch/all.h>

#include "StdCommon.h"
#include "Logging.h"
#include "ProcessGroupWrapper.h"
#include "ParallelOps.h"
//==============================================================================
namespace sarathi
{
//==============================================================================
class ColumnParallelLinear
{
public:
    ColumnParallelLinear(
        int nInputSize,
        int nOutputSize,
        bool bGatherOutput,
        int nWorldSize,
        bool bSkipBiasAdd,
        const torch::Tensor& weight,
        const std::optional<torch::Tensor>& oBias,
        const std::shared_ptr<ProcessGroupWrapper> spProcessGroupWrapper
    );

    torch::Tensor Forward(const torch::Tensor& input /*[in]*/) const;

private:
    int m_nInputSize;
    int m_nOutputSize;
    bool m_bGatherOutput;
    int m_nWorldSize;
    int m_nOutputSizePerPartition;
    bool m_bSkipBiasAdd;
    const torch::Tensor m_weight;
    const std::optional<torch::Tensor> m_oBias;
    const c10::intrusive_ptr<c10d::ProcessGroup> m_spProcessGroup;
};
//==============================================================================
class RowParallelLinear
{
public:
    RowParallelLinear(
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
    );

    torch::Tensor Forward(const torch::Tensor& input /*[in]*/) const;

private:
    int m_nInputSize;
    int m_nOutputSize;
    bool m_bInputIsParallel;
    bool m_bReduceResults;
    int m_nWorldSize;
    int m_nInputSizePerPartition;
    bool m_bSkipBiasAdd;
    const torch::Tensor m_weight;
    const std::optional<torch::Tensor> m_oBias;
    const c10::intrusive_ptr<c10d::ProcessGroup> m_spProcessGroup;
};
//==============================================================================
class VocabParallelEmbedding
{
public:
    VocabParallelEmbedding(
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
    );

    torch::Tensor Forward(const torch::Tensor& input /*[in]*/) const;

private:
    int m_nNumEmbeddings;
    int m_nEmbeddingDim;
    int m_nTensorModelParallelSize;
    int m_nRank;
    bool m_bReduceResults;
    int m_nVocabStartIndex;
    int m_nVocabEndIndex;
    int m_nNumEmbeddingsPerPartition;
    const torch::Tensor m_weight;
    const c10::intrusive_ptr<c10d::ProcessGroup> m_spProcessGroup;
};
//==============================================================================
} // namespace sarathi
//==============================================================================