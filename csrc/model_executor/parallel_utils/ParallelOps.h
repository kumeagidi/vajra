#pragma once

#include <torch/all.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include "StdCommon.h"
#include "Logging.h"
//==============================================================================
namespace sarathi
{
//==============================================================================
class ParallelOps
{
public:
    static std::vector<torch::Tensor> SplitTensorAlongLastDim(
        const torch::Tensor& input /*[in]*/,
        int64_t nNumPartitions /*[in]*/,
        bool bContiguousSplitChunks = false /*[in]*/
    );

    static torch::Tensor ReduceFromCacheModelParallelRegion(
        torch::Tensor& input  /*[inout]*/,
        c10::intrusive_ptr<c10d::ProcessGroup> spProcessGroup /*[in]*/
    );

    static torch::Tensor ReduceFromTensorModelParallelRegion(
        torch::Tensor& input /*[inout]*/,
        c10::intrusive_ptr<c10d::ProcessGroup> spProcessGroup /*[in]*/
    );

    static torch::Tensor ScatterToTensorModelParallelRegion(
        const torch::Tensor& input /*[in]*/,
        c10::intrusive_ptr<c10d::ProcessGroup> spProcessGroup /*[in]*/
    );

    static torch::Tensor GatherFromGroup(
        const torch::Tensor& input /*[in]*/,
        int nIndexRank /*[in]*/,
        int nConcatDim /*[in]*/,
        c10::intrusive_ptr<c10d::ProcessGroup> spProcessGroup /*[in]*/
    );

    static torch::Tensor GatherFromTensorModelParallelRegion(
        const torch::Tensor& input /*[in]*/,
        c10::intrusive_ptr<c10d::ProcessGroup> spProcessGroup /*[in]*/
    );

    static torch::Tensor GatherFromCacheModelParallelRegion(
        const torch::Tensor& input /*[in]*/,
        int nIndexRank /*[in]*/,
        c10::intrusive_ptr<c10d::ProcessGroup> spProcessGroup /*[in]*/
    );

    static void SendToNextPipelineStage(
        const torch::Tensor& input /*[in]*/,
        int nDstRank /*[in]*/,
        c10::intrusive_ptr<c10d::ProcessGroup> spProcessGroup /*[in]*/
    );

    static void RecvFromLastPipelineStage(
        torch::Tensor& output /*[out]*/,
        int nSrcRank /*[in]*/,
        c10::intrusive_ptr<c10d::ProcessGroup> spProcessGroup /*[in]*/
    );
};
//==============================================================================
} // namespace sarathi
//==============================================================================