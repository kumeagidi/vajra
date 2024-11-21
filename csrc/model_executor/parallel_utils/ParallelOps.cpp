#include "ParallelOps.h"
//==============================================================================
using namespace sarathi;
//==============================================================================
std::vector<torch::Tensor> ParallelOps::SplitTensorAlongLastDim(
    const torch::Tensor& input /*[in]*/,
    int64_t nNumPartitions /*[in]*/,
    bool bContiguousSplitChunks /*[in]*/
)
{
    int nLastDim = input.dim() - 1;
    int nLastDimSize = input.size(nLastDim) / nNumPartitions;
    // Split
    auto vTensorList = torch::split(input, nLastDimSize, nLastDim);
    // Note: torch.split does not create contiguous tensors by default.
    if (bContiguousSplitChunks)
    {
        for (auto& tensor : vTensorList)
        {
            tensor = tensor.contiguous();
        }
    }
    return vTensorList;
}
//==============================================================================
torch::Tensor ParallelOps::ReduceFromCacheModelParallelRegion(
    torch::Tensor& input  /*[inout]*/,
    c10::intrusive_ptr<c10d::ProcessGroup> spProcessGroup /*[in]*/)
{
    int nWorldSize = spProcessGroup->getSize();
    if (nWorldSize == 1)
    {
        return input;
    }

    std::vector<at::Tensor> vInput{input};

    auto work = spProcessGroup->allreduce(
        vInput,
        c10d::AllreduceOptions()
    );
    work->wait();
    return input;
}
//==============================================================================
torch::Tensor ParallelOps::ReduceFromTensorModelParallelRegion(
    torch::Tensor& input /*[inout]*/,
    c10::intrusive_ptr<c10d::ProcessGroup> spProcessGroup /*[in]*/)
{
    int nWorldSize = spProcessGroup->getSize();
    if (nWorldSize == 1)
    {
        return input;
    }

    std::vector<at::Tensor> vInput{input};

    auto work = spProcessGroup->allreduce(
        vInput,
        c10d::AllreduceOptions()
    );
    work->wait();
    return input;
}
//==============================================================================
torch::Tensor ParallelOps::ScatterToTensorModelParallelRegion(
    const torch::Tensor& input /*[in]*/,
    c10::intrusive_ptr<c10d::ProcessGroup> spProcessGroup /*[in]*/)
{
    int nRank = spProcessGroup->getRank();
    int nWorldSize = spProcessGroup->getSize();
    if (nWorldSize == 1)
    {
        return input;
    }
    std::vector<at::Tensor> vInputList = SplitTensorAlongLastDim(input, nWorldSize, false);
    return vInputList[nRank].contiguous();
}
//==============================================================================
torch::Tensor ParallelOps::GatherFromGroup(
    const torch::Tensor& input /*[in]*/,
    int nIndexRank /*[in]*/,
    int nConcatDim /*[in]*/,
    c10::intrusive_ptr<c10d::ProcessGroup> spProcessGroup /*[in]*/)
{
    int nWorldSize = spProcessGroup->getSize();

    std::vector<at::Tensor> vTensorList(nWorldSize, torch::empty_like(input));
    vTensorList[nIndexRank] = input;

    std::vector<std::vector<at::Tensor>> vvTensorList{vTensorList};
    std::vector<at::Tensor> vInput{input};

    auto work = spProcessGroup->allgather(
        vvTensorList,
        vInput
    );
    work->wait();
    return torch::cat(vTensorList, nConcatDim);
}
//==============================================================================
torch::Tensor ParallelOps::GatherFromTensorModelParallelRegion(
    const torch::Tensor& input /*[in]*/,
    c10::intrusive_ptr<c10d::ProcessGroup> spProcessGroup /*[in]*/)
{
    int nWorldSize = spProcessGroup->getSize();
    if (nWorldSize == 1)
    {
        return input;
    }

    std::vector<at::Tensor> vOutputTensors(nWorldSize, torch::empty_like(input));

    std::vector<std::vector<at::Tensor>> vvOutputTensors{vOutputTensors};
    std::vector<at::Tensor> vInput{input};

    auto work = spProcessGroup->allgather(
        vvOutputTensors,
        vInput
    );
    work->wait();
    return torch::cat(vOutputTensors, input.dim() - 1 /*nLastDim*/);
}
//==============================================================================
torch::Tensor ParallelOps::GatherFromCacheModelParallelRegion(
    const torch::Tensor& input /*[in]*/,
    int nIndexRank /*[in]*/,
    c10::intrusive_ptr<c10d::ProcessGroup> spProcessGroup /*[in]*/)
{
    int nWorldSize = spProcessGroup->getSize();
    if (nWorldSize == 1)
    {
        return input;
    }

    return GatherFromGroup(input, nIndexRank, 1 /*nConcatDim*/, spProcessGroup);
}
//==============================================================================
void ParallelOps::SendToNextPipelineStage(
    const torch::Tensor& input /*[in]*/,
    int nDstRank /*[in]*/,
    c10::intrusive_ptr<c10d::ProcessGroup> spProcessGroup /*[in]*/)
{
    int nWorldSize = spProcessGroup->getSize();
    if (nWorldSize == 1)
    {
        return;
    }

    std::vector<at::Tensor> vInput{input};

    auto work = spProcessGroup->send(vInput, nDstRank, 0 /*tag*/);
    work->wait();
}
//==============================================================================
void ParallelOps::RecvFromLastPipelineStage(
    torch::Tensor& output, /*[out]*/
    int nSrcRank /*[in]*/,
    c10::intrusive_ptr<c10d::ProcessGroup> spProcessGroup /*[in]*/)
{
    int nWorldSize = spProcessGroup->getSize();
    if (nWorldSize == 1)
    {
        return;
    }

    std::vector<at::Tensor> vOutput{output};
    auto work = spProcessGroup->recv(vOutput, nSrcRank, 0 /*tag*/);
    work->wait();
}
//==============================================================================
