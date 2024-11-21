#pragma once

#include <torch/all.h>

#include "StdCommon.h"
#include "Logging.h"

#include "ProcessGroupWrapper.h"
#include "ParallelOps.h"

#include "FlashinferAll.h"
//==============================================================================
namespace sarathi
{
//==============================================================================
class FlashInferAttentionWrapper {
public:
    FlashInferAttentionWrapper(
        BatchPrefillWithPagedKVCachePyTorchWrapper& wrapper,
        const std::shared_ptr<ProcessGroupWrapper> spProcessGroupWrapper,
        int nNumQHeads,
        int nNumKvHeads,
        int nHeadDim,
        int nRank,
        unsigned int nLayout = 0,
        float nSoftmaxScale = 1,
        unsigned int nPosEncodingMode = 0,
        bool bAllowFp16QKReduction = false,
        int nWindowLeft = -1,
        float nLogitsSoftCap = 0.0,
        float nRopeScale = 1.0,
        float nRopeTheta = 10000.0,
        bool bReturnLSE = false,
        bool bSkipAttentionReduction = false
    );

    void BeginForward(
        bool bAppendKvRequired /*[in]*/,
        bool bContainsMultiGroupPrefillSeq /*[in]*/,
        bool bContainsMultiGroupDecodeSeq /*[in]*/,
        unsigned int nMultiGroupSeqPrefillLen /*[in]*/,
        const std::set<int>& vMultiGroupSeqGroupIds /*[in]*/ ,
        const torch::Tensor& qoIndptr /*[in]*/,
        const torch::Tensor& kvIndptr /*[in]*/,
        const torch::Tensor& kvIndices /*[in]*/,
        const torch::Tensor& kvLastPageLen /*[in]*/,
        const torch::Tensor& appendQoIndptr /*[in]*/,
        const torch::Tensor& appendKvIndices /*[in]*/,
        const torch::Tensor& appendKvIndptr /*[in]*/,
        const torch::Tensor& appendKvLastPageLen /*[in]*/
    );

    void EndForward();

    torch::Tensor Forward(
        float nSoftmaxScale /*[in]*/,
        const torch::Tensor& q /*[in]*/,
        const torch::Tensor& k /*[in]*/,
        const torch::Tensor& v /*[in]*/,
        const torch::Tensor& kvCache /*[inout]*/
    );

private:
    // Initialization parameters
    BatchPrefillWithPagedKVCachePyTorchWrapper& m_wrapper;
    std::shared_ptr<ProcessGroupWrapper> m_spProcessGroupWrapper;
    int m_nNumQHeads;
    int m_nNumKvHeads;
    int m_nHeadDim;
    int m_nRank;
    unsigned int m_nLayout;
    float m_nSoftmaxScale;
    bool m_bCausal;
    unsigned int m_nPosEncodingMode;
    bool m_bAllowFp16QKReduction;
    int m_nWindowLeft;
    float m_nLogitsSoftCap;
    float m_nRopeScale;
    float m_nRopeTheta;
    bool m_bReturnLSE;
    bool m_bSkipAttentionReduction;

    // Runtime parameters
    bool m_bBeginForwardCalled;
    bool m_bAppendKvRequired;
    bool m_bContainsMultiGroupPrefillSeq;
    bool m_bContainsMultiGroupDecodeSeq;
    unsigned int m_nMultiGroupSeqPrefillLen;
    std::set<int> m_vMultiGroupSeqGroupIds;
    torch::Tensor m_qoIndptr;
    torch::Tensor m_kvIndptr;
    torch::Tensor m_kvIndices;
    torch::Tensor m_kvLastPageLen;
    torch::Tensor m_appendQoIndptr;
    torch::Tensor m_appendKvIndices;
    torch::Tensor m_appendKvIndptr;
    torch::Tensor m_appendKvLastPageLen;
};
//==============================================================================
} // namespace sarathi
//==============================================================================