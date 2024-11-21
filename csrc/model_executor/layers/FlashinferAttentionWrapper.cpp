#include "FlashinferAttentionWrapper.h"
//==============================================================================
using namespace sarathi;
//==============================================================================
FlashInferAttentionWrapper::FlashInferAttentionWrapper(
    BatchPrefillWithPagedKVCachePyTorchWrapper& wrapper,
    const std::shared_ptr<ProcessGroupWrapper> spProcessGroupWrapper,
    int nNumQHeads,
    int nNumKvHeads,
    int nHeadDim,
    int nRank,
    unsigned int nLayout,
    float nSoftmaxScale,
    unsigned int nPosEncodingMode,
    bool bAllowFp16QKReduction,
    int nWindowLeft,
    float nLogitsSoftCap,
    float nRopeScale,
    float nRopeTheta,
    bool bReturnLSE,
    bool bSkipAttentionReduction
) :
    m_wrapper(wrapper),
    m_spProcessGroupWrapper(spProcessGroupWrapper),
    m_nNumQHeads(nNumQHeads),
    m_nNumKvHeads(nNumKvHeads),
    m_nHeadDim(nHeadDim),
    m_nRank(nRank),
    m_nLayout(nLayout),
    m_nSoftmaxScale(nSoftmaxScale),
    m_nPosEncodingMode(nPosEncodingMode),
    m_bAllowFp16QKReduction(bAllowFp16QKReduction),
    m_nWindowLeft(nWindowLeft),
    m_nLogitsSoftCap(nLogitsSoftCap),
    m_nRopeScale(nRopeScale),
    m_nRopeTheta(nRopeTheta),
    m_bReturnLSE(bReturnLSE),
    m_bSkipAttentionReduction(bSkipAttentionReduction)
{}
//==============================================================================
void FlashInferAttentionWrapper::BeginForward(
    bool bAppendKvRequired,
    bool bContainsMultiGroupPrefillSeq,
    bool bContainsMultiGroupDecodeSeq,
    unsigned int nMultiGroupSeqPrefillLen,
    const std::set<int>& vMultiGroupSeqGroupIds,
    const torch::Tensor& qoIndptr,
    const torch::Tensor& kvIndptr,
    const torch::Tensor& kvIndices,
    const torch::Tensor& kvLastPageLen,
    const torch::Tensor& appendQoIndptr,
    const torch::Tensor& appendKvIndices,
    const torch::Tensor& appendKvIndptr,
    const torch::Tensor& appendKvLastPageLen
)
{
    ASSERT(!m_bBeginForwardCalled);

    ASSERT(!(bContainsMultiGroupPrefillSeq && bContainsMultiGroupDecodeSeq));

    if (bContainsMultiGroupPrefillSeq)
    {
        ASSERT(nMultiGroupSeqPrefillLen > 0);
        ASSERT(!vMultiGroupSeqGroupIds.empty());
    }

    if (bContainsMultiGroupDecodeSeq)
    {
        ASSERT(!vMultiGroupSeqGroupIds.empty());
    }

    m_bBeginForwardCalled = true;

    m_bAppendKvRequired = bAppendKvRequired;
    m_bContainsMultiGroupPrefillSeq = bContainsMultiGroupPrefillSeq;
    m_bContainsMultiGroupDecodeSeq = bContainsMultiGroupDecodeSeq;
    m_nMultiGroupSeqPrefillLen = nMultiGroupSeqPrefillLen;
    m_vMultiGroupSeqGroupIds = vMultiGroupSeqGroupIds;
    m_qoIndptr = qoIndptr;
    m_kvIndptr = kvIndptr;
    m_kvIndices = kvIndices;
    m_kvLastPageLen = kvLastPageLen;
    m_appendQoIndptr = appendQoIndptr;
    m_appendKvIndices = appendKvIndices;
    m_appendKvIndptr = appendKvIndptr;
    m_appendKvLastPageLen = appendKvLastPageLen;
}
//==============================================================================
void FlashInferAttentionWrapper::EndForward()
{
    ASSERT(m_bBeginForwardCalled);

    m_bBeginForwardCalled = false;
}
//==============================================================================
torch::Tensor FlashInferAttentionWrapper::Forward(
    float nSoftmaxScale /*[in]*/,
    const torch::Tensor& q /*[in]*/,
    const torch::Tensor& k /*[in]*/,
    const torch::Tensor& v /*[in]*/,
    const torch::Tensor& kvCache /*[inout]*/
)
{
    ASSERT(m_bBeginForwardCalled);

    auto _q = q.contiguous().reshape({-1, m_nNumQHeads, m_nHeadDim});
    auto appendKey = k.contiguous().reshape({-1, m_nNumKvHeads, m_nHeadDim});
    auto appendValue = v.contiguous().reshape({-1, m_nNumKvHeads, m_nHeadDim});

    if (m_bAppendKvRequired){
        append_paged_kv_cache(
            appendKey,
            appendValue,
            m_appendQoIndptr,
            kvCache,
            std::nullopt /*paged_k_cache*/,
            std::nullopt /*paged_v_cache*/,
            m_appendKvIndices,
            m_appendKvIndptr,
            m_appendKvLastPageLen,
            m_nLayout
        );
    }

    auto outputAndLse = m_wrapper.Run(
        _q,
        m_qoIndptr,
        kvCache,
        std::nullopt /*paged_k_cache*/,
        std::nullopt /*paged_v_cache*/,
        m_kvIndptr,
        m_kvIndices,
        m_kvLastPageLen,
        m_bCausal,
        m_nPosEncodingMode,
        m_bAllowFp16QKReduction,
        m_nWindowLeft,
        m_nLogitsSoftCap,
        m_nSoftmaxScale,
        m_nRopeScale,
        m_nRopeTheta,
        m_bReturnLSE
    );

    auto output = outputAndLse[0];
    auto S = outputAndLse[1];

    if (m_bSkipAttentionReduction)
    {
        output = output.reshape({-1, m_nNumQHeads * m_nHeadDim});
        return output;
    }

    if (m_bContainsMultiGroupPrefillSeq)
    {
        auto processGroup = m_spProcessGroupWrapper->GetCacheModelParallelGroup(m_vMultiGroupSeqGroupIds);
        auto vMultiGroupV = output.slice(0, 0, m_nMultiGroupSeqPrefillLen).unsqueeze(1);
        auto bMultiGroupS = S.slice(0, 0, m_nMultiGroupSeqPrefillLen).unsqueeze(1);
        
        vMultiGroupV = ParallelOps::GatherFromCacheModelParallelRegion(vMultiGroupV, m_nRank, processGroup);
        bMultiGroupS = ParallelOps::GatherFromCacheModelParallelRegion(bMultiGroupS, m_nRank, processGroup);

        auto mergedOutput = merge_states(vMultiGroupV, bMultiGroupS);
        output.slice(0, 0, m_nMultiGroupSeqPrefillLen) = mergedOutput[0];
    }
    else if (m_bContainsMultiGroupDecodeSeq)
    {    
        auto processGroup = m_spProcessGroupWrapper->GetCacheModelParallelGroup(m_vMultiGroupSeqGroupIds);

        auto vMultiGroupV = output.slice(0, -1).unsqueeze(0);
        auto vMultiGroupS = S.slice(0, -1).unsqueeze(0);

        vMultiGroupV = ParallelOps::GatherFromCacheModelParallelRegion(vMultiGroupV, m_nRank, processGroup);
        vMultiGroupS = ParallelOps::GatherFromCacheModelParallelRegion(vMultiGroupS, m_nRank, processGroup);

        auto mergedOutput = merge_states(vMultiGroupV, vMultiGroupS);
        output.slice(0, -1) = mergedOutput[0];   
    }

    output = output.reshape({-1, m_nNumQHeads * m_nHeadDim});
    return output;
}
//==============================================================================