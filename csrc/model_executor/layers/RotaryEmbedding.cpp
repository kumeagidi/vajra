#include "RotaryEmbedding.h"
#include "ops.h"
//==============================================================================
using namespace sarathi;
//==============================================================================
RotaryEmbedding::RotaryEmbedding(
    int nHeadSize,
    int nRotaryDim,
    long nMaxPositionEmbeddings,
    long nBase,
    bool bIsNeoxStyle,
    const torch::Tensor& cosSinCache
) : 
    m_nHeadSize(nHeadSize),
    m_nRotaryDim(nRotaryDim),
    m_nMaxPositionEmbeddings(nMaxPositionEmbeddings),
    m_nBase(nBase),
    m_bIsNeoxStyle(bIsNeoxStyle),
    m_cosSinCache(cosSinCache)
{}
//==============================================================================
void RotaryEmbedding::Forward(
    const torch::Tensor& positions /*[in]*/,
    torch::Tensor& query /*[inout]*/,
    torch::Tensor& key /*[inout]*/
) const
{
    rotary_embedding(
        positions,
        query,
        key,
        m_nHeadSize,
        m_cosSinCache,
        m_bIsNeoxStyle
    );
}
//==============================================================================