#include "ProcessGroupWrapper.h"
//==============================================================================
using namespace sarathi;
//==============================================================================
ProcessGroupWrapper::ProcessGroupWrapper(
    c10::intrusive_ptr<c10d::ProcessGroup> spTensorModelParallelGroup,
    c10::intrusive_ptr<c10d::ProcessGroup> spPipelineModelParallelGroup,
    const std::unordered_map<std::set<int>, c10::intrusive_ptr<c10d::ProcessGroup>>& mpCacheModelParallelGroups
) :
    m_spTensorModelParallelGroup(spTensorModelParallelGroup),
    m_spPipelineModelParallelGroup(spPipelineModelParallelGroup),
    m_mpCacheModelParallelGroups(mpCacheModelParallelGroups)
{}
//==============================================================================
c10::intrusive_ptr<c10d::ProcessGroup> ProcessGroupWrapper::GetTensorModelParallelGroup() const
{
    return m_spTensorModelParallelGroup;
}
//==============================================================================
c10::intrusive_ptr<c10d::ProcessGroup> ProcessGroupWrapper::GetPipelineModelParallelGroup() const
{
    return m_spPipelineModelParallelGroup;
}
//==============================================================================
c10::intrusive_ptr<c10d::ProcessGroup> ProcessGroupWrapper::GetCacheModelParallelGroup(const std::set<int>& groupKey) const
{
    auto it = m_mpCacheModelParallelGroups.find(groupKey);
    if (it == m_mpCacheModelParallelGroups.end())
    {
        TRACE_CRITICAL_AND_EXIT("Cache model parallel group not found for group key");
    }
    return it->second;
}
//==============================================================================
