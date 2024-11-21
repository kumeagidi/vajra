#pragma once

#include <torch/all.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include "StdCommon.h"
#include "Logging.h"
//==============================================================================
namespace sarathi
{
//==============================================================================
class ProcessGroupWrapper
{
public:
    ProcessGroupWrapper(
        c10::intrusive_ptr<c10d::ProcessGroup> spTensorModelParallelGroup,
        c10::intrusive_ptr<c10d::ProcessGroup> spPipelineModelParallelGroup,
        const std::unordered_map<std::set<int>, c10::intrusive_ptr<c10d::ProcessGroup>>& mpCacheModelParallelGroups
    );

    c10::intrusive_ptr<c10d::ProcessGroup> GetTensorModelParallelGroup() const;
    c10::intrusive_ptr<c10d::ProcessGroup> GetPipelineModelParallelGroup() const;
    c10::intrusive_ptr<c10d::ProcessGroup> GetCacheModelParallelGroup(const std::set<int>& groupKey) const;

private:
    c10::intrusive_ptr<c10d::ProcessGroup> m_spTensorModelParallelGroup;
    c10::intrusive_ptr<c10d::ProcessGroup> m_spPipelineModelParallelGroup;
    std::unordered_map<std::set<int>, c10::intrusive_ptr<c10d::ProcessGroup>> m_mpCacheModelParallelGroups;
};
//==============================================================================
} // namespace sarathi
//==============================================================================