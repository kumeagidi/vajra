#include "base_scheduler.h"
#include "scheduler_outputs.h"
#include "sequence_with_priority.h"
#include <cmath>
#include <vector>
#include <queue>
#include <algorithm>
#include <memory>
#include <iostream>

using namespace sarathi;

SarathiScheduler::SarathiScheduler(
    pybind11::object model_config,
    pybind11::object scheduler_config,
    pybind11::object cache_config,
    pybind11::object parallel_config,
    pybind11::object waiting_queue,
    pybind11::object replica_seq_manager,
    pybind11::object metric_store
) :
    BaseScheduler(model_config, scheduler_config, cache_config, parallel_config, waiting_queue, replica_seq_manager, metric_store),
    chunk_size(pybind11::cast<int> (scheduler_config.attr("chunk_size"))),
    enable_dynamic_chunking_schedule(pybind11::cast<bool> scheduler_config.attr("enable_dynamic_chunking_schedule")),
    low_chunk_size(pybind11::cast<int> (scheduler_config.attr("low_hunk_size"))),
    high_chunk_size(pybind11::cast<int> (scheduler_config.attr("high_chunk_size"))),
    chunk_schedule_max_tokens(pybind11::cast<int> (scheduler_config.attr("chunk_schedule_max_tokens"))),
    chunk_schedule_stages(pybind11::cast<int> (scheduler_config.attr("chunk_schedule_stages")))
{
    if (enable_dynamic_chunking_schedule) {
        this->_chunk_sizes = _compute_chunk_size_schedule();
        this->_tokens_per_stage = std::ceil(chunk_schedule_max_tokens / pybind11::cast<float> (chunk_schedule_stages));
    }
}

std::vector<int> SarathiScheduler::compute_chunk_size_schedule() {}

pybind11::object SarathiScheduler::get_block_space_manager_class() {
    py::module sarathi_block_space_manager = py::module::import("sarathi.core.block_space_manager.sarathi_block_space_manager");
    return sarathi_block_space_manager.attr("SarathiBlockSpaceManager")();
}

int SarathiScheduler::get_seq_next_num_prefill_tokens(pybind11::object& seq, int num_batched_tokens) 
{
    if (enable_dynamic_chunking_schedule) {
        int request_stage_idx = std::ceil(pybind11::cast<int> (seq.attr("get_num_prompt_tokens_stage_processed")()) / _tokens_per_stage);
        //assert
        chunk_size = _chunk_sizes[request_stage_idx];
    }

    int next_num_tokens = std::min(
        pybind11::cast<int> (seq.attr("get_prompt_len")()) - pybind11::cast<int> (seq.attr("get_num_prompt_tokens_stage_processed")()),
        chunk_size - num_batched_tokens
    );
    return next_num_tokens;
}

SchedulerOutputs SarathiScheduler::schedule()
{
    auto now = std::chrono::steady_clock::now();

    std::vector<pybind11::object> temp_running;
    std::vector<pybind11::str> temp_ignored_seq_ids;
    std::vector<pybind11::str> temp_preempted_seq_ids;
    std::vector<pybind11::object> temp_scheduled_seq_metadata_list;

    int num_batched_tokens = 0;

    temp_running = this->policy.attr("sort_by_priority")(pybind11::cast(running));

    std::vector<pybind11::object> running_prefills;

    while (this->running.size() > 0) {

    }

    return sarathi::SchedulerOutputs(
        _iteration_id,
        ignored_seq_ids,
        preempted_seq_ids,
        scheduled_seq_metadata_list
    );
}