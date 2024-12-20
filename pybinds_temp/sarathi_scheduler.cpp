#include "base_scheduler.h"
#include "sarathi_scheduler.h"
#include "scheduler_outputs.h"
#include "sequence_with_priority.h"
#include <cmath>
#include <vector>
#include <queue>
#include <algorithm>
#include <memory>
#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
// mamba install -c conda-forge xtensor
pybind11::module sarathi_block_space_manager = pybind11::module::import("sarathi.core.block_space_manager.sarathi_block_space_manager");
pybind11::module sequence_schedule_metadata = pybind11::module::import("sarathi.core.datatypes.sequence");


using namespace sarathi;

SarathiScheduler::SarathiScheduler(
    pybind11::object model_config,
    pybind11::object scheduler_config,
    pybind11::object cache_config,
    pybind11::object parallel_config,
    pybind11::object replica_seq_manager,
    pybind11::object metrics_store
) :
    BaseScheduler(model_config, scheduler_config, cache_config, parallel_config, replica_seq_manager, metrics_store),
    chunk_size(pybind11::cast<int> (scheduler_config.attr("chunk_size"))),
    enable_dynamic_chunking_schedule(pybind11::cast<bool> (scheduler_config.attr("enable_dynamic_chunking_schedule"))),
    low_chunk_size(pybind11::cast<int> (scheduler_config.attr("low_chunk_size"))),
    high_chunk_size(pybind11::cast<int> (scheduler_config.attr("high_chunk_size"))),
    chunk_schedule_max_tokens(pybind11::cast<int> (scheduler_config.attr("chunk_schedule_max_tokens"))),
    chunk_schedule_stages(pybind11::cast<int> (scheduler_config.attr("chunk_schedule_stages")))
{
    if (enable_dynamic_chunking_schedule) {
        _chunk_sizes = _compute_chunk_size_schedule();
        _tokens_per_stage = std::ceil(chunk_schedule_max_tokens / static_cast<float>(chunk_schedule_stages));
    }
}

std::vector<int> SarathiScheduler::_compute_chunk_size_schedule()
{
    xt::xarray<int> chunk_sizes = xt::linspace<int>(high_chunk_size, low_chunk_size, chunk_schedule_stages);
    chunk_sizes = xt::flip(chunk_sizes, 0);

    int round_of_chunk_sizes = std::min(32, low_chunk_size);
    chunk_sizes = xt::round(chunk_sizes / round_of_chunk_sizes) * round_of_chunk_sizes;
    
    std::vector<int> chunk_sizes_vector(chunk_sizes.shape()[0]);
    std::copy(chunk_sizes.begin(), chunk_sizes.end(), chunk_sizes_vector.begin());
    return chunk_sizes_vector;
}

pybind11::object SarathiScheduler::get_block_space_manager_class()
{
    return sarathi_block_space_manager.attr("SarathiBlockSpaceManager")();
}

int SarathiScheduler::_get_seq_next_num_prefill_tokens(pybind11::object& seq, int num_batched_tokens)
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

SchedulerOutputs SarathiScheduler::_schedule()
{
    std::cout << "schedule called 1" << std::endl;
    pybind11::object now = pybind11::module_::import("time").attr("monotonic")();

    std::deque<pybind11::object> temp_running;
    std::vector<pybind11::str> temp_ignored_seq_ids;
    std::vector<pybind11::str> temp_preempted_seq_ids;
    std::vector<pybind11::object> temp_scheduled_seq_metadata_list;
    pybind11::object seq;
    int next_num_prefill_tokens;

    int num_batched_tokens = 0;
    std::cout << "schedule called 1.5" << std::endl;
    std::sort(running.begin(), running.end(), [](const pybind11::object& a, const pybind11::object& b) {
        return a.attr("arrival_time").cast<float>() < b.attr("arrival_time").cast<float>();
    });
    std::cout << "schedule called 2" << std::endl;

    std::vector<pybind11::object> running_prefills;

    while (!running.empty()) {
        std::cout << "schedule called 3" << std::endl;
        seq = running.front();
        running.pop_front();
        std::cout << "schedule called 3.1" << std::endl;
        if (pybind11::cast<bool> (seq.attr("is_paused")()) == false) {
            temp_running.push_back(seq);
            continue;
        }
        std::cout << "schedule called 3.2" << std::endl;
        if (pybind11::cast<bool> (seq.attr("prompt_stage_processing_finished")) == false) {
            running_prefills.push_back(seq);
            continue;
        }
        std::cout << "schedule called 3.3" << std::endl;
        bool loopCompletedNormally = true;
        while (pybind11::cast<bool> (block_manager.attr("can_append_slot")()) == false) {
            if (!running.empty()) {
                pybind11::object victim_seq = running.back();
                running.pop_back();
                _preempt(victim_seq);
                temp_preempted_seq_ids.push_back(pybind11::cast<std::string> (seq.attr("seq_id")));
            } else {
                _preempt(seq);
                temp_preempted_seq_ids.push_back(pybind11::cast<std::string> (seq.attr("seq_id")));
                loopCompletedNormally = false;
                break;
            }
        }
        std::cout << "schedule called 3.4" << std::endl;
        if (loopCompletedNormally) {
            _append_slot(seq);
            temp_running.push_back(seq);
            num_batched_tokens++;   

            std::cout << "push_back 1" << std::endl;
            temp_scheduled_seq_metadata_list.push_back(
                sequence_schedule_metadata.attr("SequenceScheduleMetadata").attr("from_sequence")(seq)
            );
        }
    }
    std::cout << "schedule called 4" << std::endl;
    for (pybind11::object& seq: running_prefills) {
        std::cout << "enter for loop" << std::endl;
        next_num_prefill_tokens =  _get_seq_next_num_prefill_tokens(
            seq, num_batched_tokens
        );

        if (next_num_prefill_tokens == 0) {
            temp_running.push_back(seq);
            continue;
        }

        num_batched_tokens += next_num_prefill_tokens;
        std::cout << "push_back 2" << std::endl;
        temp_scheduled_seq_metadata_list.push_back(
            sequence_schedule_metadata.attr("SequenceScheduleMetadata").attr("from_sequence")(seq, pybind11::int_(next_num_prefill_tokens))
        );
        temp_running.push_back(seq);
    }

    while (!waiting.empty()) {
        sarathi::SequenceWithPriority seq_wrapped = waiting.top();
        seq = seq_wrapped.seq;
        std::cout << "schedule called 5" << std::endl;
        if (seq.attr("arrival_time").cast<float>() > now.cast<float>()) {
            break;
        }

        if (!_check_request_prompt_length(seq)) {
            temp_ignored_seq_ids.push_back(seq.attr("seq_id"));
            continue;
        }

        if (pybind11::cast<bool> (block_manager.attr("can_allocate")(seq)) == false) {
            break;
        }

        if (temp_running.size() >= pybind11::cast<int> (scheduler_config.attr("max_num_seqs"))) {
            break;
        }

        next_num_prefill_tokens = _get_seq_next_num_prefill_tokens(
            seq, num_batched_tokens
        );

        if (next_num_prefill_tokens == 0) {
            break;
        }

        seq_wrapped = waiting.top();
        waiting.pop();
        seq = seq_wrapped.seq;
        _allocate(seq);
        num_batched_tokens += next_num_prefill_tokens;
        temp_scheduled_seq_metadata_list.push_back(
            sequence_schedule_metadata.attr("SequenceScheduleMetadata").attr("from_sequence")(seq, pybind11::int_(next_num_prefill_tokens))
            
        );

        std::cout << "push_back 3.1" << std::endl;

        // std::string seq_id = pybind11::cast<std::string> (seq.attr("seq_id"));

        // std::cout << "push_back 3.2" << std::endl;
        // if (seq_seen.find(seq_id) == seq_seen.end()) {
        //     std::cout << "push_back 3.3" << std::endl;
        //     add_seq_to_seq_manager(seq);
        //     std::cout << "push_back 3.4" << std::endl;
        //     add_to_new_seqs(seq);  // deepcopy not needed for pybind11 objects
        //     std::cout << "push_back 3.5" << std::endl;
        //     seq_seen.insert(seq_id);
        //     std::cout << "push_back 3.6" << std::endl;
        // }

        metrics_store.attr("on_request_arrival")(seq);
        temp_running.push_back(seq);
        std::cout << "schedule called 6" << std::endl;
    }
    
    running = temp_running;
    std::cout << "schedule called 7" << std::endl;
    return sarathi::SchedulerOutputs(
        BaseScheduler::_iteration_id,
        temp_ignored_seq_ids,
        temp_preempted_seq_ids,
        temp_scheduled_seq_metadata_list
    );
}