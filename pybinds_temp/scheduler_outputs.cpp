#include "scheduler_outputs.h"
#include <unordered_set>
#include <queue>
#include <vector>

using namespace sarathi;

SchedulerOutputs::SchedulerOutputs(
        int id,
        std::vector<pybind11::str> ignored_seq_ids,
        std::vector<pybind11::str> preempted_seq_ids,
        std::vector<pybind11::object> scheduled_seq_metadata_list
) :
    id(id),
    ignored_seq_ids(ignored_seq_ids),
    preempted_seq_ids(preempted_seq_ids)
{
    std::sort(
        scheduled_seq_metadata_list.begin(), scheduled_seq_metadata_list.end(), [](const pybind11::object& a, const pybind11::object& b) {
            return pybind11::cast<bool> (a.attr("is_prompt")) > pybind11::cast<bool> (b.attr("is_prompt"));
        }
    );
    this->scheduled_seq_metadata_list = scheduled_seq_metadata_list;


    for (const pybind11::object& metadata : scheduled_seq_metadata_list) {
        num_prompt_tokens = pybind11::cast<int> (metadata.attr("num_prompt_tokens"));
        prompt_chunk_lens.push_back(num_prompt_tokens);
        num_batched_prompt_tokens += num_prompt_tokens;
        num_batched_output_tokens += pybind11::cast<int> (metadata.attr("num_output_tokens"));
        num_batched_tokens += pybind11::cast<int> (metadata.attr("num_tokens"));
    }
}

bool SchedulerOutputs::is_empty() {
    return scheduled_seq_metadata_list.empty();
}

bool SchedulerOutputs::has_no_output() {
    return scheduled_seq_metadata_list.empty() && ignored_seq_ids.empty() && preempted_seq_ids.empty();
}

std::vector<pybind11::str> SchedulerOutputs::seq_ids() {
    std::vector<pybind11::str> ids;
    for (const pybind11::object& metadata : scheduled_seq_metadata_list) {
        ids.push_back(metadata.attr("seq_id"));
    }
    return ids;
}

//py::str SchedulerOutputs::__repr__() {
    // We can deal w this later 
//    return "";
//}