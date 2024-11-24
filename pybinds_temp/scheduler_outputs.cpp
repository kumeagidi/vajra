#include "scheduler_outputs.h"
#include <unordered_set>
#include <queue>
#include <vector>

using namespace Sarathi;

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
        scheduledSeqMetadataList.begin(), scheduledSeqMetadataList.end(), [](const py::object& a, const py::object& b) {
            return a.attr("is_prompt").cast<bool>() > b.attr("is_prompt").cast<bool>();
        }
    );
    this->scheduled_seq_metadata_list = scheduled_seq_metadata_list;

    for (const pybind11::object& metadata : scheduled_seq_metadata_list) {
        prompt_chunk_lens.push_back(metadata.attr("num_prompt_tokens"));
        num_batched_prompt_tokens += metadata.attr("num_prompt_tokens");
        num_batched_output_tokens += metadata.attr("num_output_tokens");
        num_batched_tokens += metadata.attr("num_tokens")
    }

    num_batched_prompt_tokens = std::accumulate(prompt_chunk_lens.begin(), prompt_chunk_lens.end(), 0);

}

bool SchedulerOutputs::is_empty() {
    return scheduled_seq_metadata_list.empty();
}

bool SchedulerOutputs::has_no_output() {
    return scheduled_seq_metadata_list.empty() && ignored_seq_ids.empty() && preempted_seq_ids.empty();
}

std::vector<py::str> SchedulerOutputs::seq_ids() {
    std::vector<std::string> ids;
    for (const pybind11::object& metadata : scheduled_seq_metadata_list) {
        ids.push_back(metadata.attr("seq_id"));
    }
    return ids;
}

//py::str SchedulerOutputs::__repr__() {
    
//}