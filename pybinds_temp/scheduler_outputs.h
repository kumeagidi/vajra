#ifndef SCHEDULER_OUTPUTS_H
#define SCHEDULER_OUTPUTS_H

#include <pybind11/pybind11.h>

namespace sarathi
{
class SchedulerOutputs
{
    public:
        SchedulerOutputs(
            int id,
            std::vector<pybind11::str> ignored_seq_ids,
            std::vector<pybind11::str> preempted_seq_ids,
            std::vector<pybind11::object> scheduled_seq_metadata_list
        );

        //class methods here
        bool is_empty();
        bool has_no_output();
        std::vector<pybind11::str> seq_ids();
        //py::str __repr__();


        int num_prompt_tokens;
        std::vector<int> prompt_chunk_lens;
        int num_batched_prompt_tokens;
        int num_batched_output_tokens;
        int num_batched_tokens;
        int id;
        std::vector<pybind11::str> ignored_seq_ids;
        std::vector<pybind11::str> preempted_seq_ids;
        std::vector<pybind11::object> scheduled_seq_metadata_list;
};


//==============================================================================
} // namespace sarathi
//==============================================================================

#endif