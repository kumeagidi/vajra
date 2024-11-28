#ifndef BASE_SCHEDULER_H
#define BASE_SCHEDULER_H
#include <pybind11/pybind11.h>
#include "scheduler_outputs.h"
#include "sequence_with_priority.h"
#include <queue>
//#include "csrc/commons/Logging.h"

namespace sarathi
{
class BaseScheduler
{
    public:
        BaseScheduler(
            pybind11::object model_config,
            pybind11::object scheduler_config,
            pybind11::object cache_config,
            pybind11::object parallel_config,
            pybind11::object waiting_queue,
            pybind11::object replica_seq_manager,
            pybind11::object metric_store
        );

        //class methods here
        void reset_state();
        void add_seq(pybind11::object& seq);
        bool has_unfinished_seqs();
        int get_num_unfinished_seqs();
        virtual sarathi::SchedulerOutputs _schedule();
        void add_to_new_seqs(pybind11::object& seq);
        std::vector<pybind11::object> get_new_seqs();
        void add_seq_to_seq_manager(pybind11::object& seq);
        sarathi::SchedulerOutputs schedule();
        void free_finished_seqs();
        void on_step_completed();
        void _allocate(pybind11::object& seq);
        void _free_seq(pybind11::object& seq);
        void _append_slot(pybind11::object& seq);
        void _preempt(pybind11::object& seq);
        bool _check_request_prompt_length(pybind11::object& seq);

        pybind11::object policy;
        pybind11::object block_manager;
        int prompt_limit;
        pybind11::object max_model_len;

        pybind11::object metric_store;
        pybind11::object model_config;
        pybind11::object scheduler_config;
        pybind11::object cache_config;
        pybind11::object parallel_config;
        int _iteration_id;
        pybind11::object replica_seq_manager;
        std::unordered_set<int> seq_seen;
        std::priority_queue<sarathi::SequenceWithPriority, std::vector<sarathi::SequenceWithPriority>, std::greater<sarathi::SequenceWithPriority>> waiting;
        int num_running_batches; 
        std::vector<pybind11::object> new_seqs;
        std::vector<pybind11::object> running;
};

//==============================================================================
} // namespace sarathi
//==============================================================================

#endif