#ifndef SARATHI_SCHEDULER_H
#define SARATHI_SCHEDULER_H
#include <pybind11/pybind11.h>
#include "scheduler_outputs.h"
#include "base_scheduler.h"
#include "sequence_with_priority.h"
#include <queue>
//#include "csrc/commons/Logging.h"

namespace sarathi
{
class SarathiScheduler : BaseScheduler
{
    public:
        SarathiScheduler(
            pybind11::object model_config,
            pybind11::object scheduler_config,
            pybind11::object cache_config,
            pybind11::object parallel_config,
            pybind11::object waiting_queue,
            pybind11::object replica_seq_manager,
            pybind11::object metric_store
        );

        //class methods here
        int _compute_chunk_size_schedule();
        pybind11::object get_block_space_manager_class();
        int _get_seq_next_num_prefill_tokens(pybind11::object seq, int num_batched_tokens);
        SchedulerOutputs _schedule() override;

        int chunk_size;
        bool enable_dynamic_chunking_schedule;
        int low_chunk_size;
        int high_chunk_size;
        int chunk_schedule_max_tokens;
        int chunk_schedule_stages;
        std::vector<int> ._chunk_sizes;
        int _tokens_per_stage;
};

//==============================================================================
} // namespace sarathi
//==============================================================================

#endif