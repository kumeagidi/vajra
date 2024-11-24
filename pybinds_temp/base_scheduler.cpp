#include "base_scheduler.h"
#include <unordered_set>
#include <queue>
#include <vector>

using namespace Sarathi;

BaseScheduler::BaseScheduler(
    pybind11::object& model_config,
    pybind11::object& scheduler_config,
    pybind11::object& cache_config,
    pybind11::object& parallel_config,
    pybind11::object& waiting_queue,
    pybind11::object& replica_seq_manager,
    pybind11::object& metric_store,

) :
    metric_store(metric_store),
    model_config(model_config),
    scheduler_config(scheduler_config),
    cache_config(cache_config),
    parallel_config(parallel_config),
    _iteration_id(-1),
    replica_seq_manager(replica_seq_manager),
    metric_store(metric_store),
    seq_seen(unordered_set<int>),
    waiting(pybind11::object),
    num_running_batches(0), 
    new_seqs(std::vector<pybind11::object>()),
    running(std::vector<pybind11::object>()),
{
    pybind11::module_ policy_module = pybind11::module_::import("policy");
    pybind11::object PolicyFactory = policy_module.attr("PolicyFactory");
    this->policy = PolicyFactory.attr("get_policy")("fcfs");

    pybind11::module_ block_manager_module = pybind11::module_::import("block_space_manager_registry");
    pybind11::object BlockSpaceManagerRegistry = policy_module.attr("BlockSpaceManagerRegistry");
    this->block_manager = BlockSpaceManagerRegistry.attr("get")(
        scheduler_config.attr("get_type")(),
        cache_config.attr("block_size"),
        cache_config.attr("num_gpu_blocks"),
        model_config.attr("max_model_len")
    );

    this->prompt_limit = model_config.attr("max_model_len");  
}

void BaseScheduler::reset_state()
{
    _iteration_id = -1;
}

void BaseScheduler::add_seq(pybind11::object& seq)
{
    pybind11::object wrapped_seq = seq.attr("create_sequence_with_priority")(seq);
    waiting.attr("put")(wrapped_seq)
}

void BaseScheduler::has_unfinished_seqs()
{
    return waiting.attr("qsize")() > 0 || !running.empty()
}

void BaseScheduler::get_num_unfinished_seqs()
{
    return self.waiting.qsize() + running.size()
}

virtual BaseScheduler::pybind11::object& _schedule()
{
}

void BaseScheduler::add_to_new_seqs(pybind11::object& seq)
{
    new_seqs.push_back(seq)
}

std::vector<py::object> BaseScheduler::get_new_seqs()
{
    std::vector<pybind11::object> seqs = new_seqs;
    new_seqs.clear();
    return seqs;
}

void BaseScheduler::add_seq_to_seq_manager(const pybind11::object& seq)
{
    replica_seq_manager.attr("add_seq")(seq);
}

pybind11::object& BaseScheduler::schedule()
{
    _iteration_id++;
    std::vector<int> ignored_seq_ids = {};
    std::vector<int> preempted_seq_ids = {};
    std::vector<int> scheduled_seq_metadata_list = {};

    if (num_running_batches > parallel_config.attr("pipeline_parallel_size")) {
        return SchedulerOutputs(
            _iteration_id,
            ignored_seq_ids,
            preempted_seq_ids,
            scheduled_seq_metadata_list
        );
    }

    pybind11::object& scheduler_outputs = _schedule()

    if (!scheduler_outputs.attr("is_empty")()) {
        num_running_batches++;
    }
    return scheduler_outputs

}

void free_finished_seqs()
{
    for (pybind11::object& seq:running) {
        if (seq.attr("is_finished")()) {
            seq.attr("_free_seq")();
        }
    }
    std::vector<pybind11::object>() new_running;
    for (pybind11::object& seq : running) {
        if (!seq.attr("is_finished")()) {
            new_running.push_back(seq);
        }
    } 
    running = new_running;

void on_step_completed()
{
    free_finished_seqs();
    num_running_batches--;
}

void _allocate(pybind11::object& seq)
{
    block_manager.attr("allocate")(seq);
}

void _free_seq(pybind11::object& seq)
{
    block_manager.attr("free")(seq);
}

void _append_slot(pybind11::object& seq)
{
    ASSERT(seq.attr("is_executing")());
    block_manager.attr("append_slot")(seq);
}

void _preempt(pybind11::object& seq)
{
    ASSERT(seq.attr("is_executing")());
    _free_seq(seq);

}

void _check_request_prompt_length(pybind11::object& seq)
{
    if (seq.attr("get_len")() > prompt_limit) {
        seq.attr("set_status")//);

        return False;
    }
    return True;
}