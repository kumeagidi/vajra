#include "base_scheduler.h"
#include "scheduler_outputs.h"
#include "sequence_with_priority.h"
#include "sequence_status.h"
#include <unordered_set>
#include <queue>
#include <vector>
#include <iostream>

using namespace sarathi;

BaseScheduler::BaseScheduler(
    pybind11::object model_config,
    pybind11::object scheduler_config,
    pybind11::object cache_config,
    pybind11::object parallel_config,
    pybind11::object replica_seq_manager,
    pybind11::object metrics_store
) :
    metrics_store(metrics_store),
    model_config(model_config),
    scheduler_config(scheduler_config),
    cache_config(cache_config),
    parallel_config(parallel_config),
    _iteration_id(-1),
    replica_seq_manager(replica_seq_manager),
    prompt_limit(pybind11::cast<int> (model_config.attr("max_model_len"))),
    seq_seen(std::unordered_set<int>()),
    waiting(),
    num_running_batches(0), 
    new_seqs(std::vector<pybind11::object>()),
    running(std::deque<pybind11::object>())
{
    pybind11::module_ policy_module = pybind11::module_::import("sarathi.core.policy");
    pybind11::object PolicyFactory = policy_module.attr("PolicyFactory");
    this->policy = PolicyFactory.attr("get_policy")("fcfs");

    pybind11::module_ block_manager_module = pybind11::module_::import("sarathi.core.block_space_manager.block_space_manager_registry");
    pybind11::object BlockSpaceManagerRegistry = block_manager_module.attr("BlockSpaceManagerRegistry");
    this->block_manager = BlockSpaceManagerRegistry.attr("get")(
        scheduler_config.attr("get_type")(),
        cache_config.attr("block_size"),
        cache_config.attr("num_gpu_blocks"),
        model_config.attr("max_model_len")
    );
}

void BaseScheduler::reset_state()
{
    _iteration_id = -1;
}

void BaseScheduler::add_seq(pybind11::object& seq)
{
    float arrived_at = seq.attr("arrived_at").cast<float>();
    SequenceWithPriority seq_with_priority = SequenceWithPriority(arrived_at, seq);
    waiting.push(seq_with_priority);
}

bool BaseScheduler::has_unfinished_seqs()
{
    return waiting.size() > 0 || !running.empty();
}

int BaseScheduler::get_num_unfinished_seqs()
{
    return waiting.size() + running.size();
}

void BaseScheduler::add_to_new_seqs(pybind11::object& seq)
{
    new_seqs.push_back(seq);
}

std::vector<pybind11::object> BaseScheduler::get_new_seqs()
{
    std::vector<pybind11::object> seqs = new_seqs;
    new_seqs.clear();
    return seqs;
}

void BaseScheduler::add_seq_to_seq_manager(pybind11::object& seq)
{
    replica_seq_manager.attr("add_seq")(seq);
}

sarathi::SchedulerOutputs BaseScheduler::schedule()
{
    _iteration_id++;
    std::vector<pybind11::str> ignored_seq_ids = {};
    std::vector<pybind11::str> preempted_seq_ids = {};
    std::vector<pybind11::object> scheduled_seq_metadata_list = {};
    std::cout << " regular schedule 1" << std::endl;

    if (num_running_batches >= pybind11::cast<int> (parallel_config.attr("pipeline_parallel_size"))) {
        std::cout << " regular schedule 2" << std::endl;
        return sarathi::SchedulerOutputs(
            _iteration_id,
            ignored_seq_ids,
            preempted_seq_ids,
            scheduled_seq_metadata_list
        );
    }
    std::cout << " regular schedule 3" << std::endl;
    sarathi::SchedulerOutputs scheduler_outputs = _schedule();

     if (!scheduler_outputs.is_empty()) {
         num_running_batches++;
     }
     std::cout << " regular schedule 4" << std::endl;
     return scheduler_outputs;
}

void BaseScheduler::free_finished_seqs()
{
    for (pybind11::object& seq: running) {
        if (pybind11::cast<bool> (seq.attr("is_finished")())) {
            std::cout << "FREE SEQUENCE CALLED FROM C++ " << pybind11::cast<bool> (seq.attr("is_finished")()) << std::endl;
            _free_seq(seq);
        }
    }
    std::deque<pybind11::object> new_running;
    for (pybind11::object& seq : running) {
        if (pybind11::cast<bool> (seq.attr("is_finished")()) == false) {
            new_running.push_back(seq);
        }
    } 
    running = new_running;
}

void BaseScheduler::on_step_completed()
{
    free_finished_seqs();
    num_running_batches--;
}

void BaseScheduler::_allocate(pybind11::object& seq)
{
    block_manager.attr("allocate")(seq);
}

void BaseScheduler::_free_seq(pybind11::object& seq)
{
    block_manager.attr("free")(seq);
}

void BaseScheduler::_append_slot(pybind11::object& seq)
{
    //ASSERT(pybind11::cast<bool> (seq.attr("is_executing")()));
    block_manager.attr("append_slot")(seq);
}

void BaseScheduler::_preempt(pybind11::object& seq)
{
    //ASSERT(pybind11::cast<int> (seq.attr("is_executing")()));
    std::cout << "_preempt: trying to preempt" << std::endl;
    _free_seq(seq);
    std::cout << "_preempt: free seq completed " << std::endl;
    float arrived_at = seq.attr("arrived_at").cast<float>();
    sarathi::SequenceWithPriority wrapped_seq = SequenceWithPriority(arrived_at, seq);
    waiting.push(wrapped_seq);
}

bool BaseScheduler::_check_request_prompt_length(pybind11::object& seq)
{
    if (pybind11::cast<int> (seq.attr("get_len")()) > prompt_limit) {
        std::cout << "Too big needs finished ignored" << std::endl;
        pybind11::module_ sequence_status = pybind11::module_::import("sarathi.core.datatypes.sequence_status");
        seq.attr("set_status")(sequence_status.attr("FINISHED_IGNORED"));
        if (!waiting.empty()) {
            waiting.pop();
        }
        return false;
    }
    return true;
}