#include "base_scheduler.h"

PYBIND11_MODULE(Scheduler, m) {
    //Create class named BaseScheduler
    pybind11::module BaseScheduler = m.def_submodule("BaseScheduler", "Base scheduler for all models");
bnlkjbkjnm,
    //Create pybind from BaseScheduler to class that can be refered to in Python with name "BaseScheduler"
    pybind11::class_<sarathi::BaseScheduler, std::shared_ptr<sarathi::BaseScheduler>>(BaseScheduler, "BaseScheduler")
        .def(pybind11::init<
                        pybind11::object,  // ModelConfig
                        pybind11::object,  // BaseSchedulerConfig
                        pybind11::object,  // CacheConfig
                        pybind11::object,  // ParallelConfig
                        pybind11::object,  // PriorityQueue
                        pybind11::object,  // EngineSequenceManager
                        pybind11::object  // MetricsStore
                        >())

        // Create class attributes
        .def_readwrite("metrics_store", &sarathi::BaseScheduler::metrics_store);
        .def_readwrite("model_config", &sarathi::BaseScheduler::model_config);
        .def_readwrite("scheduler_config", &sarathi::BaseScheduler::scheduler_config);
        .def_readwrite("cache_config", &sarathi::BaseScheduler::cache_config);
        .def_readwrite("parallel_config", &sarathi::BaseScheduler::parallel_config);
        .def_readwrite("_iteration_id", &sarathi::BaseScheduler::_iteration_id);
        .def_readwrite("policy", &sarathi::BaseScheduler::policy);
        .def_readwrite("block_manager", &sarathi::BaseScheduler::block_manager);
        .def_readwrite("prompt_limit", &sarathi::BaseScheduler::prompt_limit);
        .def_readwrite("replica_seq_manager", &sarathi::BaseScheduler::replica_seq_manager);
        .def_readwrite("new_seqs", &sarathi::BaseScheduler::new_seqs);
        .def_readwrite("metrics_store", &sarathi::BaseScheduler::metrics_store);
        .def_readwrite("seq_seen", &sarathi::BaseScheduler::seq_seen);
        .def_readwrite("num_running_batches", &sarathi::BaseScheduler::num_running_batches);
        .def_readwrite("waiting", &sarathi::BaseScheduler::waiting);
        .def_readwrite("running", &sarathi::BaseScheduler::running);

        // Create class methods
        .def("reset_state", &sarathi::BaseScheduler::reset_state);
        .def("add_seq", &sarathi::BaseScheduler::add_seq);
        .def("has_unfinished_seqs", &sarathi::BaseScheduler::has_unfinished_seqs);
        .def("get_num_unfinished_seqs", &sarathi::BaseScheduler::get_num_unfinished_seqs);
        .def("_schedule", &sarathi::BaseScheduler::_schedule); // abstract
        .def("add_to_new_seqs", &sarathi::BaseScheduler::add_to_new_seqs); // synchronized
        .def("get_new_seqs", &sarathi::BaseScheduler::get_new_seqs); // synchronized
        .def("add_seq_to_seq_manager", &sarathi::BaseScheduler::add_seq_to_seq_manager); // synchronized
        .def("schedule", &sarathi::BaseScheduler::schedule);
        .def("free_finished_seqs", &sarathi::BaseScheduler::free_finished_seqs);
        .def("free_finished_seqs", &sarathi::BaseScheduler::free_finished_seqs);
        .def("_allocate", &sarathi::BaseScheduler::_allocate);
        .def("_free_seq", &sarathi::BaseScheduler::_free_seq);
        .def("_append_slot", &sarathi::BaseScheduler::_append_slot);
        .def("_preempt", &sarathi::BaseScheduler::_preempt);
        .def("_check_request_prompt_length", &sarathi::BaseScheduler::_check_request_prompt_length);
}