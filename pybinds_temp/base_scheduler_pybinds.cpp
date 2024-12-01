#include "base_scheduler.h"
#include "sarathi_scheduler.h"
#include "scheduler_outputs.h"
#define PYBIND11_DETAILED_ERROR_MESSAGES

PYBIND11_MODULE(_base_scheduler_C, m) {
    //Create class named BaseScheduler
    pybind11::module BaseScheduler = m.def_submodule("BaseScheduler", "Base scheduler for all models");

    //Create pybind from BaseScheduler to class that can be refered to in Python with name "BaseScheduler"
    pybind11::class_<sarathi::BaseScheduler>(BaseScheduler, "BaseScheduler")
            .def_readwrite("metrics_store", &sarathi::BaseScheduler::metrics_store)
            .def_readwrite("model_config", &sarathi::BaseScheduler::model_config)
            .def_readwrite("scheduler_config", &sarathi::BaseScheduler::scheduler_config)
            .def_readwrite("cache_config", &sarathi::BaseScheduler::cache_config)
            .def_readwrite("parallel_config", &sarathi::BaseScheduler::parallel_config)
            .def_readwrite("_iteration_id", &sarathi::BaseScheduler::_iteration_id)
            .def_readwrite("policy", &sarathi::BaseScheduler::policy)
            .def_readwrite("block_manager", &sarathi::BaseScheduler::block_manager)
            .def_readwrite("replica_seq_manager", &sarathi::BaseScheduler::replica_seq_manager)
            .def_readwrite("new_seqs", &sarathi::BaseScheduler::new_seqs)
            .def_readwrite("seq_seen", &sarathi::BaseScheduler::seq_seen)
            .def_readwrite("num_running_batches", &sarathi::BaseScheduler::num_running_batches)
            .def_readwrite("waiting", &sarathi::BaseScheduler::waiting)
            .def_readwrite("running", &sarathi::BaseScheduler::running)
            .def("reset_state", &sarathi::BaseScheduler::reset_state)
            .def("add_seq", &sarathi::BaseScheduler::add_seq)
            .def("has_unfinished_seqs", &sarathi::BaseScheduler::has_unfinished_seqs)
            .def("get_num_unfinished_seqs", &sarathi::BaseScheduler::get_num_unfinished_seqs)
            .def("add_to_new_seqs", &sarathi::BaseScheduler::add_to_new_seqs)
            .def("on_step_completed", &sarathi::BaseScheduler::on_step_completed)
            .def("get_new_seqs", &sarathi::BaseScheduler::get_new_seqs)
            .def("add_seq_to_seq_manager", &sarathi::BaseScheduler::add_seq_to_seq_manager)
            .def("schedule", &sarathi::BaseScheduler::schedule)
            .def("free_finished_seqs", &sarathi::BaseScheduler::free_finished_seqs)
            .def("_allocate", &sarathi::BaseScheduler::_allocate)
            .def("_free_seq", &sarathi::BaseScheduler::_free_seq)
            .def("_append_slot", &sarathi::BaseScheduler::_append_slot)
            .def("_preempt", &sarathi::BaseScheduler::_preempt)
            .def("_check_request_prompt_length", &sarathi::BaseScheduler::_check_request_prompt_length);
        
    pybind11::class_<sarathi::SarathiScheduler, sarathi::BaseScheduler>(BaseScheduler, "SarathiScheduler")
        .def(pybind11::init<
                pybind11::object,  // ModelConfig
                pybind11::object,  // SarathiSchedulerConfig
                pybind11::object,  // CacheConfig
                pybind11::object,  // ParallelConfig
                pybind11::object,  // EngineSequenceManager
                pybind11::object  // MetricStore
                >())
                .def_readwrite("chunk_size", &sarathi::SarathiScheduler::chunk_size)
                .def_readwrite("enable_dynamic_chunking_schedule", &sarathi::SarathiScheduler::enable_dynamic_chunking_schedule)
                .def_readwrite("low_chunk_size", &sarathi::SarathiScheduler::low_chunk_size)
                .def_readwrite("high_chunk_size", &sarathi::SarathiScheduler::high_chunk_size)
                .def_readwrite("chunk_schedule_max_tokens", &sarathi::SarathiScheduler::chunk_schedule_max_tokens)
                .def_readwrite("chunk_schedule_stages", &sarathi::SarathiScheduler::chunk_schedule_stages)
                .def_readwrite("_chunk_sizes", &sarathi::SarathiScheduler::_chunk_sizes)
                .def_readwrite("_tokens_per_stage", &sarathi::SarathiScheduler::_tokens_per_stage)
                .def("_compute_chunk_size_schedule", &sarathi::SarathiScheduler::_compute_chunk_size_schedule)
                .def("get_block_space_manager_class", &sarathi::SarathiScheduler::get_block_space_manager_class)
                .def("_get_seq_next_num_prefill_tokens", &sarathi::SarathiScheduler::_get_seq_next_num_prefill_tokens)
                .def("_schedule", &sarathi::SarathiScheduler::_schedule);

    
    pybind11::class_<sarathi::SequenceWithPriority>(BaseScheduler, "SequenceWithPriority")
        .def(pybind11::init<
                float,  // priority
                pybind11::object // sequence
                >())
                .def_readwrite("priority", &sarathi::SequenceWithPriority::priority)
                .def_readwrite("seq", &sarathi::SequenceWithPriority::seq)
                .def("__lt__", &sarathi::SequenceWithPriority::operator<)
                .def("__eq__", &sarathi::SequenceWithPriority::operator==)
                .def("__gt__", &sarathi::SequenceWithPriority::operator>);
                // .def("__repr__", [](const SequenceWithPriority &obj) {
                //     return "<SequenceWithPriority(priority=" + std::to_string(obj.priority) + ", seq=" + py::str(obj.seq).cast<std::string>() + ")>";
                // });
    
    pybind11::class_<sarathi::SchedulerOutputs>(BaseScheduler, "SchedulerOutputs")
        .def(pybind11::init<
                        int,
                        std::vector<pybind11::str>,
                        std::vector<pybind11::str>,
                        std::vector<pybind11::object>
                        >())
                        .def_readwrite("id", &sarathi::SchedulerOutputs::id)
                        .def_readwrite("ignored_seq_ids", &sarathi::SchedulerOutputs::ignored_seq_ids)
                        .def_readwrite("preempted_seq_ids", &sarathi::SchedulerOutputs::preempted_seq_ids)
                        .def_readwrite("scheduled_seq_metadata_list", &sarathi::SchedulerOutputs::scheduled_seq_metadata_list)
                        .def_readwrite("prompt_chunk_lens", &sarathi::SchedulerOutputs::prompt_chunk_lens)
                        .def_readwrite("num_prompt_tokens", &sarathi::SchedulerOutputs::num_prompt_tokens)
                        .def_readwrite("num_batched_prompt_tokens", &sarathi::SchedulerOutputs::num_batched_prompt_tokens)
                        .def_readwrite("num_batched_output_tokens", &sarathi::SchedulerOutputs::num_batched_output_tokens)
                        .def_readwrite("num_batched_tokens", &sarathi::SchedulerOutputs::num_batched_tokens)
                        .def("is_empty", &sarathi::SchedulerOutputs::is_empty)
                        .def("has_no_output", &sarathi::SchedulerOutputs::has_no_output)
                        .def("seq_ids", &sarathi::SchedulerOutputs::seq_ids)
                        .def(pybind11::pickle(
                            [](const sarathi::SchedulerOutputs &p) { // __getstate__
                                /* Return a tuple that fully encodes the state of the object */
                                return pybind11::make_tuple(
                                    p.id,                          // p.id() represents the id
                                    p.ignored_seq_ids,              // ignored_seq_ids
                                    p.preempted_seq_ids,            // preempted_seq_ids
                                    p.scheduled_seq_metadata_list,  // scheduled_seq_metadata_list
                                    p.num_prompt_tokens,            // num_prompt_tokens
                                    p.prompt_chunk_lens,            // prompt_chunk_lens
                                    p.num_batched_prompt_tokens,    // num_batched_prompt_tokens
                                    p.num_batched_output_tokens,    // num_batched_output_tokens
                                    p.num_batched_tokens            // num_batched_tokens
                                );
                            },
                            [](pybind11::tuple t) { // __setstate__

                                /* Create a new C++ instance */
                                sarathi::SchedulerOutputs p(
                                    t[0].cast<int>(),
                                    t[1].cast<std::vector<pybind11::str>>(),
                                    t[2].cast<std::vector<pybind11::str>>(),
                                    t[3].cast<std::vector<pybind11::object>>()
                                );

                                // Manually set derived state (attributes that were computed)
                                p.num_prompt_tokens = t[4].cast<int>();
                                p.prompt_chunk_lens = t[5].cast<std::vector<int>>();
                                p.num_batched_prompt_tokens = t[6].cast<int>();
                                p.num_batched_output_tokens = t[7].cast<int>();
                                p.num_batched_tokens = t[8].cast<int>();

                                return p;
                            }
                        ));
}