PYBIND11_MODULE(datatypes, m) {
    //Create class named SchedulerOutputs
    pybind11::module SchedulerOutputs = m.def_submodule("SchedulerOutputs", "Outputs for scheduler");

    //Create pybind from SchedulerOutputs to class that can be refered to in Python with name "SchedulerOutputs"
    pybind11::class_<sarathi::SchedulerOutputs, std::shared_ptr<sarathi::SchedulerOutputs>>(SchedulerOutputs, "SchedulerOutputs")
        .def(pybind11::init<
                        int,
                        std::vector<pybind11::str>,
                        std::vector<pybind11::str>
                        std::vector<pybind11::object>
                        >())

        // Create class attributes
        .def_readwrite("id", &sarathi::SchedulerOutputs::id);
        .def_readwrite("ignored_seq_ids", &sarathi::SchedulerOutputs::ignored_seq_ids);
        .def_readwrite("preempted_seq_ids", &sarathi::SchedulerOutputs::preempted_seq_ids);
        .def_readwrite("scheduled_seq_metadata_list", &sarathi::SchedulerOutputs::scheduled_seq_metadata_list);
        .def_readwrite("prompt_chunk_lens", &sarathi::SchedulerOutputs::prompt_chunk_lens);
        .def_readwrite("num_batched_prompt_tokens", &sarathi::SchedulerOutputs::num_batched_prompt_tokens);
        .def_readwrite("num_batched_output_tokens", &sarathi::SchedulerOutputs::num_batched_output_tokens);
        .def_readwrite("num_batched_tokens", &sarathi::SchedulerOutputs::num_batched_tokens);

        // Create class methods
        .def("is_empty", &sarathi::SchedulerOutputs::is_empty);
        .def("has_no_output", &sarathi::SchedulerOutputs::has_no_output);
        .def("seq_ids", &sarathi::SchedulerOutputs::seq_ids);
        .def("__repr__", &sarathi::SchedulerOutputs::__repr__);
