#include <pybind11/pybind11.h>
#include "sequence_status.h"

PYBIND11_MODULE(sequence_status, m) {
    pybind11::module SequenceStatus = m.def_submodule("SequenceStatus", "Sequence status enum");

    pybind11::class_<sarathi::SequenceStatus, std::shared_ptr<sarathi::SequenceStatus>>(SequenceStatus, "SequenceStatus")
        .def_static("is_finished", &sarathi::SequenceStatus::is_finished)
        .def_static("is_executing", &sarathi::SequenceStatus::is_executing)
        .def_static("is_waiting", &sarathi::SequenceStatus::is_waiting)
        .def_static("is_waiting_preempted", &sarathi::SequenceStatus::is_waiting_preempted)
        .def_static("is_paused", &sarathi::SequenceStatus::is_paused)
        .def_static("is_running", &sarathi::SequenceStatus::is_running)
        .def_static("get_finished_reason", &sarathi::SequenceStatus::get_finished_reason);

    pybind11::enum_<sarathi::SequenceStatus::Status>(SequenceStatus, "Status")
        .value("WAITING", sarathi::SequenceStatus::Status::WAITING)
        .value("WAITING_PREEMPTED", sarathi::SequenceStatus::Status::WAITING_PREEMPTED)
        .value("RUNNING", sarathi::SequenceStatus::Status::RUNNING)
        .value("PAUSED", sarathi::SequenceStatus::Status::PAUSED)
        .value("FINISHED_STOPPED", sarathi::SequenceStatus::Status::FINISHED_STOPPED)
        .value("FINISHED_LENGTH_CAPPED", sarathi::SequenceStatus::Status::FINISHED_LENGTH_CAPPED)
        .value("FINISHED_IGNORED", sarathi::SequenceStatus::Status::FINISHED_IGNORED)
        .export_values();
}