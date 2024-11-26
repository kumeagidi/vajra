#include "sequence_status.h"

using namespace sarathi;

bool SequenceStatus::is_finished(Status status) {
    return status == FINISHED_STOPPED ||
           status == FINISHED_LENGTH_CAPPED ||
           status == FINISHED_IGNORED;
}

bool SequenceStatus::is_executing(Status status) {
    return status == RUNNING || status == PAUSED;
}

bool SequenceStatus::is_waiting(Status status) {
    return status == WAITING;
}

bool SequenceStatus::is_waiting_preempted(Status status) {
    return status == WAITING_PREEMPTED;
}

bool SequenceStatus::is_paused(Status status) {
    return status == PAUSED;
}

bool SequenceStatus::is_running(Status status) {
    return status == RUNNING;
}

pybind11::str SequenceStatus::get_finished_reason(Status status) {
    if (status == FINISHED_STOPPED) {
        return pybind11::str("stop");
    } else if (status == FINISHED_LENGTH_CAPPED || 
               status == FINISHED_IGNORED) {
        return pybind11::str("length");
    }
    return pybind11::str("");
}