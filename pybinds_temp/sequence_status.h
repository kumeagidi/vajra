#ifndef SEQUENCE_STATUS_H
#define SEQUENCE_STATUS_H
#include <pybind11/pybind11.h>

namespace sarathi
{
class SequenceStatus 
{
    public:
        enum Status {
            WAITING,
            WAITING_PREEMPTED,
            RUNNING,
            PAUSED,
            FINISHED_STOPPED,
            FINISHED_LENGTH_CAPPED,
            FINISHED_IGNORED
        };

        static bool is_finished(Status status);
        static bool is_executing(Status status);
        static bool is_waiting(Status status);
        static bool is_waiting_preempted(Status status);
        static bool is_paused(Status status);
        static bool is_running(Status status);
        static pybind11::str get_finished_reason(Status status);
};
}
#endif