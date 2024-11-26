#include <pybind11/pybind11.h>
#ifndef SEQUENCE_WITH_PRIORITY_H
#define SEQUENCE_WITH_PRIORITY_H

namespace sarathi
{
class SequenceWithPriority
{
    public:
        SequenceWithPriority (
            float priority,
            pybind11::object seq
        );

        float priority;
        pybind11::object seq;

        bool operator<(const SequenceWithPriority& other) const;
        bool operator==(const SequenceWithPriority& other) const;
        bool operator>(const SequenceWithPriority& other) const;
    };
}

#endif