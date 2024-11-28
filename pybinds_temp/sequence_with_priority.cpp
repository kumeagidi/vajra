#include "sequence_with_priority.h"

using namespace sarathi;

SequenceWithPriority::SequenceWithPriority(
    float priority,
    pybind11::object seq
) : 
    priority(priority),
    seq(seq) 
{}

bool SequenceWithPriority::operator<(const SequenceWithPriority& other) const {
    return priority < other.priority;
}

bool SequenceWithPriority::operator==(const SequenceWithPriority& other) const {
    return priority == other.priority;
}

bool SequenceWithPriority::operator>(const SequenceWithPriority& other) const {
    return priority > other.priority;
}
