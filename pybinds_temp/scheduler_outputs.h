#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

namespace sarathi
{
class SchedulerOutputs
{
    public:
        SchedulerOutputs(
            int id,
            std::vector<pybind11::str> ignored_seq_ids,
            std::vector<pybind11::str> preempted_seq_ids,
            std::vector<pybind11::object> scheduled_seq_metadata_list
        );

        //class methods here
        bool is_empty();
        bool has_no_output();
        std::vector<py::str> seq_ids();
        //py::str __repr__();

    private:
        std::vector<int> promptChunkLens;
        int numBatchedPromptTokens;
        int numBatchedOutputTokens;
        int numBatchedTokens;
};


//==============================================================================
} // namespace sarathi
//==============================================================================

