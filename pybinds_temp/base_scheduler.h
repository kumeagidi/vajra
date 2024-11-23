#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

namespace sarathi
{
class BaseScheduler
{
    public:
        BaseScheduler(
            pybind11::object& model_config,
            pybind11::object& scheduler_config,
            pybind11::object& cache_config,
            pybind11::object& parallel_config,
            pybind11::object& waiting_queue,
            pybind11::object& replica_seq_manager,
            pybind11::object& metric_store,
        );

        //class methods here
        void reset_state();
        void add_seq(const pybind11::object& seq);
        void has_unfinished_seqs() const;
        void get_num_unfinished_seqs() const;
        virtual pybind11::object& _schedule();
        void add_to_new_seqs(pybind11::object& seq);
        std::vector<pybind11::object> get_new_seqs(); // could be pybind::list but is a list of Sequences (python objects)
        void add_seq_to_seq_manager(const pybind11::object& seq);
        pybind11::object& schedule();
        void free_finished_seqs();
        void on_step_completed();
        void _allocate(pybind11::object& seq);
        void _free_seq(pybind11::object& seq);
        void _append_slot(pybind11::object& seq);
        void _preempt(pybind11::object& seq);
        void _check_request_prompt_length(pybind11::object& seq);

    private:
        pybind11::object policy;
        pybind11::object block_manager;
        pybind11::object max_model_len;
};


//==============================================================================
} // namespace sarathi
//==============================================================================

