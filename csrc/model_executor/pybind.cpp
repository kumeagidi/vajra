#include <torch/extension.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include "Llama.h"
#include "FlashinferAttentionWrapper.h"
#include "LinearLayers.h"
#include "NormLayers.h"
#include "RotaryEmbedding.h"
#include "ProcessGroupWrapper.h"
//==============================================================================
namespace pybind11 {
namespace detail {
template <> struct type_caster<std::set<int>> {
public:
    PYBIND11_TYPE_CASTER(std::set<int>, _("Set[int]"));
    bool load(handle src, bool) {
        if (!py::isinstance<py::set>(src) && !py::isinstance<py::frozenset>(src))
            return false;
        for (auto item : src) {
            if (!py::isinstance<py::int_>(item))
                return false;
            value.insert(item.cast<int>());
        }
        return true;
    }
    static handle cast(const std::set<int> &src, return_value_policy, handle) {
        py::set s;
        for (int v : src)
            s.add(py::cast(v));
        return s.release();
    }
};
}} // namespace pybind11::detail
//==============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::module modelExecutor = m.def_submodule("model_executor", "Sarathi model executor");

    pybind11::class_<sarathi::LlamaMLP, std::shared_ptr<sarathi::LlamaMLP>>(modelExecutor, "LlamaMLP")
        .def(pybind11::init<std::shared_ptr<sarathi::ColumnParallelLinear>, std::shared_ptr<sarathi::RowParallelLinear>>())
        .def("forward", &sarathi::LlamaMLP::Forward);

    pybind11::class_<sarathi::LlamaAttention, std::shared_ptr<sarathi::LlamaAttention>>(modelExecutor, "LlamaAttention")
        .def(pybind11::init<int, int, float,
                      std::shared_ptr<sarathi::ColumnParallelLinear>, 
                      std::shared_ptr<sarathi::RowParallelLinear>, 
                      std::shared_ptr<sarathi::RotaryEmbedding>, 
                      std::shared_ptr<sarathi::FlashInferAttentionWrapper>>())
        .def("forward", &sarathi::LlamaAttention::Forward);

    pybind11::class_<sarathi::LlamaDecoderLayer, std::shared_ptr<sarathi::LlamaDecoderLayer>>(modelExecutor, "LlamaDecoderLayer")
        .def(pybind11::init<
                      std::shared_ptr<sarathi::LlamaAttention>, 
                      std::shared_ptr<sarathi::LlamaMLP>, 
                      std::shared_ptr<sarathi::RMSNorm>, 
                      std::shared_ptr<sarathi::RMSNorm>>())
        .def("forward", &sarathi::LlamaDecoderLayer::Forward);

    pybind11::class_<sarathi::LlamaModel, std::shared_ptr<sarathi::LlamaModel>>(modelExecutor, "LlamaModel")
        .def(pybind11::init<std::shared_ptr<sarathi::VocabParallelEmbedding>, 
                      std::vector<std::shared_ptr<sarathi::LlamaDecoderLayer>>, 
                      std::shared_ptr<sarathi::RMSNorm>>())
        .def("forward", &sarathi::LlamaModel::Forward);

    pybind11::class_<sarathi::FlashInferAttentionWrapper, std::shared_ptr<sarathi::FlashInferAttentionWrapper>>(modelExecutor, "FlashInferAttentionWrapper")
        .def(pybind11::init<BatchPrefillWithPagedKVCachePyTorchWrapper&,
                      const std::shared_ptr<sarathi::ProcessGroupWrapper>,
                      int, int, int, int, unsigned int, float, unsigned int,
                      bool, int, float, float, float, bool, bool>())
        .def("begin_forward", &sarathi::FlashInferAttentionWrapper::BeginForward)
        .def("end_forward", &sarathi::FlashInferAttentionWrapper::EndForward)
        .def("forward", &sarathi::FlashInferAttentionWrapper::Forward);

    pybind11::class_<sarathi::ColumnParallelLinear, std::shared_ptr<sarathi::ColumnParallelLinear>>(modelExecutor, "ColumnParallelLinear")
        .def(pybind11::init<int, int, bool, int, bool, torch::Tensor, std::optional<torch::Tensor>, std::shared_ptr<sarathi::ProcessGroupWrapper>>())
        .def("forward", &sarathi::ColumnParallelLinear::Forward);

    pybind11::class_<sarathi::RowParallelLinear, std::shared_ptr<sarathi::RowParallelLinear>>(modelExecutor, "RowParallelLinear")
        .def(pybind11::init<int, int, bool, bool, int, int, bool, torch::Tensor, std::optional<torch::Tensor>, std::shared_ptr<sarathi::ProcessGroupWrapper>>())
        .def("forward", &sarathi::RowParallelLinear::Forward);

    pybind11::class_<sarathi::VocabParallelEmbedding, std::shared_ptr<sarathi::VocabParallelEmbedding>>(modelExecutor, "VocabParallelEmbedding")
        .def(pybind11::init<int, int, int, int, bool, int, int, int, torch::Tensor, std::shared_ptr<sarathi::ProcessGroupWrapper>>())
        .def("forward", &sarathi::VocabParallelEmbedding::Forward);

    pybind11::class_<sarathi::RMSNorm, std::shared_ptr<sarathi::RMSNorm>>(modelExecutor, "RMSNorm")
        .def(pybind11::init<torch::Tensor, double>())
        .def("forward", &sarathi::RMSNorm::Forward);

    pybind11::class_<sarathi::RotaryEmbedding, std::shared_ptr<sarathi::RotaryEmbedding>>(modelExecutor, "RotaryEmbedding")
        .def(pybind11::init<int, int, long, long, bool, torch::Tensor>())
        .def("forward", &sarathi::RotaryEmbedding::Forward);

    pybind11::class_<sarathi::ProcessGroupWrapper, std::shared_ptr<sarathi::ProcessGroupWrapper>>(modelExecutor, "ProcessGroupWrapper")
        .def(pybind11::init<c10::intrusive_ptr<c10d::ProcessGroup>, c10::intrusive_ptr<c10d::ProcessGroup>, const std::unordered_map<std::set<int>, c10::intrusive_ptr<c10d::ProcessGroup>>&>())
        .def("get_tensor_model_parallel_group", &sarathi::ProcessGroupWrapper::GetTensorModelParallelGroup)
        .def("get_pipeline_model_parallel_group", &sarathi::ProcessGroupWrapper::GetPipelineModelParallelGroup)
        .def("get_cache_model_parallel_group", &sarathi::ProcessGroupWrapper::GetCacheModelParallelGroup);
}
//==============================================================================