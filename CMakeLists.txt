cmake_minimum_required(VERSION 3.21)

project(sarathi_extensions LANGUAGES CXX)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/FindFlashinfer.cmake)

#
# Supported python versions.  These versions will be searched in order, the
# first match will be selected.  These should be kept in sync with setup.py.
#
set(PYTHON_SUPPORTED_VERSIONS "3.8" "3.9" "3.10" "3.11")

# Supported NVIDIA architectures.
set(CUDA_SUPPORTED_ARCHS "7.0;7.5;8.0;8.6;8.9;9.0")

# Supported AMD GPU architectures.
set(HIP_SUPPORTED_ARCHS "gfx908;gfx90a;gfx942;gfx1100")

#
# Supported/expected torch versions for CUDA/ROCm.
#
# Currently, having an incorrect pytorch version results in a warning
# rather than an error.
#
# Note: the CUDA torch version is derived from pyproject.toml and various
# requirements.txt files and should be kept consistent.  The ROCm torch
# versions are derived from Dockerfile.rocm
#

set(TORCH_SUPPORTED_VERSION_CUDA "2.1.2")

#
# Try to find python package with an executable that exactly matches
# `SARATHI_PYTHON_EXECUTABLE` and is one of the supported versions.
#
if (SARATHI_PYTHON_EXECUTABLE)
  find_python_from_executable(${SARATHI_PYTHON_EXECUTABLE} "${PYTHON_SUPPORTED_VERSIONS}")
else()
  message(FATAL_ERROR
    "Please set SARATHI_PYTHON_EXECUTABLE to the path of the desired python version"
    " before running cmake configure.")
endif()

#
# Update cmake's `CMAKE_PREFIX_PATH` with torch location.
#
append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")

#
# Import torch cmake configuration.
# Torch also imports CUDA (and partially HIP) languages with some customizations,
# so there is no need to do this explicitly with check_language/enable_language,
# etc.
#
find_package(Torch REQUIRED)

# Add debug symbols
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

execute_process(
        COMMAND python3 -c "import torch; print(torch._C._PYBIND11_COMPILER_TYPE, end='')"
        OUTPUT_VARIABLE _PYBIND11_COMPILER_TYPE
)
execute_process(
        COMMAND python3 -c "import torch; print(torch._C._PYBIND11_STDLIB, end='')"
        OUTPUT_VARIABLE _PYBIND11_STDLIB
)
execute_process(
        COMMAND python3 -c "import torch; print(torch._C._PYBIND11_BUILD_ABI, end='')"
        OUTPUT_VARIABLE _PYBIND11_BUILD_ABI
)

message(STATUS "PYBIND11_COMPILER_TYPE:" ${_PYBIND11_COMPILER_TYPE})
message(STATUS "PYBIND11_STDLIB:" ${_PYBIND11_STDLIB})
message(STATUS "PYBIND11_BUILD_ABI:" ${_PYBIND11_BUILD_ABI})

add_compile_definitions(PYBIND11_COMPILER_TYPE="${_PYBIND11_COMPILER_TYPE}" PYBIND11_STDLIB="${_PYBIND11_STDLIB}" PYBIND11_BUILD_ABI="${_PYBIND11_BUILD_ABI}")


#
# Normally `torch.utils.cpp_extension.CUDAExtension` would add
# `libtorch_python.so` for linking against an extension. Torch's cmake
# configuration does not include this library (presumably since the cmake
# config is used for standalone C++ binaries that link against torch).
# The `libtorch_python.so` library defines some of the glue code between
# torch/python via pybind and is required by SARATHI extensions for this
# reason. So, add it by manually using `append_torchlib_if_found` from
# torch's cmake setup.
#
append_torchlib_if_found(torch_python)

#
# Set up GPU language and check the torch version and warn if it isn't
# what is expected.
#
if (CUDA_FOUND)
  set(SARATHI_GPU_LANG "CUDA")

  if (NOT Torch_VERSION VERSION_EQUAL ${TORCH_SUPPORTED_VERSION_CUDA})
    message(WARNING "Pytorch version ${TORCH_SUPPORTED_VERSION_CUDA} "
      "expected for CUDA build, saw ${Torch_VERSION} instead.")
  endif()
else()
    message(FATAL_ERROR "Can't find CUDA installation.")
endif()

#
# Override the GPU architectures detected by cmake/torch and filter them by
# the supported versions for the current language.
# The final set of arches is stored in `SARATHI_GPU_ARCHES`.
#
override_gpu_arches(SARATHI_GPU_ARCHES
  ${SARATHI_GPU_LANG}
  "${${SARATHI_GPU_LANG}_SUPPORTED_ARCHS}")

#
# Query torch for additional GPU compilation flags for the given
# `SARATHI_GPU_LANG`.
# The final set of arches is stored in `SARATHI_GPU_FLAGS`.
#
get_torch_gpu_compiler_flags(SARATHI_GPU_FLAGS ${SARATHI_GPU_LANG})

#
# Set nvcc parallelism.
#
if(NVCC_THREADS AND SARATHI_GPU_LANG STREQUAL "CUDA")
  list(APPEND SARATHI_GPU_FLAGS "--threads=${NVCC_THREADS}")
endif()

#
# Define extension targets
#

#
# _C extension
#

set(SARATHI_KERNEL_COMMONS_EXT_SRC
  "csrc/kernels/pos_encoding_kernels.cu"
  "csrc/kernels/layernorm_kernels.cu"
  "csrc/kernels/activation_kernels.cu"
  "csrc/kernels/moe_align_block_size_kernels.cu"
  "csrc/kernels/moe_topk_softmax_kernels.cu"
)

define_gpu_static_target(
  _kernels_common
  DESTINATION sarathi
  LANGUAGE ${SARATHI_GPU_LANG}
  SOURCES ${SARATHI_KERNEL_COMMONS_EXT_SRC}
  COMPILE_FLAGS ${SARATHI_GPU_FLAGS}
  ARCHITECTURES ${SARATHI_GPU_ARCHES}
)

set(SARATHI_KERNELS_EXT_SRC
  "csrc/kernels/pybind.cpp"
)

define_gpu_extension_target(
  _kernels_C
  DESTINATION sarathi
  LANGUAGE ${SARATHI_GPU_LANG}
  SOURCES ${SARATHI_KERNELS_EXT_SRC}
  COMPILE_FLAGS ${SARATHI_GPU_FLAGS}
  ARCHITECTURES ${SARATHI_GPU_ARCHES}
  LIBRARIES _kernels_common
  WITH_SOABI)

set(SARATHI_MODEL_EXECUTOR_EXT_SRC
  "csrc/model_executor/pybind.cpp"
  "csrc/model_executor/parallel_utils/ProcessGroupWrapper.cpp"
  "csrc/model_executor/parallel_utils/ParallelOps.cpp"
  "csrc/model_executor/models/Llama.cpp"
  "csrc/model_executor/layers/RotaryEmbedding.cpp"
  "csrc/model_executor/layers/NormLayers.cpp"
  "csrc/model_executor/layers/LinearLayers.cpp"
  "csrc/model_executor/layers/FlashinferAttentionWrapper.cpp"
)

define_gpu_extension_target(
  _model_executor_C
  DESTINATION sarathi
  LANGUAGE ${SARATHI_GPU_LANG}
  SOURCES ${SARATHI_MODEL_EXECUTOR_EXT_SRC}
  COMPILE_FLAGS ${SARATHI_GPU_FLAGS}
  ARCHITECTURES ${SARATHI_GPU_ARCHES}
  LIBRARIES _kernels_common Flashinfer::Flashinfer
  WITH_SOABI)


#####
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)  
#
set(VAJRA_BASE_SCHEDULER_EXT_SRC
"pybinds_temp/base_scheduler_pybinds.cpp"
"pybinds_temp/base_scheduler.cpp"
"pybinds_temp/scheduler_outputs_pybinds.cpp"
"pybinds_temp/scheduler_outputs.cpp"
"pybinds_temp/sequence_status.cpp"
"pybinds_temp/sequence_status_pybinds.cpp"
"pybinds_temp/sequence_with_priority.cpp"
"pybinds_temp/sarathi_scheduler.cpp"
)
#

# install()
pybind11_add_module(_base_scheduler_C ${VAJRA_BASE_SCHEDULER_EXT_SRC})
target_link_libraries(_base_scheduler_C PUBLIC pybind11::module pybind11::embed)
install(TARGETS _base_scheduler_C DESTINATION .)


####
#
# Add the `default` target which detects which extensions should be
# built based on platform/architecture.  This is the same logic that
# setup.py uses to select which extensions should be built and should
# be kept in sync.
#
# The `default` target makes direct use of cmake easier since knowledge
# of which extensions are supported has been factored in, e.g.
#
# mkdir build && cd build
# cmake -G Ninja -DSARATHI_PYTHON_EXECUTABLE=`which python3` -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=../sarathi ..
# cmake --build . --target default
#
add_custom_target(default)
add_dependencies(default _kernels_C)
add_dependencies(default _model_executor_C)
add_dependencies(default _base_scheduler_C)
