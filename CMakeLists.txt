cmake_minimum_required(VERSION 3.12...3.18)
project(sumcumprod_jax LANGUAGES CXX)

message(STATUS "Using CMake version " ${CMAKE_VERSION})

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

include_directories(${CMAKE_CURRENT_LIST_DIR}/lib)

# CPU op library
pybind11_add_module(cpu_ops ${CMAKE_CURRENT_LIST_DIR}/lib/cpu_ops.cc)

find_package(OpenMP)  # Added

if(OpenMP_CXX_FOUND)  # Added
    message("OpenMP is enabled")
    target_link_libraries(cpu_ops PUBLIC OpenMP::OpenMP_CXX)  # Added
    if (APPLE)
      add_compile_options(-Xpreprocessor -fopenmp -lomp)
    else()
      add_compile_options(-fopenmp)
else()
    message("OpenMP is not enabled")
endif()

install(TARGETS cpu_ops DESTINATION sumcumprod_jax)

if (SUMCUMPROD_JAX_CUDA)
  message(STATUS "Building with CUDA")
  enable_language(CUDA)
  include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  pybind11_add_module(
    gpu_ops
    ${CMAKE_CURRENT_LIST_DIR}/lib/kernels.cc.cu
    ${CMAKE_CURRENT_LIST_DIR}/lib/gpu_ops.cc)
  install(TARGETS gpu_ops DESTINATION sumcumprod_jax)
else()
  message(STATUS "Building without CUDA")
endif()
