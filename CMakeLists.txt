message(STATUS "
  =============> PRELU FP16 SETTING <===============
  By default prelu plugin's fp16 mode is disable since it require device that sm version is equal or greater than SM_60, so you need to enable it manully.
  usage cmake -DENABLE_PRELU_FP16=ON ..
  ==================================================
")

message(STATUS "
  =============> USAGE <===============
  cmake -DENABLE_PRELU_FP16=ON/OFF -DSM_VERSION=xx -DBUILD_PYTHON=ON/OFF -DBUILD_TEST=ON/OFF ..
  =====================================
")

cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_FLAGS "-std=c++11")
project(tinytrt)

# set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib CACHE PATH "")
option(BUILD_PYTHON "compile python api" ON)
option(BUILD_TEST "compile test" OFF)

find_package(CUDA REQUIRED)

include(cmake/CUDA_utils.cmake)

set(SM_VERSION "" CACHE STRING "Description")

if (SM_VERSION)
    set(CUDA_targeted_archs ${SM_VERSION})
    CUDA_get_gencode_args(CUDA_gencode_flags ${CUDA_targeted_archs})
else ()
    # Discover what architectures does nvcc support
    CUDA_find_supported_arch_values(CUDA_supported_archs ${CUDA_known_archs})
    message(STATUS "CUDA supported archs: ${CUDA_supported_archs}")

    set(CUDA_TARGET_ARCHS_SORTED ${CUDA_TARGET_ARCHS})
    list(SORT CUDA_TARGET_ARCHS_SORTED)
    CUDA_find_supported_arch_values(CUDA_targeted_archs ${CUDA_TARGET_ARCHS_SORTED})
    message(STATUS "CUDA targeted archs: ${CUDA_targeted_archs}")
    if (NOT CUDA_targeted_archs)
        message(FATAL_ERROR "None of the provided CUDA architectures ({${CUDA_TARGET_ARCHS}}) is supported by nvcc, use one or more of: ${CUDA_supported_archs}")
    endif ()
    CUDA_get_gencode_args(CUDA_gencode_flags ${CUDA_targeted_archs})
endif ()

# Add ptx & bin flags for cuda
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_gencode_flags}")

option(ENABLE_PRELU_FP16 "" OFF)
if (ENABLE_PRELU_FP16)
    add_definitions(-DFP16_PRELU)
endif ()

include_directories(spdlog)
include_directories(pybind11/include)
include_directories(./)


# 这是 nms 自定义层
include_directories(./concurrentqueue)


message(STATUS "
  =============> TensorRT <===============
  =====================================
")

# TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES include)
MESSAGE(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")

message(STATUS "
Generated gencode flags: ${CUDA_gencode_flags} 
BUILD_PYTHON : ${BUILD_PYTHON} 
BUILD_TEST : ${BUILD_TEST} 
ENABLE_PRELU_FP16 : ${ENABLE_PRELU_FP16} 
")

# cub
if (NOT DEFINED CUB_ROOT_DIR)
    set(CUB_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cub")
endif ()
INCLUDE_DIRECTORIES(${CUB_ROOT_DIR})


file(GLOB_RECURSE trt_source
        ZyTrt.cpp
        TrtBuffer.cpp
        Int8EntropyCalibrator.cpp
        )
cuda_add_library(zytrt SHARED ${trt_source})
target_compile_options(zytrt PUBLIC -std=c++11 -Wall)
#target_compile_options(tinytrt PUBLIC -std=c++11 -Wall -Wfloat-conversion)
set_target_properties(zytrt PROPERTIES POSITION_INDEPENDENT_CODE ON)


if (BUILD_PYTHON)
    # set(Python3_ROOT_DIR /root/miniconda3/bin)
    # find_package(Python3 REQUIRED)
    include_directories(${PYTHON_INCLUDE_DIRS})
    add_subdirectory(pybind11)
    pybind11_add_module(pytrt SHARED PyZyTrt.cpp)
    target_link_libraries(pytrt PRIVATE zytrt)
    target_link_libraries(pytrt PRIVATE nvinfer)
    target_link_libraries(pytrt PRIVATE nvinfer_plugin)
    target_link_libraries(pytrt PRIVATE nvparsers)
    target_link_libraries(pytrt PRIVATE nvonnxparser)
    target_link_libraries(pytrt PRIVATE nvcaffe_parser)
endif ()
