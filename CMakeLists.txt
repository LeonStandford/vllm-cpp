cmake_minimum_required(VERSION 3.24)

option(USE_HIP "Build with HIP for AMD GPUs" OFF)

if(NOT USE_HIP AND NOT CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
    # nvcc lives in different places per distro (/usr/local/cuda on Ubuntu,
    # /opt/cuda on Arch), so look it up instead of hardcoding one path
    find_program(CMAKE_CUDA_COMPILER
        NAMES nvcc
        HINTS ENV CUDA_PATH ENV CUDA_HOME /usr/local/cuda /opt/cuda
        PATH_SUFFIXES bin)
    if(NOT CMAKE_CUDA_COMPILER)
        # let CMake's own detection produce the error message
        unset(CMAKE_CUDA_COMPILER CACHE)
    endif()
endif()

if(USE_HIP)
    # project(... LANGUAGES ... HIP) honors -DCMAKE_HIP_ARCHITECTURES, otherwise
    # auto-detects the host GPU and errors on a no-GPU build host.
    project(tiny-vllm LANGUAGES CXX HIP)
else()
    project(tiny-vllm LANGUAGES CXX CUDA)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT USE_HIP)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O2")
if(USE_HIP)
    set(CMAKE_HIP_FLAGS_RELEASE "-O2")
    set(CMAKE_HIP_FLAGS_DEBUG "-g -DDEBUG")
else()
    set(CMAKE_CUDA_FLAGS_RELEASE "-O2")
    set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -DDEBUG")
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g -DDEBUG")

if(USE_HIP)
    find_package(hipblas REQUIRED)
    find_package(hip REQUIRED)
else()
    find_package(CUDAToolkit REQUIRED)
endif()

add_executable(tiny-vllm
    src/main.cpp
    src/kernels.cu
    src/tokenizer.cpp
)

if(USE_HIP)
    # Both main.cpp and kernels.cu need HIP compilation because they use bfloat16 types
    # which require the HIP compiler (hip_bf16.h uses clang-specific builtins)
    set_source_files_properties(src/main.cpp src/kernels.cu PROPERTIES LANGUAGE HIP)
    target_compile_definitions(tiny-vllm PRIVATE USE_HIP)
    set_target_properties(tiny-vllm PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
endif()

target_include_directories(tiny-vllm PRIVATE src)
target_include_directories(tiny-vllm PRIVATE include)

if(USE_HIP)
    target_link_libraries(tiny-vllm PRIVATE
        hip::host
        roc::hipblas
    )
else()
    target_link_libraries(tiny-vllm PRIVATE
        CUDA::cublas
        CUDA::cudart
    )
endif()
