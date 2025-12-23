cmake_minimum_required(VERSION 3.18)
project(rambo_omp CXX)

# =============================================================================
# C++ Standard
# =============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Check for C++20 support (OpenMP offloading requires Clang)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "10.0")
        message(FATAL_ERROR "Clang 10.0 or later required for C++20 support. Found: ${CMAKE_CXX_COMPILER_VERSION}")
    endif()
else()
    message(WARNING "OpenMP GPU offloading typically requires Clang. Current compiler: ${CMAKE_CXX_COMPILER_ID}")
endif()
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

# =============================================================================
# GPU Architecture
# =============================================================================
# Common values: sm_80 (Ampere), sm_86 (GA102), sm_89 (Ada), sm_90 (Hopper)
# Note: sm_70 (Volta) and sm_75 (Turing) were removed in CUDA 13.0
# Set to "native" for auto-detection, or specify manually (e.g., sm_89)
set(CUDA_GPU_ARCH "native" CACHE STRING "CUDA GPU architecture: native (auto-detect), or sm_XX")

# Auto-detect GPU architecture if set to "native"
if(CUDA_GPU_ARCH STREQUAL "native")
    find_program(NVIDIA_SMI nvidia-smi)
    if(NVIDIA_SMI)
        execute_process(
            COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE GPU_COMPUTE_CAP
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE NVIDIA_SMI_RESULT
        )
        if(NVIDIA_SMI_RESULT EQUAL 0 AND GPU_COMPUTE_CAP)
            # Handle multiple GPUs - take the first one
            string(REGEX MATCH "^[0-9]+\\.[0-9]+" GPU_COMPUTE_CAP "${GPU_COMPUTE_CAP}")
            # Convert 8.9 -> sm_89
            string(REPLACE "." "" GPU_ARCH_NUM "${GPU_COMPUTE_CAP}")
            set(CUDA_GPU_ARCH "sm_${GPU_ARCH_NUM}")
            message(STATUS "Auto-detected GPU compute capability: ${GPU_COMPUTE_CAP} -> ${CUDA_GPU_ARCH}")
        else()
            message(WARNING "Failed to detect GPU architecture, using default sm_80")
            set(CUDA_GPU_ARCH "sm_80")
        endif()
    else()
        message(WARNING "nvidia-smi not found, using default architecture sm_80")
        set(CUDA_GPU_ARCH "sm_80")
    endif()
endif()
message(STATUS "CUDA GPU architecture: ${CUDA_GPU_ARCH}")

# =============================================================================
# OpenMP Offloading Flags
# =============================================================================
# Requires clang++ with OpenMP offloading support
set(OPENMP_FLAGS
    -fopenmp
    -fopenmp-targets=nvptx64-nvidia-cuda
    -Xopenmp-target=nvptx64-nvidia-cuda -march=${CUDA_GPU_ARCH}
)

add_executable(rambo_omp
    main.cpp
    integrator.cpp
    rambo_omp.cpp
    rng.cpp
)

target_compile_options(rambo_omp PRIVATE ${OPENMP_FLAGS})
target_link_options(rambo_omp PRIVATE ${OPENMP_FLAGS})

