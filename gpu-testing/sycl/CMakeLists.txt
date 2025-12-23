cmake_minimum_required(VERSION 3.18)

# =============================================================================
# SYCL Configuration
# =============================================================================
# SYCL requires a SYCL-enabled compiler (Intel DPC++, AdaptiveCpp, etc.)
#
# For Intel DPC++ with CUDA backend:
#   export CXX=clang++  (from Intel LLVM/DPC++ build)
#   cmake -DSYCL_BACKEND=CUDA ..
#
# For Intel DPC++ with Intel GPU:
#   cmake -DSYCL_BACKEND=INTEL ..
#
# For AdaptiveCpp (formerly hipSYCL):
#   cmake -DSYCL_BACKEND=ADAPTIVECPP ..
# =============================================================================

# =============================================================================
# SYCL Backend Selection
# =============================================================================
set(SYCL_BACKEND "CUDA" CACHE STRING "SYCL backend: CUDA, INTEL, ADAPTIVECPP")
set_property(CACHE SYCL_BACKEND PROPERTY STRINGS CUDA INTEL ADAPTIVECPP)
message(STATUS "SYCL backend: ${SYCL_BACKEND}")

# =============================================================================
# Find SYCL Compiler
# =============================================================================
# Look for clang++ with SYCL support
if(NOT DEFINED CMAKE_CXX_COMPILER)
    find_program(SYCL_COMPILER 
        NAMES clang++ icpx syclcc
        HINTS ENV PATH
        DOC "SYCL-enabled C++ compiler"
    )
    if(SYCL_COMPILER)
        set(CMAKE_CXX_COMPILER ${SYCL_COMPILER} CACHE STRING "C++ compiler" FORCE)
        message(STATUS "Found SYCL compiler: ${SYCL_COMPILER}")
    else()
        message(WARNING "No SYCL compiler found. Please set CMAKE_CXX_COMPILER manually.")
    endif()
endif()

project(RamboSYCL CXX)

# =============================================================================
# C++ Standard
# =============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Check for C++20 support (SYCL compilers are typically Clang-based)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|IntelLLVM")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "10.0")
        message(FATAL_ERROR "Clang/SYCL compiler 10.0 or later required for C++20 support. Found: ${CMAKE_CXX_COMPILER_VERSION}")
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "10.0")
        message(FATAL_ERROR "GCC 10.0 or later required for C++20 support. Found: ${CMAKE_CXX_COMPILER_VERSION}")
    endif()
endif()
message(STATUS "SYCL/C++ compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

# =============================================================================
# Build Type
# =============================================================================
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# =============================================================================
# CUDA GPU Architecture (for CUDA backend)
# =============================================================================
# Common values: sm_80 (Ampere), sm_86 (GA102), sm_89 (Ada), sm_90 (Hopper)
# Note: sm_70 (Volta) and sm_75 (Turing) were removed in CUDA 13.0
# Set to "native" for auto-detection, or specify manually (e.g., sm_89)
set(CUDA_GPU_ARCH "native" CACHE STRING "CUDA GPU architecture: native (auto-detect), or sm_XX")

# Auto-detect GPU architecture if set to "native"
if(CUDA_GPU_ARCH STREQUAL "native" AND SYCL_BACKEND STREQUAL "CUDA")
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

# =============================================================================
# Build executable
# =============================================================================
add_executable(rambo_sycl main.cpp)
target_include_directories(rambo_sycl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# =============================================================================
# Backend-specific flags
# =============================================================================
if(SYCL_BACKEND STREQUAL "CUDA")
    # Intel DPC++ / Intel LLVM with CUDA backend
    target_compile_options(rambo_sycl PRIVATE 
        -fsycl
        -fsycl-targets=nvptx64-nvidia-cuda
        -Xsycl-target-backend 
        --cuda-gpu-arch=${CUDA_GPU_ARCH}
        $<$<CONFIG:Release>:-O3>
    )
    target_link_options(rambo_sycl PRIVATE
        -fsycl
        -fsycl-targets=nvptx64-nvidia-cuda
        -Xsycl-target-backend 
        --cuda-gpu-arch=${CUDA_GPU_ARCH}
    )
    
elseif(SYCL_BACKEND STREQUAL "INTEL")
    # Intel DPC++ for Intel GPUs
    target_compile_options(rambo_sycl PRIVATE 
        -fsycl
        $<$<CONFIG:Release>:-O3>
    )
    target_link_options(rambo_sycl PRIVATE
        -fsycl
    )
    
elseif(SYCL_BACKEND STREQUAL "ADAPTIVECPP")
    # AdaptiveCpp (formerly hipSYCL)
    find_package(AdaptiveCpp REQUIRED)
    add_sycl_to_target(TARGET rambo_sycl)
endif()

# =============================================================================
# Platform-specific linking
# =============================================================================
if(UNIX AND NOT APPLE)
    target_link_libraries(rambo_sycl PRIVATE m)
endif()

# =============================================================================
# Print configuration
# =============================================================================
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
if(SYCL_BACKEND STREQUAL "CUDA")
    message(STATUS "CUDA GPU architecture: ${CUDA_GPU_ARCH}")
endif()
