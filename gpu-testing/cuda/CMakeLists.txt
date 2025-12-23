cmake_minimum_required(VERSION 3.18)

# =============================================================================
# CUDA Compiler Detection
# =============================================================================
# Try to find nvcc in PATH before CMake's default search
# This helps when multiple CUDA versions are installed
# =============================================================================
if(NOT DEFINED CMAKE_CUDA_COMPILER)
    find_program(CUDA_NVCC_EXECUTABLE nvcc
        HINTS 
            ENV CUDA_HOME
            ENV CUDA_PATH
        PATH_SUFFIXES bin
    )
    if(CUDA_NVCC_EXECUTABLE)
        set(CMAKE_CUDA_COMPILER ${CUDA_NVCC_EXECUTABLE} CACHE FILEPATH "CUDA compiler")
        message(STATUS "Found nvcc: ${CUDA_NVCC_EXECUTABLE}")
    endif()
endif()

project(RamboCUDA LANGUAGES CXX CUDA)

# =============================================================================
# C++ and CUDA Standards
# =============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Check for C++20 support
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "10.0")
        message(FATAL_ERROR "GCC 10.0 or later required for C++20 support. Found: ${CMAKE_CXX_COMPILER_VERSION}")
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "10.0")
        message(FATAL_ERROR "Clang 10.0 or later required for C++20 support. Found: ${CMAKE_CXX_COMPILER_VERSION}")
    endif()
endif()
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER_VERSION}")

# =============================================================================
# Build Type
# =============================================================================
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# =============================================================================
# CUDA Architecture Detection
# =============================================================================
# Auto-detect GPU architecture if not specified
# Can be overridden with: cmake -DCMAKE_CUDA_ARCHITECTURES=86 ..
#
# Common values:
#   75 - Turing (RTX 20xx, GTX 16xx)
#   80 - Ampere (A100)
#   86 - Ampere (RTX 30xx)
#   89 - Ada Lovelace (RTX 40xx, RTX 2000 Ada)
#   90 - Hopper (H100)
# =============================================================================
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures")
endif()
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# Enable separable compilation for complex projects
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# =============================================================================
# Build executable
# =============================================================================
add_executable(rambo_cuda main.cu)

# Optimization flags
target_compile_options(rambo_cuda PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:
        $<$<CONFIG:Release>:-O3 --use_fast_math>
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        $<$<CONFIG:Release>:-O3>
    >
)

# =============================================================================
# Platform-specific linking
# =============================================================================
if(UNIX AND NOT APPLE)
    target_link_libraries(rambo_cuda PRIVATE m)
endif()

# =============================================================================
# Print configuration
# =============================================================================
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA Standard: ${CMAKE_CUDA_STANDARD}")
