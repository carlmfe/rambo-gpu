cmake_minimum_required(VERSION 3.18)

project(RamboAlpaka CXX)

# =============================================================================
# C++ Standard (C++20 required for concepts)
# =============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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

# =============================================================================
# Build Type
# =============================================================================
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# =============================================================================
# Backend Selection
# =============================================================================
# Use: cmake -DALPAKA_BACKEND=CUDA ..   (NVIDIA GPUs)
#      cmake -DALPAKA_BACKEND=HIP ..    (AMD GPUs)
#      cmake -DALPAKA_BACKEND=SYCL ..   (Intel GPUs, or portable)
#      cmake -DALPAKA_BACKEND=CPU ..    (CPU serial, works everywhere)
#      cmake -DALPAKA_BACKEND=OMP ..    (OpenMP threads on CPU)
# =============================================================================
set(ALPAKA_BACKEND "CPU" CACHE STRING "Alpaka backend: CUDA, HIP, SYCL, CPU, OMP")
set_property(CACHE ALPAKA_BACKEND PROPERTY STRINGS CUDA HIP SYCL CPU OMP)
message(STATUS "Alpaka backend: ${ALPAKA_BACKEND}")

# =============================================================================
# Find Alpaka
# =============================================================================
# Alpaka can be found via:
#   1. CMAKE_PREFIX_PATH pointing to Alpaka installation
#   2. alpaka_ROOT environment variable  
#   3. System-wide installation
#
# Example: cmake -Dalpaka_ROOT=/path/to/alpaka/install ..
# =============================================================================

# Configure backend-specific settings BEFORE finding alpaka
set(alpaka_RELOCATABLE_DEVICE_CODE ON)

if(ALPAKA_BACKEND STREQUAL "CUDA")
    # Check if CUDA is available
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "CUDA backend requested but no CUDA compiler found. "
                            "Use -DALPAKA_BACKEND=CPU for CPU-only builds.")
    endif()
    enable_language(CUDA)
    
    set(alpaka_ACC_GPU_CUDA_ENABLE ON CACHE BOOL "" FORCE)
    
    # Auto-detect CUDA architecture if not specified
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures")
    endif()
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    
    add_compile_definitions(ALPAKA_USE_CUDA)
    
elseif(ALPAKA_BACKEND STREQUAL "CPU")
    set(alpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE ON CACHE BOOL "" FORCE)
    add_compile_definitions(ALPAKA_USE_CPU_SERIAL)

elseif(ALPAKA_BACKEND STREQUAL "HIP")
    # AMD GPU via HIP/ROCm
    # Alpaka must be built with HIP support
    set(alpaka_ACC_GPU_HIP_ENABLE ON CACHE BOOL "" FORCE)
    
    # Auto-detect AMD GPU architecture
    set(HIP_GPU_ARCH "native" CACHE STRING "AMD GPU architecture: native (auto-detect), or gfxXXXX")
    if(HIP_GPU_ARCH STREQUAL "native")
        find_program(ROCMINFO rocminfo)
        if(ROCMINFO)
            execute_process(
                COMMAND rocminfo
                OUTPUT_VARIABLE ROCM_INFO_OUTPUT
                ERROR_QUIET
                RESULT_VARIABLE ROCMINFO_RESULT
            )
            if(ROCMINFO_RESULT EQUAL 0)
                string(REGEX MATCH "gfx[0-9a-z]+" DETECTED_HIP_ARCH "${ROCM_INFO_OUTPUT}")
                if(DETECTED_HIP_ARCH)
                    set(HIP_GPU_ARCH "${DETECTED_HIP_ARCH}")
                    message(STATUS "Auto-detected AMD GPU: ${HIP_GPU_ARCH}")
                else()
                    message(WARNING "Failed to parse AMD GPU, using gfx906")
                    set(HIP_GPU_ARCH "gfx906")
                endif()
            endif()
        else()
            message(WARNING "rocminfo not found, using default gfx906")
            set(HIP_GPU_ARCH "gfx906")
        endif()
    endif()
    message(STATUS "HIP GPU architecture: ${HIP_GPU_ARCH}")
    set(CMAKE_HIP_ARCHITECTURES ${HIP_GPU_ARCH} CACHE STRING "HIP architectures")
    
    add_compile_definitions(ALPAKA_USE_HIP)

elseif(ALPAKA_BACKEND STREQUAL "SYCL")
    # Intel GPU (or portable) via SYCL
    # Alpaka must be built with SYCL support
    set(alpaka_ACC_SYCL_ENABLE ON CACHE BOOL "" FORCE)
    add_compile_definitions(ALPAKA_USE_SYCL)
    
elseif(ALPAKA_BACKEND STREQUAL "OMP")
    find_package(OpenMP REQUIRED)
    set(alpaka_ACC_CPU_B_OMP2_T_SEQ_ENABLE ON CACHE BOOL "" FORCE)
    add_compile_definitions(ALPAKA_USE_CPU_OMP)
endif()

find_package(alpaka REQUIRED)
message(STATUS "Alpaka version: ${alpaka_VERSION}")

# =============================================================================
# Build executable
# =============================================================================
alpaka_add_executable(rambo_alpaka main.cpp)
target_link_libraries(rambo_alpaka PUBLIC alpaka::alpaka)

if(ALPAKA_BACKEND STREQUAL "OMP")
    target_link_libraries(rambo_alpaka PUBLIC OpenMP::OpenMP_CXX)
endif()

# =============================================================================
# Platform-specific linking
# =============================================================================
if(UNIX AND NOT APPLE)
    target_link_libraries(rambo_alpaka PRIVATE m)
endif()

# =============================================================================
# Print configuration
# =============================================================================
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
