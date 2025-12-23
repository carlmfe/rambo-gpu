cmake_minimum_required(VERSION 3.18)

# =============================================================================
# SYCL Configuration - Multi-Vendor GPU Support
# =============================================================================
# SYCL is a portable standard that can target NVIDIA, AMD, and Intel GPUs.
#
# For NVIDIA GPUs (Intel DPC++/LLVM with CUDA backend):
#   cmake -DSYCL_BACKEND=CUDA ..
#
# For AMD GPUs (Intel DPC++/LLVM with HIP backend):
#   cmake -DSYCL_BACKEND=HIP ..
#
# For Intel GPUs (Intel DPC++ / oneAPI):
#   cmake -DSYCL_BACKEND=INTEL ..
#
# For AdaptiveCpp (formerly hipSYCL) - supports multiple backends:
#   cmake -DSYCL_BACKEND=ADAPTIVECPP -DADAPTIVECPP_TARGETS="cuda:sm_89" ..
#   cmake -DSYCL_BACKEND=ADAPTIVECPP -DADAPTIVECPP_TARGETS="hip:gfx1100" ..
# =============================================================================

# =============================================================================
# SYCL Backend Selection
# =============================================================================
set(SYCL_BACKEND "CUDA" CACHE STRING "SYCL backend: CUDA, HIP, INTEL, ADAPTIVECPP")
set_property(CACHE SYCL_BACKEND PROPERTY STRINGS CUDA HIP INTEL ADAPTIVECPP)
message(STATUS "SYCL backend: ${SYCL_BACKEND}")

# =============================================================================
# Find SYCL Compiler
# =============================================================================
# Look for clang++ with SYCL support
if(NOT DEFINED CMAKE_CXX_COMPILER)
    find_program(SYCL_COMPILER 
        NAMES clang++ icpx acpp syclcc
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
# GPU Architecture Detection (Multi-Vendor)
# =============================================================================
# NVIDIA: sm_80, sm_86, sm_89, sm_90 (use nvidia-smi to detect)
# AMD: gfx906, gfx908, gfx90a, gfx1030, gfx1100 (use rocminfo to detect)
# Intel: auto-detected by the compiler
# =============================================================================

# NVIDIA GPU architecture
set(CUDA_GPU_ARCH "native" CACHE STRING "NVIDIA GPU architecture: native (auto-detect), or sm_XX")

# AMD GPU architecture  
set(HIP_GPU_ARCH "native" CACHE STRING "AMD GPU architecture: native (auto-detect), or gfxXXXX")

# Auto-detect NVIDIA GPU architecture
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
            string(REGEX MATCH "^[0-9]+\\.[0-9]+" GPU_COMPUTE_CAP "${GPU_COMPUTE_CAP}")
            string(REPLACE "." "" GPU_ARCH_NUM "${GPU_COMPUTE_CAP}")
            set(CUDA_GPU_ARCH "sm_${GPU_ARCH_NUM}")
            message(STATUS "Auto-detected NVIDIA GPU: ${GPU_COMPUTE_CAP} -> ${CUDA_GPU_ARCH}")
        else()
            message(WARNING "Failed to detect NVIDIA GPU, using default sm_80")
            set(CUDA_GPU_ARCH "sm_80")
        endif()
    else()
        message(WARNING "nvidia-smi not found, using default sm_80")
        set(CUDA_GPU_ARCH "sm_80")
    endif()
endif()

# Auto-detect AMD GPU architecture
if(HIP_GPU_ARCH STREQUAL "native" AND SYCL_BACKEND STREQUAL "HIP")
    find_program(ROCMINFO rocminfo)
    if(ROCMINFO)
        execute_process(
            COMMAND rocminfo
            OUTPUT_VARIABLE ROCM_INFO_OUTPUT
            ERROR_QUIET
            RESULT_VARIABLE ROCMINFO_RESULT
        )
        if(ROCMINFO_RESULT EQUAL 0)
            # Extract gfx architecture from rocminfo output
            string(REGEX MATCH "gfx[0-9a-z]+" HIP_GPU_ARCH "${ROCM_INFO_OUTPUT}")
            if(HIP_GPU_ARCH)
                message(STATUS "Auto-detected AMD GPU: ${HIP_GPU_ARCH}")
            else()
                message(WARNING "Failed to parse AMD GPU architecture, using default gfx906")
                set(HIP_GPU_ARCH "gfx906")
            endif()
        else()
            message(WARNING "rocminfo failed, using default gfx906")
            set(HIP_GPU_ARCH "gfx906")
        endif()
    else()
        message(WARNING "rocminfo not found, using default gfx906")
        set(HIP_GPU_ARCH "gfx906")
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
    # Intel DPC++ / Intel LLVM with CUDA backend (NVIDIA GPUs)
    message(STATUS "Configuring for NVIDIA GPU with CUDA backend")
    message(STATUS "CUDA GPU architecture: ${CUDA_GPU_ARCH}")
    
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
    target_compile_definitions(rambo_sycl PRIVATE SYCL_BACKEND_CUDA)
    
elseif(SYCL_BACKEND STREQUAL "HIP")
    # Intel DPC++ / Intel LLVM with HIP backend (AMD GPUs)
    message(STATUS "Configuring for AMD GPU with HIP backend")
    message(STATUS "HIP GPU architecture: ${HIP_GPU_ARCH}")
    
    target_compile_options(rambo_sycl PRIVATE 
        -fsycl
        -fsycl-targets=amdgcn-amd-amdhsa
        -Xsycl-target-backend
        --offload-arch=${HIP_GPU_ARCH}
        $<$<CONFIG:Release>:-O3>
    )
    target_link_options(rambo_sycl PRIVATE
        -fsycl
        -fsycl-targets=amdgcn-amd-amdhsa
        -Xsycl-target-backend
        --offload-arch=${HIP_GPU_ARCH}
    )
    target_compile_definitions(rambo_sycl PRIVATE SYCL_BACKEND_HIP)
    
elseif(SYCL_BACKEND STREQUAL "INTEL")
    # Intel DPC++ for Intel GPUs (different generations)
    message(STATUS "Configuring for Intel GPU")
    
    # Intel GPU architecture selection (optional)
    set(INTEL_GPU_ARCH "" CACHE STRING "Intel GPU architecture: empty for auto, or specific device")
    
    target_compile_options(rambo_sycl PRIVATE 
        -fsycl
        $<$<CONFIG:Release>:-O3>
    )
    target_link_options(rambo_sycl PRIVATE
        -fsycl
    )
    target_compile_definitions(rambo_sycl PRIVATE SYCL_BACKEND_INTEL)
    
elseif(SYCL_BACKEND STREQUAL "ADAPTIVECPP")
    # AdaptiveCpp (formerly hipSYCL) - portable across vendors
    message(STATUS "Configuring with AdaptiveCpp")
    
    # AdaptiveCpp target specification
    # Examples: "cuda:sm_89", "hip:gfx1100", "omp", "generic"
    set(ADAPTIVECPP_TARGETS "" CACHE STRING "AdaptiveCpp targets (e.g., cuda:sm_89, hip:gfx1100)")
    if(ADAPTIVECPP_TARGETS)
        message(STATUS "AdaptiveCpp targets: ${ADAPTIVECPP_TARGETS}")
    endif()
    
    find_package(AdaptiveCpp REQUIRED)
    add_sycl_to_target(TARGET rambo_sycl)
    target_compile_definitions(rambo_sycl PRIVATE SYCL_BACKEND_ADAPTIVECPP)
    
else()
    message(FATAL_ERROR "Unknown SYCL backend: ${SYCL_BACKEND}. Use CUDA, HIP, INTEL, or ADAPTIVECPP.")
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
message(STATUS "")
message(STATUS "=== SYCL Build Configuration ===")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "SYCL Backend: ${SYCL_BACKEND}")
if(SYCL_BACKEND STREQUAL "CUDA")
    message(STATUS "GPU Architecture: ${CUDA_GPU_ARCH}")
elseif(SYCL_BACKEND STREQUAL "HIP")
    message(STATUS "GPU Architecture: ${HIP_GPU_ARCH}")
endif()
message(STATUS "================================")
