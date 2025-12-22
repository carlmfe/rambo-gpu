cmake_minimum_required(VERSION 3.25)

project("RamboAlpaka" CXX)

# =============================================================================
# Backend Selection
# =============================================================================
# Use: cmake -DALPAKA_BACKEND=CUDA ..   (default)
#      cmake -DALPAKA_BACKEND=CPU ..    (CPU serial)
#      cmake -DALPAKA_BACKEND=OMP ..    (OpenMP threads)
# =============================================================================

set(ALPAKA_BACKEND "CUDA" CACHE STRING "Alpaka backend: CUDA, CPU, OMP")
set_property(CACHE ALPAKA_BACKEND PROPERTY STRINGS CUDA CPU OMP)

message(STATUS "Building with backend: ${ALPAKA_BACKEND}")

# Load Alpaka module
find_package(EnvModules REQUIRED)
env_module(load alpaka/2.0.0_cuda)

set(CMAKE_CXX_STANDARD 20)
set(alpaka_RELOCATABLE_DEVICE_CODE ON)

# Configure backend-specific settings
if(ALPAKA_BACKEND STREQUAL "CUDA")
    set(alpaka_ACC_GPU_CUDA_ENABLE ON)
    set(CMAKE_CUDA_ARCHITECTURES 89)
    add_compile_definitions(ALPAKA_USE_CUDA)
elseif(ALPAKA_BACKEND STREQUAL "CPU")
    set(alpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE ON)
    add_compile_definitions(ALPAKA_USE_CPU_SERIAL)
elseif(ALPAKA_BACKEND STREQUAL "OMP")
    set(alpaka_ACC_CPU_B_OMP2_T_SEQ_ENABLE ON)
    find_package(OpenMP REQUIRED)
    add_compile_definitions(ALPAKA_USE_CPU_OMP)
endif()

find_package(alpaka REQUIRED)

alpaka_add_executable(rambo_alpaka
    main.cpp
)
target_link_libraries(rambo_alpaka PUBLIC alpaka::alpaka)

if(ALPAKA_BACKEND STREQUAL "OMP")
    target_link_libraries(rambo_alpaka PUBLIC OpenMP::OpenMP_CXX)
endif()

