cmake_minimum_required(VERSION 3.14)
project(rambo_omp CXX)

set(CMAKE_CXX_COMPILER clang++)

# Detect LLVM version
execute_process(
    COMMAND clang++ --version
    OUTPUT_VARIABLE CLANG_VERSION_RAW
)

message(STATUS "Clang version: ${CLANG_VERSION_RAW}")

# Add OpenMP flags
set(OPENMP_FLAGS
    -fopenmp
    -fopenmp-targets=nvptx64-nvidia-cuda
    -Xopenmp-target=nvptx64-nvidia-cuda -march=sm_89
)

add_executable(rambo_omp
    main.cpp
    integrator.cpp
    rambo_omp.cpp
    rng.cpp
)

target_compile_options(rambo_omp PRIVATE ${OPENMP_FLAGS})
target_link_options(rambo_omp PRIVATE ${OPENMP_FLAGS})

