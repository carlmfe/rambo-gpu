cmake_minimum_required(VERSION 3.18)

project("RamboCUDA" CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Auto-detect CUDA architecture or set manually
# For RTX 2000 Ada: 89, for RTX 30xx: 86, for RTX 20xx: 75
set(CMAKE_CUDA_ARCHITECTURES 89)

# Enable separable compilation if needed
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

add_executable(rambo_cuda
    main.cu
)

# Optimization flags
target_compile_options(rambo_cuda PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)
