cmake_minimum_required(VERSION 3.10)
project(RamboAlpaka LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(alpaka 1.3 REQUIRED CONFIG)

option(RAMBO_USE_CUDA "Use CUDA backend" OFF)
option(RAMBO_USE_OMP  "Use OpenMP CPU backend" OFF)

set(SOURCES
    main.cpp
    rambo_alpaka.cpp
    integrator.cpp
    kernels.cpp
)

add_executable(rambo_alpaka ${SOURCES})
target_link_libraries(rambo_alpaka PRIVATE alpaka::alpaka)
target_include_directories(rambo_alpaka PRIVATE ${CMAKE_SOURCE_DIR}/include)

if(RAMBO_USE_CUDA)
    enable_language(CUDA) # enable CUDA compiler support in CMake
    target_compile_definitions(rambo_alpaka PRIVATE ALPAKA_ACC_GPU_CUDA_ENABLED=1)
    message(STATUS "Building RamboAlpaka with ALPAKA CUDA backend")
elseif(RAMBO_USE_OMP)
    find_package(OpenMP)
    target_compile_definitions(rambo_alpaka PRIVATE ALPAKA_ACC_CPU_B_OMP2_THREADS_ENABLED=1)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(rambo_alpaka PRIVATE OpenMP::OpenMP_CXX)
    endif()
    message(STATUS "Building RamboAlpaka with Alpaka OpenMP backend")
else()
    target_compile_definitions(rambo_alpaka PRIVATE ALPAKA_ACC_CPU_B_SERIAL_ENABLED=1)
    message(STATUS "Building RamboAlpaka with Alpaka serial CPU backend")
endif()
