cmake_minimum_required(VERSION 3.16)

project(RamboBase CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable optimizations
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

# Header-only implementation - only main.cpp needs to be compiled
add_executable(rambo_base main.cpp)

# Link math library
target_link_libraries(rambo_base PRIVATE m)
