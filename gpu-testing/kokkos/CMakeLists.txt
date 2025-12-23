cmake_minimum_required(VERSION 3.18)

project(RamboKokkos CXX)

# =============================================================================
# Find Kokkos FIRST (it may set required compiler settings)
# =============================================================================
# Kokkos can be found via:
#   1. CMAKE_PREFIX_PATH pointing to Kokkos installation
#   2. Kokkos_ROOT environment variable
#   3. System-wide installation
#
# Example: cmake -DKokkos_ROOT=/path/to/kokkos/install ..
# =============================================================================
find_package(Kokkos REQUIRED)

# Print Kokkos configuration
message(STATUS "Kokkos version: ${Kokkos_VERSION}")
message(STATUS "Kokkos devices: ${Kokkos_DEVICES}")

# =============================================================================
# Build Type
# =============================================================================
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# =============================================================================
# Build executable
# =============================================================================
add_executable(rambo_kokkos main.cpp)
target_link_libraries(rambo_kokkos Kokkos::kokkos)

# =============================================================================
# Platform-specific linking
# =============================================================================
if(UNIX AND NOT APPLE)
    target_link_libraries(rambo_kokkos m)
endif()

message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")# =============================================================================
# Print configuration
# =============================================================================
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
