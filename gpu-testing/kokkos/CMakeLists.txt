cmake_minimum_required(VERSION 3.16)

project("RamboKokkos" CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Load Kokkos module
find_package(EnvModules REQUIRED)
env_module(load kokkos/dev)

find_package(Kokkos REQUIRED)

add_executable(rambo_kokkos
    main.cpp
)
target_link_libraries(rambo_kokkos Kokkos::kokkos)
