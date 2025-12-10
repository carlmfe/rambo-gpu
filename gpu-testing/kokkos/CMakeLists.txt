cmake_minimum_required(VERSION 3.10)
project(rambo_kokkos)

find_package(Kokkos REQUIRED)
add_executable(rambo_kokkos 
    main.cpp 
    rambo_kokkos.cpp
    integrator.cpp
)
target_link_libraries(rambo_kokkos Kokkos::kokkos)
