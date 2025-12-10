cmake_minimum_required(VERSION 3.10)

project(gpu-testing-base)

add_executable(rambo_base
  main.cpp
  integrator.cpp
  rambo.cpp
)
