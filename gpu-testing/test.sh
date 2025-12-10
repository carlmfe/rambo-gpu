#!/bin/bash

script_dir=$(dirname "$(realpath "$0")")

omp_path=$script_dir/omp/build/rambo_omp
kokkos_cuda_path=$script_dir/kokkos/build-cuda/rambo_kokkos
kokkos_serial_path=$script_dir/kokkos/build-serial/rambo_kokkos
cuda_path=$script_dir/cuda/rambo_cuda

n_events=100000000

echo "Running Kokkos Serial version..."
time $kokkos_serial_path $n_events
echo ""
sleep 1
echo "Running OpenMP version..."
time $omp_path $n_events
echo ""
sleep 1
echo "Running Kokkos CUDA version..."
time $kokkos_cuda_path $n_events
echo ""
sleep 1
echo "Running CUDA version..."
time $cuda_path $n_events
