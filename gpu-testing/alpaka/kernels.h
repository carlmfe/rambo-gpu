#ifndef KERNELS_H
#define KERNELS_H

#include <alpaka/alpaka.hpp>

// Kernel function declarations
template<typename T>
void myKernel(alpaka::Block<T> const& block, alpaka::Grid<T> const& grid);

template<typename T>
void anotherKernel(alpaka::Block<T> const& block, alpaka::Grid<T> const& grid);

#endif // KERNELS_H