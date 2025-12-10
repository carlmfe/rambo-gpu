#pragma once

#include <alpaka/alpaka.hpp>

using Idx = std::size_t;
using Dim = alpaka::DimInt<1u>;

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
using Acc = alpaka::AccGpuUniformCudaHipRt<Dim, Idx>;
#elif defined(ALPAKA_ACC_CPU_B_OMP2_THREADS_ENABLED)
using Acc = alpaka::AccCpuOmp2Threads<Dim, Idx>;
#else
using Acc = alpaka::AccCpuSerial<Dim, Idx>;
#endif

using Dev = alpaka::Dev<Acc>;
using Queue = alpaka::Queue<Acc, alpaka::Blocking>;

void ramboAlpakaMain(Dev const& device, Queue& queue);
