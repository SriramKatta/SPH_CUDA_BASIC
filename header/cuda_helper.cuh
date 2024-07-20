#pragma once

#include <iostream>
#include <cassert>

#define HOST_DEVICE_FUNC __host__ __device__

#define GRID_STRIDE(index, stride)                     \
    int index = blockIdx.x * blockDim.x + threadIdx.x; \
    int stride = gridDim.x * blockDim.x;

#define NBD_LOOP(CODE)                       \
    for (int ix = -1; ix <= 1; ++ix)         \
        for (int iy = -1; iy <= 1; ++iy)     \
            for (int iz = -1; iz <= 1; ++iz) \
            {                                \
                CODE;                        \
            }


inline cudaError_t checkCuda(cudaError_t result, int n)
{
    if (result != cudaSuccess)
    {
        fprintf(stderr, "CUDA Runtime Error: %s\n Kernel No: %d ", cudaGetErrorString(result), n);
        assert(result == cudaSuccess);
    }
    return result;
}
