#include "motion_amp.h"
#include <device_launch_parameters.h>

__device__ void atomicMaxFloat(float* address, float val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
            __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
}

__global__ void amplify_kernel(float* original, float* filtered, float* output, int width, int height, float alpha, float threshold, float* d_max_mag) {
    __shared__ float block_max;
    if (threadIdx.x == 0 && threadIdx.y == 0) block_max = 0.0f;
    __syncthreads();

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float mag = 0.0f;
    float filtered_val = 0.0f;

    if (x < width && y < height) {
        int idx = y * width + x;
        filtered_val = filtered[idx];
        mag = fabsf(filtered_val);
        
        // Block-level atomic max (shared memory has much less contention than global)
        atomicMaxFloat(&block_max, mag);

        if (mag > threshold) {
            filtered_val = 0.0f;
        }

        float result = original[idx] + alpha * filtered_val;
        output[idx] = fminf(fmaxf(result, 0.0f), 1.0f);
    }

    __syncthreads();
    // Only one thread per block updates global max
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicMaxFloat(d_max_mag, block_max);
    }
}

extern "C" void apply_amplify(float* d_original, float* d_filtered, float* d_output, int width, int height, float alpha, float threshold, float* d_max_mag) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    amplify_kernel<<<gridSize, blockSize>>>(d_original, d_filtered, d_output, width, height, alpha, threshold, d_max_mag);
}

extern "C" void apply_amplify_stream(float* d_original, float* d_filtered, float* d_output, int width, int height, float alpha, float threshold, float* d_max_mag, cudaStream_t stream) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    amplify_kernel<<<gridSize, blockSize, 0, stream>>>(d_original, d_filtered, d_output, width, height, alpha, threshold, d_max_mag);
}
