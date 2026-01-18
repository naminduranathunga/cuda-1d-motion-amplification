#include "motion_amp.h"
#include <device_launch_parameters.h>

__global__ void amplify_kernel(float* original, float* filtered, float* output, int width, int height, float alpha) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    
    // Result = original + alpha * filtered_gradient
    float result = original[idx] + alpha * filtered[idx];

    // Clamp to [0, 1] for image
    output[idx] = fminf(fmaxf(result, 0.0f), 1.0f);
}

extern "C" void apply_amplify(float* d_original, float* d_filtered, float* d_output, int width, int height, float alpha) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    amplify_kernel<<<gridSize, blockSize>>>(d_original, d_filtered, d_output, width, height, alpha);
}
