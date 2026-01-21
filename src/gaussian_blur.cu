#include "motion_amp.h"
#include <device_launch_parameters.h>

__global__ void gaussian_blur_kernel(float* input, float* output, int width, int height, float sigma) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Use a 5x5 dynamic kernel based on sigma
    float sum = 0.0f;
    float weight_sum = 0.0f;
    float s2 = 2.0f * sigma * sigma;

    for (int i = -2; i <= 2; ++i) {
        for (int j = -2; j <= 2; ++j) {
            int nx = min(max(x + j, 0), width - 1);
            int ny = min(max(y + i, 0), height - 1);
            
            float dist_sq = (float)(i * i + j * j);
            float w = expf(-dist_sq / s2);
            
            sum += input[ny * width + nx] * w;
            weight_sum += w;
        }
    }

    output[y * width + x] = sum / weight_sum;
}

extern "C" void apply_gaussian_blur(float* d_input, float* d_output, int width, int height, float sigma) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    gaussian_blur_kernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, sigma);
}
