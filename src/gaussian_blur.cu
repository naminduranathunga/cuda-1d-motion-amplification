#include "motion_amp.h"
#include <device_launch_parameters.h>

__global__ void gaussian_blur_kernel(float* input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // 5x5 Gaussian Kernel (approximate)
    float kernel[5][5] = {
        {1, 4, 7, 4, 1},
        {4, 16, 26, 16, 4},
        {7, 26, 41, 26, 7},
        {4, 16, 26, 16, 4},
        {1, 4, 7, 4, 1}
    };
    float kernel_sum = 273.0f;

    float sum = 0.0f;
    for (int i = -2; i <= 2; ++i) {
        for (int j = -2; j <= 2; ++j) {
            int nx = min(max(x + j, 0), width - 1);
            int ny = min(max(y + i, 0), height - 1);
            sum += input[ny * width + nx] * kernel[i + 2][j + 2];
        }
    }

    output[y * width + x] = sum / kernel_sum;
}

extern "C" void apply_gaussian_blur(float* d_input, float* d_output, int width, int height) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    gaussian_blur_kernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);
}
