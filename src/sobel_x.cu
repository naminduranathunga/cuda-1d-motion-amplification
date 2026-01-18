#include "motion_amp.h"
#include <device_launch_parameters.h>

__global__ void sobel_x_kernel(float* input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Sobel X kernel
    // -1 0 1
    // -2 0 2
    // -1 0 1

    float gx = 0.0f;
    
    // Boundary check for simplicity in kernel
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        gx += -1.0f * input[(y - 1) * width + (x - 1)];
        gx +=  1.0f * input[(y - 1) * width + (x + 1)];
        gx += -2.0f * input[(y) * width + (x - 1)];
        gx +=  2.0f * input[(y) * width + (x + 1)];
        gx += -1.0f * input[(y + 1) * width + (x - 1)];
        gx +=  1.0f * input[(y + 1) * width + (x + 1)];
    }

    output[y * width + x] = gx;
}

extern "C" void apply_sobel_x(float* d_input, float* d_output, int width, int height) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    sobel_x_kernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);
}
