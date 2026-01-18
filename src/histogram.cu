#include "motion_amp.h"
#include <device_launch_parameters.h>

__global__ void histogram_kernel(float* input, int* histogram, int x1, int y1, int x2, int y2, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + x1;
    int y = blockIdx.y * blockDim.y + threadIdx.y + y1;

    if (x >= x2 || y >= y2 || x >= width || y >= height) return;

    float val = input[y * width + x];
    int bin = (int)(val * 255.0f);
    bin = min(max(bin, 0), 255);

    atomicAdd(&histogram[bin], 1);
}

extern "C" void compute_roi_histogram(float* d_input, int* d_histogram, int x1, int y1, int x2, int y2, int width, int height) {
    // Reset histogram to zero
    cudaMemset(d_histogram, 0, 256 * sizeof(int));

    int roi_w = x2 - x1;
    int roi_h = y2 - y1;

    if (roi_w <= 0 || roi_h <= 0) return;

    dim3 blockSize(16, 16);
    dim3 gridSize((roi_w + blockSize.x - 1) / blockSize.x, (roi_h + blockSize.y - 1) / blockSize.y);
    
    histogram_kernel<<<gridSize, blockSize>>>(d_input, d_histogram, x1, y1, x2, y2, width, height);
}
