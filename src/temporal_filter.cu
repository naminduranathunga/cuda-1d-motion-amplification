#include "motion_amp.h"
#include <device_launch_parameters.h>

// Simple IIR Bandpass: difference of two low-pass filters
// state[0]: low-pass with high cutoff (alpha_h)
// state[1]: low-pass with low cutoff (alpha_l)
// state array size should be 2 * width * height

__global__ void temporal_filter_kernel(float* input, float* state, float* output, int width, int height, float alpha_l, float alpha_h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int state_idx_l = idx;
    int state_idx_h = idx + (width * height);

    float val = input[idx];

    // Update low-pass filters
    state[state_idx_l] = (1.0f - alpha_l) * state[state_idx_l] + alpha_l * val;
    state[state_idx_h] = (1.0f - alpha_h) * state[state_idx_h] + alpha_h * val;

    // Bandpass output
    output[idx] = state[state_idx_h] - state[state_idx_l];
}

extern "C" void apply_temporal_filter(float* d_input, float* d_state, float* d_output, int width, int height, float alpha_l, float alpha_h) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    temporal_filter_kernel<<<gridSize, blockSize>>>(d_input, d_state, d_output, width, height, alpha_l, alpha_h);
}
