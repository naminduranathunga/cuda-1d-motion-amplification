#ifndef MOTION_AMP_H
#define MOTION_AMP_H

#include <cuda_runtime.h>

extern "C" {

struct GPUContext {
    float *d_input;
    float *d_blur;
    float *d_sobel;
    float *d_filtered;
    float *d_output;
    float *d_state;
    int width;
    int height;
};

struct Metrics {
    float host_to_device_ms;
    float gaussian_blur_ms;
    float sobel_x_ms;
    float temporal_filter_ms;
    float amplification_ms;
    float device_to_host_ms;
};

// Memory management
void* allocate_device_memory(size_t size);
void free_device_memory(void* ptr);

GPUContext* initGPU(int width, int height, float sigma);
void cleanupGPU(GPUContext* context);

// Kernels wrappers
void set_gaussian_weights(float sigma);
void init_blur_texture(float* d_input, int width, int height);
void cleanup_blur_texture();
void apply_gaussian_blur(float* d_input, float* d_output, int width, int height, float sigma);
void apply_gaussian_blur_tex2d(float* d_input, float* d_output, int width, int height, float sigma);
void apply_sobel_x(float* d_input, float* d_output, int width, int height);
void apply_temporal_filter(float* d_input, float* d_state, float* d_output, int width, int height, float low_cutoff, float high_cutoff);
void apply_amplify(float* d_original, float* d_filtered, float* d_output, int width, int height, float alpha);
void compute_roi_histogram(float* d_input, int* d_histogram, int x1, int y1, int x2, int y2, int width, int height);

// Main processing function for Python to call
void process_frame(float* h_input, float* h_output, GPUContext* context, float alpha, float alpha_l, float alpha_h, Metrics* metrics);
void get_histogram(float* h_input, int* h_histogram, int x1, int y1, int x2, int y2, int width, int height);

}

#endif // MOTION_AMP_H
