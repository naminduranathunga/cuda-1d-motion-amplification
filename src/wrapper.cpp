#include "motion_amp.h"
#include <iostream>

extern "C" {

void* allocate_device_memory(size_t size) {
    void* d_ptr;
    cudaError_t err = cudaMalloc(&d_ptr, size);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        return nullptr;
    }
    // Initialize with zeros for state memory
    cudaMemset(d_ptr, 0, size);
    return d_ptr;
}

void free_device_memory(void* ptr) {
    if (ptr) {
        cudaFree(ptr);
    }
}

GPUContext* initGPU(int width, int height, float sigma) {
    GPUContext* ctx = new GPUContext();
    ctx->width = width;
    ctx->height = height;
    size_t img_size = width * height * sizeof(float);
    size_t state_size = 2 * img_size;

    cudaMalloc(&ctx->d_input, img_size);
    cudaMalloc(&ctx->d_blur, img_size);
    cudaMalloc(&ctx->d_sobel, img_size);
    cudaMalloc(&ctx->d_filtered, img_size);
    cudaMalloc(&ctx->d_output, img_size);
    cudaMalloc(&ctx->d_state, state_size);

    cudaMemset(ctx->d_state, 0, state_size);
    
    set_gaussian_weights(sigma);
    init_blur_texture(ctx->d_input, width, height);

    return ctx;
}

void cleanupGPU(GPUContext* ctx) {
    if (ctx) {
        cleanup_blur_texture();
        cudaFree(ctx->d_input);
        cudaFree(ctx->d_blur);
        cudaFree(ctx->d_sobel);
        cudaFree(ctx->d_filtered);
        cudaFree(ctx->d_output);
        cudaFree(ctx->d_state);
        delete ctx;
    }
}

void process_frame(float* h_input, float* h_output, GPUContext* ctx, float alpha, float alpha_l, float alpha_h, Metrics* metrics) {
    int width = ctx->width;
    int height = ctx->height;
    size_t img_size = width * height * sizeof(float);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float milliseconds = 0;

    // 1. Host to Device
    cudaEventRecord(start);
    cudaMemcpy(ctx->d_input, h_input, img_size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    metrics->host_to_device_ms = milliseconds;

    // 2. Gaussian Blur
    cudaEventRecord(start);
    apply_gaussian_blur_tex2d(ctx->d_input, ctx->d_blur, width, height, 0.0f); // sigma is pre-calculated
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    metrics->gaussian_blur_ms = milliseconds;

    // 3. Sobel X
    cudaEventRecord(start);
    apply_sobel_x(ctx->d_blur, ctx->d_sobel, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    metrics->sobel_x_ms = milliseconds;

    // 4. Temporal Filter
    cudaEventRecord(start);
    apply_temporal_filter(ctx->d_sobel, ctx->d_state, ctx->d_filtered, width, height, alpha_l, alpha_h);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    metrics->temporal_filter_ms = milliseconds;

    // 5. Amplification
    cudaEventRecord(start);
    apply_amplify(ctx->d_input, ctx->d_filtered, ctx->d_output, width, height, alpha);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    metrics->amplification_ms = milliseconds;

    // 6. Device to Host
    cudaEventRecord(start);
    cudaMemcpy(h_output, ctx->d_output, img_size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    metrics->device_to_host_ms = milliseconds;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void get_histogram(float* h_input, int* h_histogram, int x1, int y1, int x2, int y2, int width, int height) {
    size_t img_size = width * height * sizeof(float);
    size_t hist_size = 256 * sizeof(int);

    float *d_input;
    int *d_histogram;
    cudaMalloc(&d_input, img_size);
    cudaMalloc(&d_histogram, hist_size);

    cudaMemcpy(d_input, h_input, img_size, cudaMemcpyHostToDevice);
    
    compute_roi_histogram(d_input, d_histogram, x1, y1, x2, y2, width, height);

    cudaMemcpy(h_histogram, d_histogram, hist_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_histogram);
}

}
