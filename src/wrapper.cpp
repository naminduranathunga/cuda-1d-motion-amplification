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

void process_frame(float* h_input, float* h_output, float* d_state, int width, int height, float alpha, Metrics* metrics) {
    size_t img_size = width * height * sizeof(float);
    
    float *d_input, *d_blur, *d_sobel, *d_filtered, *d_output;
    cudaMalloc(&d_input, img_size);
    cudaMalloc(&d_blur, img_size);
    cudaMalloc(&d_sobel, img_size);
    cudaMalloc(&d_filtered, img_size);
    cudaMalloc(&d_output, img_size);

    cudaEvent_t start, stop, total_start, total_stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&total_start);
    cudaEventCreate(&total_stop);

    float milliseconds = 0;

    // 1. Host to Device
    cudaEventRecord(start);
    cudaMemcpy(d_input, h_input, img_size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    metrics->host_to_device_ms = milliseconds;

    // 2. Gaussian Blur
    cudaEventRecord(start);
    apply_gaussian_blur(d_input, d_blur, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    metrics->gaussian_blur_ms = milliseconds;

    // 3. Sobel X
    cudaEventRecord(start);
    apply_sobel_x(d_blur, d_sobel, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    metrics->sobel_x_ms = milliseconds;

    // 4. Temporal Filter
    // alpha_l and alpha_h could be parameters, hardcoding for now or adding to process_frame
    float alpha_l = 0.05f; 
    float alpha_h = 0.2f;
    cudaEventRecord(start);
    apply_temporal_filter(d_sobel, d_state, d_filtered, width, height, alpha_l, alpha_h);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    metrics->temporal_filter_ms = milliseconds;

    // 5. Amplification
    cudaEventRecord(start);
    apply_amplify(d_input, d_filtered, d_output, width, height, alpha);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    metrics->amplification_ms = milliseconds;

    // 6. Device to Host
    cudaEventRecord(start);
    cudaMemcpy(h_output, d_output, img_size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    metrics->device_to_host_ms = milliseconds;

    // Cleanup local temp buffers
    cudaFree(d_input);
    cudaFree(d_blur);
    cudaFree(d_sobel);
    cudaFree(d_filtered);
    cudaFree(d_output);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(total_start);
    cudaEventDestroy(total_stop);
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
