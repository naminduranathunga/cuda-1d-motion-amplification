#ifndef MOTION_AMP_H
#define MOTION_AMP_H

#include <cuda_runtime.h>

#define MAX_STREAM_BUFFERS 5

extern "C" {

struct GPUContext {
    float *d_input;
    float *d_blur;
    float *d_temp_blur;
    float *d_sobel;
    float *d_filtered;
    float *d_output;
    float *d_state;
    float *d_max_mag;
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
    float max_magnitude;
};

// Per-buffer resources for the circular stream pipeline
struct StreamBuffer {
    // Device memory (per-buffer, so multiple frames can be in-flight)
    float *d_input;
    float *d_blur;
    float *d_temp_blur;
    float *d_sobel;
    float *d_filtered;
    float *d_output;
    float *d_max_mag;
    // Pinned host memory for async transfers
    float *h_input_pinned;
    float *h_output_pinned;
    // CUDA stream and events
    cudaStream_t stream;
    cudaEvent_t event_h2d_done;
    cudaEvent_t event_kernels_done;
    cudaEvent_t event_d2h_done;
    // Per-stage timing events: t0=before H2D, t1=after H2D, t2=after blur,
    // t3=after sobel, t4=after temporal, t5=after amplify, t6=after D2H
    cudaEvent_t evt_t0;
    cudaEvent_t evt_t1;
    cudaEvent_t evt_t2;
    cudaEvent_t evt_t3;
    cudaEvent_t evt_t4;
    cudaEvent_t evt_t5;
    cudaEvent_t evt_t6;
    // Texture object for this buffer's d_input
    cudaTextureObject_t tex_input;
};

struct StreamPipeline {
    StreamBuffer buffers[MAX_STREAM_BUFFERS];
    float *d_state;           // Shared temporal filter state (serial across frames)
    int width;
    int height;
    int num_buffers;
    // Event to serialize kernel execution across streams (temporal dependency)
    cudaEvent_t event_prev_kernel_done;
};

struct PipelineMetrics {
    float total_wall_time_ms;
    float avg_host_to_device_ms;
    float avg_gaussian_blur_ms;
    float avg_sobel_x_ms;
    float avg_temporal_filter_ms;
    float avg_amplification_ms;
    float avg_device_to_host_ms;
    float avg_frame_time_ms;
    float max_magnitude;
    int total_frames;
};

// Memory management
void* allocate_device_memory(size_t size);
void free_device_memory(void* ptr);

GPUContext* initGPU(int width, int height, float sigma);
void cleanupGPU(GPUContext* context);

// Kernels wrappers (with stream parameter)
void set_gaussian_weights(float sigma);
void init_blur_texture(float* d_input, int width, int height);
void cleanup_blur_texture();
void init_blur_texture_for_buffer(float* d_input, int width, int height, cudaTextureObject_t* tex_out);
void cleanup_blur_texture_for_buffer(cudaTextureObject_t* tex);
void apply_gaussian_blur(float* d_input, float* d_output, int width, int height, float sigma);
void apply_gaussian_blur_tex2d(float* d_input, float* d_temp, float* d_output, int width, int height);
void apply_gaussian_blur_tex2d_stream(cudaTextureObject_t tex, float* d_temp, float* d_output, int width, int height, cudaStream_t stream);
void apply_sobel_x(float* d_input, float* d_output, int width, int height);
void apply_sobel_x_stream(float* d_input, float* d_output, int width, int height, cudaStream_t stream);
void apply_temporal_filter(float* d_input, float* d_state, float* d_output, int width, int height, float low_cutoff, float high_cutoff);
void apply_temporal_filter_stream(float* d_input, float* d_state, float* d_output, int width, int height, float low_cutoff, float high_cutoff, cudaStream_t stream);
void apply_amplify(float* d_original, float* d_filtered, float* d_output, int width, int height, float alpha, float threshold, float* d_max_mag);
void apply_amplify_stream(float* d_original, float* d_filtered, float* d_output, int width, int height, float alpha, float threshold, float* d_max_mag, cudaStream_t stream);
void compute_roi_histogram(float* d_input, int* d_histogram, int x1, int y1, int x2, int y2, int width, int height);

// Original synchronous processing (kept for backward compatibility)
void process_frame(float* h_input, float* h_output, GPUContext* context, float alpha, float alpha_l, float alpha_h, float threshold, Metrics* metrics);
void get_histogram(float* h_input, int* h_histogram, int x1, int y1, int x2, int y2, int width, int height);

// Stream pipeline API
StreamPipeline* initStreamPipeline(int width, int height, float sigma, int num_buffers);
void cleanupStreamPipeline(StreamPipeline* pipeline);
void submit_frame(StreamPipeline* pipeline, float* h_input, int frame_idx, float alpha, float alpha_l, float alpha_h, float threshold);
void collect_frame(StreamPipeline* pipeline, float* h_output, int frame_idx, Metrics* metrics);

}

#endif // MOTION_AMP_H
