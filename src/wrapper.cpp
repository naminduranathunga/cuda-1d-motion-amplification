#include "motion_amp.h"
#include <iostream>
#include <cstring>

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
    cudaMalloc(&ctx->d_temp_blur, img_size);
    cudaMalloc(&ctx->d_sobel, img_size);
    cudaMalloc(&ctx->d_filtered, img_size);
    cudaMalloc(&ctx->d_output, img_size);
    cudaMalloc(&ctx->d_state, state_size);
    cudaMalloc(&ctx->d_max_mag, sizeof(float));

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
        cudaFree(ctx->d_temp_blur);
        cudaFree(ctx->d_sobel);
        cudaFree(ctx->d_filtered);
        cudaFree(ctx->d_output);
        cudaFree(ctx->d_state);
        cudaFree(ctx->d_max_mag);
        delete ctx;
    }
}

void process_frame(float* h_input, float* h_output, GPUContext* ctx, float alpha, float alpha_l, float alpha_h, float threshold, Metrics* metrics) {
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

    // 2. Gaussian Blur (Separable)
    cudaEventRecord(start);
    apply_gaussian_blur_tex2d(ctx->d_input, ctx->d_temp_blur, ctx->d_blur, width, height); 
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
    cudaMemset(ctx->d_max_mag, 0, sizeof(float));
    apply_amplify(ctx->d_input, ctx->d_filtered, ctx->d_output, width, height, alpha, threshold, ctx->d_max_mag);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    metrics->amplification_ms = milliseconds;

    // Retrieve max magnitude
    cudaMemcpy(&metrics->max_magnitude, ctx->d_max_mag, sizeof(float), cudaMemcpyDeviceToHost);

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

// ============================================================
// Stream Pipeline Implementation (Circular Buffer)
// ============================================================

StreamPipeline* initStreamPipeline(int width, int height, float sigma, int num_buffers) {
    if (num_buffers < 1) num_buffers = 1;
    if (num_buffers > MAX_STREAM_BUFFERS) num_buffers = MAX_STREAM_BUFFERS;

    StreamPipeline* pipeline = new StreamPipeline();
    pipeline->width = width;
    pipeline->height = height;
    pipeline->num_buffers = num_buffers;

    size_t img_size = width * height * sizeof(float);
    size_t state_size = 2 * img_size;

    // Shared temporal filter state
    cudaMalloc(&pipeline->d_state, state_size);
    cudaMemset(pipeline->d_state, 0, state_size);

    // Event for serializing kernel execution across frames
    cudaEventCreate(&pipeline->event_prev_kernel_done);

    // Set gaussian weights (constant memory, shared across all streams)
    set_gaussian_weights(sigma);

    // Initialize each buffer slot
    for (int i = 0; i < num_buffers; i++) {
        StreamBuffer& buf = pipeline->buffers[i];

        // Device memory
        cudaMalloc(&buf.d_input, img_size);
        cudaMalloc(&buf.d_blur, img_size);
        cudaMalloc(&buf.d_temp_blur, img_size);
        cudaMalloc(&buf.d_sobel, img_size);
        cudaMalloc(&buf.d_filtered, img_size);
        cudaMalloc(&buf.d_output, img_size);
        cudaMalloc(&buf.d_max_mag, sizeof(float));

        // Pinned host memory for async transfers
        cudaMallocHost(&buf.h_input_pinned, img_size);
        cudaMallocHost(&buf.h_output_pinned, img_size);

        // CUDA stream
        cudaStreamCreate(&buf.stream);

        // Events (disable timing on internal events for lower overhead)
        cudaEventCreateWithFlags(&buf.event_h2d_done, cudaEventDisableTiming);
        cudaEventCreate(&buf.event_kernels_done);
        cudaEventCreateWithFlags(&buf.event_d2h_done, cudaEventDisableTiming);

        // Per-stage timing events (all with timing enabled)
        cudaEventCreate(&buf.evt_t0);
        cudaEventCreate(&buf.evt_t1);
        cudaEventCreate(&buf.evt_t2);
        cudaEventCreate(&buf.evt_t3);
        cudaEventCreate(&buf.evt_t4);
        cudaEventCreate(&buf.evt_t5);
        cudaEventCreate(&buf.evt_t6);

        // Texture object for this buffer's d_input
        buf.tex_input = 0;
        init_blur_texture_for_buffer(buf.d_input, width, height, &buf.tex_input);
    }

    std::cout << "Stream pipeline initialized: " << num_buffers << " buffers, "
              << width << "x" << height << std::endl;

    return pipeline;
}

void cleanupStreamPipeline(StreamPipeline* pipeline) {
    if (!pipeline) return;

    // Synchronize all streams before cleanup
    for (int i = 0; i < pipeline->num_buffers; i++) {
        cudaStreamSynchronize(pipeline->buffers[i].stream);
    }

    for (int i = 0; i < pipeline->num_buffers; i++) {
        StreamBuffer& buf = pipeline->buffers[i];

        cleanup_blur_texture_for_buffer(&buf.tex_input);

        cudaFree(buf.d_input);
        cudaFree(buf.d_blur);
        cudaFree(buf.d_temp_blur);
        cudaFree(buf.d_sobel);
        cudaFree(buf.d_filtered);
        cudaFree(buf.d_output);
        cudaFree(buf.d_max_mag);

        cudaFreeHost(buf.h_input_pinned);
        cudaFreeHost(buf.h_output_pinned);

        cudaStreamDestroy(buf.stream);
        cudaEventDestroy(buf.event_h2d_done);
        cudaEventDestroy(buf.event_kernels_done);
        cudaEventDestroy(buf.event_d2h_done);
        cudaEventDestroy(buf.evt_t0);
        cudaEventDestroy(buf.evt_t1);
        cudaEventDestroy(buf.evt_t2);
        cudaEventDestroy(buf.evt_t3);
        cudaEventDestroy(buf.evt_t4);
        cudaEventDestroy(buf.evt_t5);
        cudaEventDestroy(buf.evt_t6);
    }

    cudaFree(pipeline->d_state);
    cudaEventDestroy(pipeline->event_prev_kernel_done);

    delete pipeline;
    std::cout << "Stream pipeline cleaned up." << std::endl;
}

void submit_frame(StreamPipeline* pipeline, float* h_input, int frame_idx,
                  float alpha, float alpha_l, float alpha_h, float threshold) {
    int buf_idx = frame_idx % pipeline->num_buffers;
    StreamBuffer& buf = pipeline->buffers[buf_idx];
    int width = pipeline->width;
    int height = pipeline->height;
    size_t img_size = width * height * sizeof(float);

    // If this buffer was previously used, wait for its D2H to complete
    // before we overwrite its buffers. For the first round of frames (<num_buffers),
    // the events haven't been recorded yet, so we skip.
    if (frame_idx >= pipeline->num_buffers) {
        cudaEventSynchronize(buf.event_d2h_done);
    }

    // Copy input data to pinned host memory
    memcpy(buf.h_input_pinned, h_input, img_size);

    // T0: before H2D
    cudaEventRecord(buf.evt_t0, buf.stream);

    // Async H2D transfer on this buffer's stream
    cudaMemcpyAsync(buf.d_input, buf.h_input_pinned, img_size, cudaMemcpyHostToDevice, buf.stream);
    cudaEventRecord(buf.event_h2d_done, buf.stream);

    // T1: after H2D
    cudaEventRecord(buf.evt_t1, buf.stream);

    // Serialize kernel execution: wait for previous frame's kernels
    // This ensures d_state consistency (temporal filter is frame-order-dependent)
    if (frame_idx > 0) {
        cudaStreamWaitEvent(buf.stream, pipeline->event_prev_kernel_done, 0);
    }

    // Launch kernels on this buffer's stream
    // 1. Gaussian Blur
    apply_gaussian_blur_tex2d_stream(buf.tex_input, buf.d_temp_blur, buf.d_blur, width, height, buf.stream);

    // T2: after blur
    cudaEventRecord(buf.evt_t2, buf.stream);

    // 2. Sobel X
    apply_sobel_x_stream(buf.d_blur, buf.d_sobel, width, height, buf.stream);

    // T3: after sobel
    cudaEventRecord(buf.evt_t3, buf.stream);

    // 3. Temporal Filter (uses shared d_state)
    apply_temporal_filter_stream(buf.d_sobel, pipeline->d_state, buf.d_filtered, width, height, alpha_l, alpha_h, buf.stream);

    // T4: after temporal filter
    cudaEventRecord(buf.evt_t4, buf.stream);

    // 4. Amplification
    cudaMemsetAsync(buf.d_max_mag, 0, sizeof(float), buf.stream);
    apply_amplify_stream(buf.d_input, buf.d_filtered, buf.d_output, width, height, alpha, threshold, buf.d_max_mag, buf.stream);

    // T5: after amplification
    cudaEventRecord(buf.evt_t5, buf.stream);

    // Record kernel completion event (for serializing next frame's kernels)
    cudaEventRecord(pipeline->event_prev_kernel_done, buf.stream);
    cudaEventRecord(buf.event_kernels_done, buf.stream);

    // Async D2H transfer
    cudaMemcpyAsync(buf.h_output_pinned, buf.d_output, img_size, cudaMemcpyDeviceToHost, buf.stream);

    // T6: after D2H
    cudaEventRecord(buf.evt_t6, buf.stream);

    // Record D2H completion
    cudaEventRecord(buf.event_d2h_done, buf.stream);
}

void collect_frame(StreamPipeline* pipeline, float* h_output, int frame_idx, Metrics* metrics) {
    int buf_idx = frame_idx % pipeline->num_buffers;
    StreamBuffer& buf = pipeline->buffers[buf_idx];
    size_t img_size = pipeline->width * pipeline->height * sizeof(float);

    // Wait for all operations on this buffer's stream to complete
    cudaEventSynchronize(buf.event_d2h_done);

    // Copy result from pinned memory to caller's buffer
    memcpy(h_output, buf.h_output_pinned, img_size);

    // Retrieve max magnitude (synchronous, but stream is already done)
    cudaMemcpy(&metrics->max_magnitude, buf.d_max_mag, sizeof(float), cudaMemcpyDeviceToHost);

    // Compute per-stage timing from recorded events
    float ms = 0;
    cudaEventElapsedTime(&ms, buf.evt_t0, buf.evt_t1);
    metrics->host_to_device_ms = ms;

    cudaEventElapsedTime(&ms, buf.evt_t1, buf.evt_t2);
    metrics->gaussian_blur_ms = ms;

    cudaEventElapsedTime(&ms, buf.evt_t2, buf.evt_t3);
    metrics->sobel_x_ms = ms;

    cudaEventElapsedTime(&ms, buf.evt_t3, buf.evt_t4);
    metrics->temporal_filter_ms = ms;

    cudaEventElapsedTime(&ms, buf.evt_t4, buf.evt_t5);
    metrics->amplification_ms = ms;

    cudaEventElapsedTime(&ms, buf.evt_t5, buf.evt_t6);
    metrics->device_to_host_ms = ms;
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
