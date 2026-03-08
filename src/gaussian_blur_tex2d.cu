#include "motion_amp.h"
#include <device_launch_parameters.h>

__constant__ float d_gaussian_weights[5];

__global__ void gaussian_blur_h_kernel(cudaTextureObject_t input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float sum = 0.0f;

    #pragma unroll
    for (int i = -2; i <= 2; ++i) {
        sum += tex2D<float>(input, x + i, y) * d_gaussian_weights[i + 2];
    }

    output[y * width + x] = sum;
}

__global__ void gaussian_blur_v_kernel(float* input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float sum = 0.0f;

    #pragma unroll
    for (int i = -2; i <= 2; ++i) {
        int ny = min(max(y + i, 0), height - 1);
        sum += input[ny * width + x] * d_gaussian_weights[i + 2];
    }

    output[y * width + x] = sum;
}

cudaTextureObject_t createTexture2D(float* d_input, int width, int height) {
    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = d_input;
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
    resDesc.res.pitch2D.width = width;
    resDesc.res.pitch2D.height = height;
    resDesc.res.pitch2D.pitchInBytes = width * sizeof(float);

    // How texture is read
    cudaTextureDesc texDesc{};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;   // Get values as points
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t tex = 0;
    cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr);
    return tex;
}


void preComputeGaussianWeights1D(float* dest, float sigma) {
    float s2 = 2.0f * sigma * sigma;
    float sum = 0.0f;
    for (int i = -2; i <= 2; ++i) {
        float dist_sq = (float)(i * i);
        float w = expf(-dist_sq / s2);
        dest[i + 2] = w;
        sum += w;
    }
    // Normalize
    for (int i = 0; i < 5; ++i) {
        dest[i] /= sum;
    }
}

static cudaTextureObject_t h_texture_input = 0;

extern "C" void init_blur_texture(float* d_input, int width, int height) {
    if (h_texture_input) {
        cudaDestroyTextureObject(h_texture_input);
    }
    h_texture_input = createTexture2D(d_input, width, height);
}

extern "C" void cleanup_blur_texture() {
    if (h_texture_input) {
        cudaDestroyTextureObject(h_texture_input);
        h_texture_input = 0;
    }
}

extern "C" void set_gaussian_weights(float sigma) {
    float h_weights[5];
    preComputeGaussianWeights1D(h_weights, sigma);
    cudaMemcpyToSymbol(d_gaussian_weights, h_weights, sizeof(h_weights));
}

extern "C" void apply_gaussian_blur_tex2d(float* d_input, float* d_temp, float* d_output, int width, int height) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // 1. Horizontal pass: Texture -> Temp (Linear)
    gaussian_blur_h_kernel<<<gridSize, blockSize>>>(h_texture_input, d_temp, width, height);
    
    // 2. Vertical pass: Temp (Linear) -> Output (Linear)
    gaussian_blur_v_kernel<<<gridSize, blockSize>>>(d_temp, d_output, width, height);
}

// Per-buffer texture management for stream pipeline
extern "C" void init_blur_texture_for_buffer(float* d_input, int width, int height, cudaTextureObject_t* tex_out) {
    if (*tex_out) {
        cudaDestroyTextureObject(*tex_out);
    }
    *tex_out = createTexture2D(d_input, width, height);
}

extern "C" void cleanup_blur_texture_for_buffer(cudaTextureObject_t* tex) {
    if (*tex) {
        cudaDestroyTextureObject(*tex);
        *tex = 0;
    }
}

extern "C" void apply_gaussian_blur_tex2d_stream(cudaTextureObject_t tex, float* d_temp, float* d_output, int width, int height, cudaStream_t stream) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // 1. Horizontal pass: Texture -> Temp
    gaussian_blur_h_kernel<<<gridSize, blockSize, 0, stream>>>(tex, d_temp, width, height);
    
    // 2. Vertical pass: Temp -> Output
    gaussian_blur_v_kernel<<<gridSize, blockSize, 0, stream>>>(d_temp, d_output, width, height);
}

