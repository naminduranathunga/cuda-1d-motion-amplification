#include "motion_amp.h"
#include <device_launch_parameters.h>

__constant__ float d_gaussian_weights[25];
__constant__ float d_weight_sum;

__global__ void gaussian_blur_kernel_tex2d(cudaTextureObject_t input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Use a 5x5 dynamic kernel based on sigma
    float sum = 0.0f;

    #pragma unroll
    for (int i = -2; i <= 2; ++i) {
        #pragma unroll
        for (int j = -2; j <= 2; ++j) {
            float w = d_gaussian_weights[(i+2)*5 + (j+2)];
            sum += tex2D<float>(input, x + j, y + i) * w;
        }
    }

    output[y * width + x] = sum / d_weight_sum;
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


void preComputeGaussianWeigths(float* dest, float sigma) {
    float s2 = 2.0f * sigma * sigma;
    for (int i = -2; i <= 2; ++i) {
        for (int j = -2; j <= 2; ++j) {
            float dist_sq = (float)(i * i + j * j);
            float w = expf(-dist_sq / s2);
            int row = i + 2;
            int col = j + 2;
            dest[row * 5 + col] = w;
        }
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
    float h_weights[25];
    preComputeGaussianWeigths(h_weights, sigma);
    cudaMemcpyToSymbol(d_gaussian_weights, h_weights, sizeof(h_weights));

    // calculate sum
    float h_weight_sum = 0.0f;
    for (int i = 0; i < 25; ++i) {
        h_weight_sum += h_weights[i];
    }
    cudaMemcpyToSymbol(d_weight_sum, &h_weight_sum, sizeof(float));
}

extern "C" void apply_gaussian_blur_tex2d(float* d_input, float* d_output, int width, int height, float sigma) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    gaussian_blur_kernel_tex2d<<<gridSize, blockSize>>>(h_texture_input, d_output, width, height);
}
