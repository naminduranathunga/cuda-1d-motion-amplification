cmake_minimum_required(VERSION 3.18)
project(motion_amp LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

# Create a shared library
add_library(motion_amp SHARED
    src/gaussian_blur.cu
    src/sobel_x.cu
    src/temporal_filter.cu
    src/amplify.cu
    src/histogram.cu
    src/wrapper.cpp
    # Tex2D versions
    src/gaussian_blur_tex2d.cu
)

# Target properties
set_target_properties(motion_amp PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

find_package(CUDAToolkit REQUIRED)

# Include directories
target_include_directories(motion_amp PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Python needs visibility
target_link_libraries(motion_amp PRIVATE CUDA::cudart)
