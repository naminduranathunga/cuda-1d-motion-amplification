We are going to make an application that amplifies the motion in a video using CUDA.

## Project information.
- We use hybrid approach to implement this. (Python + C++ + CUDA CMAKE)
- Python is used for video loading, converting to grayscale, and saving the output video.
- C++ is used for the main application.
- CUDA is used for the motion amplification. (nvcc)
- Scope: We amplify horizontal (X) motion only.

## The algorithm

1. Convert video to grayscale frames.
2. Compute gradient:
    - Gussian blur + Sobel X
3. Temporal band-pass filter gradient
4. Amplify filtered signal
5. Add back to original frame
6. Save the output video

## Implementation Plan

1. Implement the basic application. 
    - Setup workspace
    - Python loader
    - C++ main application
    - Algorithms (Element-wise)
        - Gaussian blur
        - Sobel X
        - Temporal band-pass filter
        - Amplification

    - Python initiates cuda c++ program.
    - Algorithm kernels in c++ -> Use only global memory. element wise access. No loops within kernels unless extreme necessary.
    - Python calls C++ functions.
 


## Metrixes

This is an academic project. So, we collect data at each step. 
- Copy time Host -> Device
- Gussian Blur Kernel runtime
- Sobel Kernel runtime
- Temporal band-pass filter runtime
- Amplification runtime
- Copy time Device -> Host