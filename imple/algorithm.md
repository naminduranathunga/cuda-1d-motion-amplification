Algorithm (standard academic pipeline)
1. Video input

Regular RGB or grayscale video

30 FPS is enough

No special camera hardware

2. Spatial decomposition

Apply a Laplacian or Gaussian pyramid per frame.

Purpose:

Separate motion by spatial frequency

Avoid amplifying noise

Mathematically:

Each frame → multi-scale representation

Each scale processed independently

3. Temporal band-pass filtering (key step)

For each pixel at each pyramid level:

Take its intensity over time

Apply a temporal band-pass filter (e.g. IIR / FIR)

This isolates:

Heartbeat range

Small vibrations

Micro facial movements

This step is embarrassingly parallel → perfect for CUDA

4. Motion amplification

Multiply the filtered signal by factor α:

𝐼
′
(
𝑥
,
𝑦
,
𝑡
)
=
𝐼
(
𝑥
,
𝑦
,
𝑡
)
+
𝛼
⋅
𝐵
(
𝑥
,
𝑦
,
𝑡
)
I
′
(x,y,t)=I(x,y,t)+α⋅B(x,y,t)

Where:

𝐵
B = band-passed temporal signal

𝛼
α = amplification factor (e.g. 10–50)

5. Reconstruction

Collapse pyramid levels

Rebuild full frame

Write amplified video

Why no high-speed camera is needed
Method	Camera
Optical flow / Lagrangian	Often needs high FPS
Eulerian motion magnification	Normal video works ✅

Reason:

It amplifies sub-pixel intensity variations, not explicit motion vectors.

Why this is perfect for your GPU CA
GPU-heavy stages

Pyramid construction → convolution

Temporal filtering → per-pixel time series

Reconstruction → convolution

GPU optimization focus

Global vs shared memory

Coalesced frame access

Reusing pyramid buffers

Constant memory for filter coefficients

What you should implement (1-week scope)

Minimal academic version

Grayscale video

Single pyramid level

Simple IIR temporal filter

CUDA kernels:

Frame copy

Temporal filter

Amplification

This is already conference-paper worthy for undergrad.