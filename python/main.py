import cv2
import numpy as np
import ctypes
import os
import time

# Load the shared library
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../build/libmotion_amp.so"))
lib = ctypes.CDLL(lib_path)

# Define the Metrics struct
class Metrics(ctypes.Structure):
    _fields_ = [
        ("host_to_device_ms", ctypes.c_float),
        ("gaussian_blur_ms", ctypes.c_float),
        ("sobel_x_ms", ctypes.c_float),
        ("temporal_filter_ms", ctypes.c_float),
        ("amplification_ms", ctypes.c_float),
        ("device_to_host_ms", ctypes.c_float),
        ("max_magnitude", ctypes.c_float),
    ]

# Define the GPUContext struct
class GPUContext(ctypes.Structure):
    _fields_ = [
        ("d_input", ctypes.c_void_p),
        ("d_blur", ctypes.c_void_p),
        ("d_temp_blur", ctypes.c_void_p),
        ("d_sobel", ctypes.c_void_p),
        ("d_filtered", ctypes.c_void_p),
        ("d_output", ctypes.c_void_p),
        ("d_state", ctypes.c_void_p),
        ("d_max_mag", ctypes.c_void_p),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
    ]

# Define StreamBuffer struct (must match C++ layout)
class StreamBuffer(ctypes.Structure):
    _fields_ = [
        ("d_input", ctypes.c_void_p),
        ("d_blur", ctypes.c_void_p),
        ("d_temp_blur", ctypes.c_void_p),
        ("d_sobel", ctypes.c_void_p),
        ("d_filtered", ctypes.c_void_p),
        ("d_output", ctypes.c_void_p),
        ("d_max_mag", ctypes.c_void_p),
        ("h_input_pinned", ctypes.c_void_p),
        ("h_output_pinned", ctypes.c_void_p),
        ("stream", ctypes.c_void_p),       # cudaStream_t
        ("event_h2d_done", ctypes.c_void_p),  # cudaEvent_t
        ("event_kernels_done", ctypes.c_void_p),
        ("event_d2h_done", ctypes.c_void_p),
        # Per-stage timing events
        ("evt_t0", ctypes.c_void_p),
        ("evt_t1", ctypes.c_void_p),
        ("evt_t2", ctypes.c_void_p),
        ("evt_t3", ctypes.c_void_p),
        ("evt_t4", ctypes.c_void_p),
        ("evt_t5", ctypes.c_void_p),
        ("evt_t6", ctypes.c_void_p),
        ("tex_input", ctypes.c_uint64),     # cudaTextureObject_t is unsigned long long
    ]

# Define StreamPipeline struct
class StreamPipeline(ctypes.Structure):
    _fields_ = [
        ("buffers", StreamBuffer * 5),  # MAX_STREAM_BUFFERS = 5
        ("d_state", ctypes.c_void_p),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("num_buffers", ctypes.c_int),
        ("event_prev_kernel_done", ctypes.c_void_p),
    ]

# Define PipelineMetrics struct
class PipelineMetrics(ctypes.Structure):
    _fields_ = [
        ("total_wall_time_ms", ctypes.c_float),
        ("avg_host_to_device_ms", ctypes.c_float),
        ("avg_gaussian_blur_ms", ctypes.c_float),
        ("avg_sobel_x_ms", ctypes.c_float),
        ("avg_temporal_filter_ms", ctypes.c_float),
        ("avg_amplification_ms", ctypes.c_float),
        ("avg_device_to_host_ms", ctypes.c_float),
        ("avg_frame_time_ms", ctypes.c_float),
        ("max_magnitude", ctypes.c_float),
        ("total_frames", ctypes.c_int),
    ]

# Define types for functions
lib.allocate_device_memory.argtypes = [ctypes.c_size_t]
lib.allocate_device_memory.restype = ctypes.c_void_p

lib.free_device_memory.argtypes = [ctypes.c_void_p]

lib.initGPU.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float]
lib.initGPU.restype = ctypes.POINTER(GPUContext)

lib.cleanupGPU.argtypes = [ctypes.POINTER(GPUContext)]

lib.get_histogram.argtypes = [
    ctypes.POINTER(ctypes.c_float), # h_input
    ctypes.POINTER(ctypes.c_int),   # h_histogram (size 256)
    ctypes.c_int,                  # x1
    ctypes.c_int,                  # y1
    ctypes.c_int,                  # x2
    ctypes.c_int,                  # y2
    ctypes.c_int,                  # width
    ctypes.c_int                   # height
]

def compute_histogram(image_float, x1, y1, x2, y2):
    height, width = image_float.shape
    h_histogram = np.zeros(256, dtype=np.int32)
    lib.get_histogram(
        image_float.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        h_histogram.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        int(x1), int(y1), int(x2), int(y2),
        width, height
    )
    return h_histogram

lib.process_frame.argtypes = [
    ctypes.POINTER(ctypes.c_float), # h_input
    ctypes.POINTER(ctypes.c_float), # h_output
    ctypes.POINTER(GPUContext),     # context
    ctypes.c_float,                # alpha
    ctypes.c_float,                # alpha_l
    ctypes.c_float,                # alpha_h
    ctypes.c_float,                # threshold
    ctypes.POINTER(Metrics)        # metrics
]

# Stream pipeline function bindings
lib.initStreamPipeline.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int]
lib.initStreamPipeline.restype = ctypes.POINTER(StreamPipeline)

lib.cleanupStreamPipeline.argtypes = [ctypes.POINTER(StreamPipeline)]

lib.submit_frame.argtypes = [
    ctypes.POINTER(StreamPipeline),   # pipeline
    ctypes.POINTER(ctypes.c_float),   # h_input
    ctypes.c_int,                     # frame_idx
    ctypes.c_float,                   # alpha
    ctypes.c_float,                   # alpha_l
    ctypes.c_float,                   # alpha_h
    ctypes.c_float,                   # threshold
]

lib.collect_frame.argtypes = [
    ctypes.POINTER(StreamPipeline),   # pipeline
    ctypes.POINTER(ctypes.c_float),   # h_output
    ctypes.c_int,                     # frame_idx
    ctypes.POINTER(Metrics),          # metrics
]

def generate_synthetic_video(filename, width=640, height=480, frames=100):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 30.0, (width, height), isColor=False)
    
    for i in range(frames):
        # White rectangle with small horizontal oscillation
        img = np.zeros((height, width), dtype=np.uint8)
        x_offset = int(5 * np.sin(2 * np.pi * i / 10)) # 0.5Hz oscillation at 30fps
        cv2.rectangle(img, (200 + x_offset, 150), (400 + x_offset, 350), 255, -1)
        out.write(img)
    
    out.release()
    print(f"Generated {filename}")

def process_video(input_video, output_video, alpha=50.0, sigma=1.0, low_freq=0.5, high_freq=2.0, threshold=0.1, user_fps=None):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error opening video: {input_video}")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if user_fps:
        fps = user_fps
    
    # Calculate IIR alphas
    def freq_to_alpha(f, fs):
        # alpha = (2*pi*f/fs) / (1 + 2*pi*f/fs)
        w = 2 * np.pi * f / fs
        return w / (1 + w)

    alpha_l = freq_to_alpha(low_freq, fps)
    alpha_h = freq_to_alpha(high_freq, fps)
    print(f"Settings: Sigma={sigma}, Low={low_freq}Hz, High={high_freq}Hz, FPS={fps}")

    # Try to use avc1 (H.264) for web compatibility, fallback to mp4v
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height), isColor=False)
    
    if not out.isOpened():
        print("Warning: avc1 codec failed, falling back to mp4v (may not play in browsers)")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height), isColor=False)

    # Initialize GPU context once
    ctx = lib.initGPU(width, height, sigma)

    metrics_list = []

    print(f"Processing {input_video}...")
    
    frame_idx = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess: grayscale and float32 [0, 1]
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            h_input = gray.astype(np.float32) / 255.0
            h_output = np.zeros_like(h_input)

            # Call CUDA
            metrics = Metrics()
            lib.process_frame(
                h_input.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                h_output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctx,
                alpha,
                alpha_l,
                alpha_h,
                threshold,
                ctypes.byref(metrics)
            )

            metrics_list.append([
                metrics.host_to_device_ms,
                metrics.gaussian_blur_ms,
                metrics.sobel_x_ms,
                metrics.temporal_filter_ms,
                metrics.amplification_ms,
                metrics.device_to_host_ms,
                metrics.max_magnitude
            ])

            # Postprocess: convert back to uint8
            res = (h_output * 255.0).clip(0, 255).astype(np.uint8)
            out.write(res)
            
            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"Processed frame {frame_idx}")
    finally:
        lib.cleanupGPU(ctx)
        cap.release()
        out.release()

    # Calculate average metrics
    avg_metrics = np.mean(metrics_list, axis=0)
    print("\n--- Performance Metrics (Average ms per frame) ---")
    print(f"Host -> Device:      {avg_metrics[0]:.4f} ms")
    print(f"Gaussian Blur:       {avg_metrics[1]:.4f} ms")
    print(f"Sobel X:             {avg_metrics[2]:.4f} ms")
    print(f"Temporal Filter:     {avg_metrics[3]:.4f} ms")
    print(f"Motion Amplify:      {avg_metrics[4]:.4f} ms")
    print(f"Device -> Host:      {avg_metrics[5]:.4f} ms")
    print(f"Total GPU time:      {np.sum(avg_metrics[:6]):.4f} ms")
    
    return avg_metrics


def process_video_streamed(input_video, output_video, alpha=50.0, sigma=1.0, low_freq=0.5, high_freq=2.0, threshold=0.1, user_fps=None, num_buffers=5):
    """Process video using CUDA streams with circular buffer for latency hiding."""
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error opening video: {input_video}")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if user_fps:
        fps = user_fps
    
    # Calculate IIR alphas
    def freq_to_alpha(f, fs):
        w = 2 * np.pi * f / fs
        return w / (1 + w)

    alpha_l = freq_to_alpha(low_freq, fps)
    alpha_h = freq_to_alpha(high_freq, fps)
    print(f"[Streamed] Settings: Sigma={sigma}, Low={low_freq}Hz, High={high_freq}Hz, FPS={fps}, Buffers={num_buffers}")

    # Try to use avc1 (H.264) for web compatibility, fallback to mp4v
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height), isColor=False)
    
    if not out.isOpened():
        print("Warning: avc1 codec failed, falling back to mp4v")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height), isColor=False)

    # Initialize stream pipeline
    pipeline = lib.initStreamPipeline(width, height, ctypes.c_float(sigma), num_buffers)

    metrics_list = []
    max_mag_overall = 0.0

    print(f"Processing {input_video} with {num_buffers} stream buffers...")
    
    wall_start = time.time()
    
    # Read all frames first (or process in a streaming fashion)
    # We use a sliding window approach:
    #   - Submit up to num_buffers frames ahead
    #   - Collect the oldest submitted frame before submitting more
    
    frame_idx = 0
    next_collect_idx = 0
    h_output = np.zeros((height, width), dtype=np.float32)
    
    try:
        frames_submitted = 0
        eof = False
        
        while True:
            # Submit frames until we fill the pipeline or run out of frames
            while frames_submitted - next_collect_idx < num_buffers and not eof:
                ret, frame = cap.read()
                if not ret:
                    eof = True
                    break
                
                # Preprocess: grayscale and float32 [0, 1]
                if len(frame.shape) == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray = frame
                
                h_input = gray.astype(np.float32) / 255.0
                
                lib.submit_frame(
                    pipeline,
                    h_input.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    frames_submitted,
                    ctypes.c_float(alpha),
                    ctypes.c_float(alpha_l),
                    ctypes.c_float(alpha_h),
                    ctypes.c_float(threshold)
                )
                frames_submitted += 1
            
            # If nothing left to collect, we're done
            if next_collect_idx >= frames_submitted:
                break
            
            # Collect the oldest frame
            metrics = Metrics()
            lib.collect_frame(
                pipeline,
                h_output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                next_collect_idx,
                ctypes.byref(metrics)
            )
            
            if metrics.max_magnitude > max_mag_overall:
                max_mag_overall = metrics.max_magnitude
            
            metrics_list.append([
                metrics.host_to_device_ms,
                metrics.gaussian_blur_ms,
                metrics.sobel_x_ms,
                metrics.temporal_filter_ms,
                metrics.amplification_ms,
                metrics.device_to_host_ms,
                metrics.max_magnitude
            ])
            
            # Postprocess: convert back to uint8
            res = (h_output * 255.0).clip(0, 255).astype(np.uint8)
            out.write(res)
            
            next_collect_idx += 1
            if next_collect_idx % 50 == 0:
                print(f"Processed frame {next_collect_idx}")
    
    finally:
        lib.cleanupStreamPipeline(pipeline)
        cap.release()
        out.release()

    wall_end = time.time()
    wall_time_ms = (wall_end - wall_start) * 1000.0
    total_frames = next_collect_idx

    # Calculate average metrics
    if metrics_list:
        avg_metrics = np.mean(metrics_list, axis=0)
    else:
        avg_metrics = np.zeros(7)
    
    throughput_fps = total_frames / (wall_time_ms / 1000.0) if total_frames > 0 and wall_time_ms > 0 else 0
    avg_frame_time_ms = wall_time_ms / total_frames if total_frames > 0 else 0

    print(f"\n--- Streamed Pipeline Metrics ({num_buffers} buffers) ---")
    print(f"Total frames:        {total_frames}")
    print(f"Wall clock time:     {wall_time_ms:.2f} ms")
    print(f"Avg frame time:      {avg_frame_time_ms:.4f} ms")
    print(f"Throughput:          {throughput_fps:.2f} fps")
    print(f"Max motion magnitude: {max_mag_overall:.6f}")
    
    return {
        'host_to_device': float(avg_metrics[0]),
        'gaussian_blur': float(avg_metrics[1]),
        'sobel_x': float(avg_metrics[2]),
        'temporal_filter': float(avg_metrics[3]),
        'amplification': float(avg_metrics[4]),
        'device_to_host': float(avg_metrics[5]),
        'max_magnitude': float(max_mag_overall),
        'throughput_fps': float(throughput_fps),
        'wall_time_ms': float(wall_time_ms),
        'avg_frame_time_ms': float(avg_frame_time_ms),
        'total_frames': int(total_frames),
    }


if __name__ == "__main__":
    input_file = "sample_2.mp4"
    output_file = "sample_2_output.mp4"
    
    if not os.path.exists(input_file):
        generate_synthetic_video(input_file)
    
    # Use streamed pipeline by default
    process_video_streamed(input_file, output_file, num_buffers=5)
