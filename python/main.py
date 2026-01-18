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
    ]

# Define types for functions
lib.allocate_device_memory.argtypes = [ctypes.c_size_t]
lib.allocate_device_memory.restype = ctypes.c_void_p

lib.free_device_memory.argtypes = [ctypes.c_void_p]

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
    ctypes.c_void_p,                # d_state
    ctypes.c_int,                  # width
    ctypes.c_int,                  # height
    ctypes.c_float,                # alpha
    ctypes.POINTER(Metrics)        # metrics
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

def process_video(input_video, output_video, alpha=50.0):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error opening video: {input_video}")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Try to use avc1 (H.264) for web compatibility, fallback to mp4v
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height), isColor=False)
    
    if not out.isOpened():
        print("Warning: avc1 codec failed, falling back to mp4v (may not play in browsers)")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height), isColor=False)

    # Allocate state memory (2 * width * height * float)
    state_size = 2 * width * height * ctypes.sizeof(ctypes.c_float)
    d_state = lib.allocate_device_memory(state_size)

    metrics_list = []

    print(f"Processing {input_video}...")
    
    frame_idx = 0
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
            d_state,
            width,
            height,
            alpha,
            ctypes.byref(metrics)
        )

        metrics_list.append([
            metrics.host_to_device_ms,
            metrics.gaussian_blur_ms,
            metrics.sobel_x_ms,
            metrics.temporal_filter_ms,
            metrics.amplification_ms,
            metrics.device_to_host_ms
        ])

        # Postprocess: convert back to uint8
        res = (h_output * 255.0).clip(0, 255).astype(np.uint8)
        out.write(res)
        
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"Processed frame {frame_idx}")

    lib.free_device_memory(d_state)
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
    print(f"Total GPU time:      {np.sum(avg_metrics):.4f} ms")
    
    return avg_metrics

if __name__ == "__main__":
    input_file = "sample_2.mp4"
    output_file = "sample_2_output.mp4"
    
    if not os.path.exists(input_file):
        generate_synthetic_video(input_file)
        
    process_video(input_file, output_file)
