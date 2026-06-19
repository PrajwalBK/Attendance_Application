
import time
import cv2
import numpy as np
import os
import sys

# Add parent directory to path to import config
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from insightface.app import FaceAnalysis

def benchmark_insightface(use_gpu=False):
    print(f"\n--- Benchmarking InsightFace (GPU={use_gpu}) ---")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
    
    start_load = time.time()
    try:
        # Use buffalo_l as per config
        app = FaceAnalysis(name='buffalo_l', providers=providers)
        app.prepare(ctx_id=0, det_size=(640, 640))
    except Exception as e:
        print(f"Failed to load InsightFace (GPU={use_gpu}): {e}")
        return

    load_time = time.time() - start_load
    print(f"Model Load Time: {load_time:.4f}s")
    
    # Load Image
    img_path = os.path.join(parent_dir, 'ui', 'prosper1.png')
    if not os.path.exists(img_path):
        # Create dummy image if not found
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.rectangle(img, (100, 100), (300, 300), (255, 255, 255), -1)
        print("Using dummy image.")
    else:
        img = cv2.imread(img_path)
        print(f"Using image: {img_path}")

    # Warmup
    app.get(img)
    
    # Benchmark
    times = []
    for i in range(10):
        start = time.time()
        faces = app.get(img)
        dt = time.time() - start
        times.append(dt)
        print(f"Run {i+1}: {dt:.4f}s | Faces: {len(faces)}")
        
    avg_time = sum(times) / len(times)
    print(f"Average Inference Time (GPU={use_gpu}): {avg_time:.4f}s")
    print(f"FPS: {1.0/avg_time:.2f}")

def benchmark_opencv_ssd():
    print(f"\n--- Benchmarking OpenCV SSD ---")
    
    proto = os.path.join(parent_dir, 'data', 'models', 'deploy.prototxt')
    model = os.path.join(parent_dir, 'data', 'models', 'res10_300x300_ssd_iter_140000.caffemodel')
    
    if not os.path.exists(proto) or not os.path.exists(model):
        print("OpenCV models not found. Skipping.")
        return

    start_load = time.time()
    net = cv2.dnn.readNetFromCaffe(proto, model)
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    load_time = time.time() - start_load
    print(f"Model Load Time: {load_time:.4f}s")

    # Load Image
    img_path = os.path.join(parent_dir, 'ui', 'prosper1.png')
    if not os.path.exists(img_path):
        img = np.zeros((640, 640, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img_path)

    # Warmup
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    net.forward()

    times = []
    for i in range(10):
        start = time.time()
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        dt = time.time() - start
        times.append(dt)
        print(f"Run {i+1}: {dt:.4f}s")

    avg_time = sum(times) / len(times)
    print(f"Average Inference Time (SSD): {avg_time:.4f}s")
    print(f"FPS: {1.0/avg_time:.2f}")

if __name__ == "__main__":
    benchmark_insightface(use_gpu=False)
    benchmark_insightface(use_gpu=True)
    benchmark_opencv_ssd()
