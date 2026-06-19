
import onnxruntime as ort
import sys

print(f"Python Version: {sys.version}")
print(f"ONNX Runtime Version: {ort.__version__}")
print(f"Available Providers: {ort.get_available_providers()}")

try:
    import cv2
    print(f"OpenCV Version: {cv2.__version__}")
    # Check if OpenCV has CUDA
    count = cv2.cuda.getCudaEnabledDeviceCount()
    print(f"OpenCV CUDA Devices: {count}")
except Exception as e:
    print(f"OpenCV Check Failed: {e}")
