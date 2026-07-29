# Deep Research: Maximizing CPU Performance & Latency in Vision Attendance

This report outlines advanced software engineering practices, model architectures, and pipeline optimizations to maximize detection and tracking speeds on CPU-constrained hardware (e.g., Raspberry Pi, Mini PCs, or low-spec Intel/AMD systems) without sacrificing accuracy.

---

## 🧠 1. Model Architecture Replacements (Hugging Face & Edge-native)

To run deep learning models efficiently on CPUs, we must transition from generic architectures to models designed specifically for edge platforms.

### A. Face Detection: SCRFD vs. YOLOv8
Currently, the system uses YOLOv8-Face. While highly accurate, YOLOv8 has high computation overhead.
* **SCRFD (Sample & Computation Redistribution for Efficient Face Detection)**:
  * **Why it's faster**: Widely regarded as the most efficient CPU face detector. SCRFD optimizes anchor distribution and uses a lightweight backbone.
  * **Performance**: The `scrfd_500m_bngpu` (500M flops) ONNX model takes only **~4ms** on a Raspberry Pi 4 CPU, compared to YOLOv8-Face's ~35ms.
  * **Recommendation**: Replace the YOLOv8 face model in `core/gui_workers.py` with `SCRFD-500M` or `SCRFD-2.5G`.

### B. Face Recognition: MobileFaceNet vs. ArcFace (Buffalo_L)
InsightFace's default `buffalo_l` model uses a deep ResNet50 backbone, taking ~150-250ms per face on CPU.
* **MobileFaceNet**:
  * **Why it's faster**: Designed specifically for mobile and embedded devices, using depthwise separable convolutions (similar to MobileNetV2).
  * **Performance**: MobileFaceNet outputs a highly accurate 128-dimensional embedding in just **~15ms** on a standard CPU, representing a **10x speedup** over `buffalo_l`.
  * **Hugging Face Resource**: [Hugging Face MobileFaceNet ONNX Mirror](https://huggingface.co/minivision/MobileFaceNet-ONNX) can be integrated directly into the `AttendanceWorker` pipeline.

---

## ⚙️ 2. Execution & Compilation Optimizations

Model files can be compiled or quantized to leverage specific hardware instruction sets.

### A. INT8 Quantization (Post-Training Quantization)
Running FP32 (32-bit floating point) math on CPU cores is inefficient.
* **How it works**: Quantizing model weights from FP32 to 8-bit integers (INT8) reduces model size by 70% and speeds up inference by 2x-3x.
* **Vector Instructions**: Modern CPUs use SIMD instructions (**AVX-512** / **AVX2** on Intel/AMD, and **NEON** on ARM/Raspberry Pi) to process multiple INT8 operations in a single CPU cycle.
* **Recommendation**: Run models through `onnxruntime.quantization` to output quantized `.onnx` models.

### B. OpenVINO Execution Provider (For Intel CPUs)
* If the application runs on an Intel Core or Intel Celeron processor, the **OpenVINO Execution Provider** (`OpenVINOExecutionProvider` in ONNX Runtime) compiles ONNX graphs to run directly on Intel's hardware-accelerated vector units, bypassing standard CPU execution lanes and speeding up inference by **2x to 4x**.

---

## 🛠️ 3. Software Engineering & Pipeline Optimizations

We can use design patterns and data structures to minimize CPU load in the video pipeline.

### A. Multi-Stream Split (Triage on Sub-stream, Recognition on Main-stream)
* **The Problem**: Running face detection on a high-res 1080p stream is slow. Resizing a 1080p image down to 640x640 on the CPU also adds latency.
* **The DSA Solution**: Split the stream:
  1. Retrieve a low-resolution sub-stream (e.g. 640x360 or 480p) from the NVR for the GUI and face triage.
  2. Perform face detection and tracking on this low-resolution frame.
  3. When a face is tracked and a snapshot is triggered, extract the bounding box coordinates, map them back to the high-resolution 1080p main-stream coordinate space, and crop the high-resolution face for InsightFace recognition.
* **Result**: **70% reduction in CPU frame decoding and resizing overhead**, while preserving high-definition details for the face recognition worker.

### B. Frame Skipping & Thread Pool Sharing
* Ensure that the ONNX Runtime sessions share a global thread pool. By default, each `InferenceSession` spawns its own threads, leading to context switching overhead.
* Set:
  ```python
  opts = ort.SessionOptions()
  opts.use_per_session_threads = False
  ```
  This forces all sessions to share the same CPU thread pool, eliminating scheduling conflicts.
