#  Real-Time Edge-Optimized Eye Health & Ergonomic Monitor

A high-performance, cross-platform computer vision application designed to monitor user eye health and ergonomic habits in real-time. By leveraging a quantized deep learning model for facial landmark regression, the system continuously tracks Eye Aspect Ratio (EAR), blink rate, and physical viewing distance.

Engineered specifically for **low-latency CPU inference**, the pipeline employs advanced code optimization techniques—including INT8 static quantization and adaptive frame-rate throttling—making it highly suitable for edge deployment and resource-constrained environments.

---

## 🏗️ System Architecture & Pipeline

The system processes each frame through a highly optimized, sequential pipeline:

1. **Capture & Preprocessing:** Grabs the frame, applies dynamic gamma correction (if `luminance < 60`), resizes to 192x192, and applies ImageNet normalization.
2. **Edge Inference:** Passes the tensor through an **INT8 quantized MobileNetV2** via ONNX Runtime for ultra-fast CPU inference.
3. **Signal Processing:** Extracts 68-point landmarks, computes bilateral EAR, and calculates the pixel distance between outer eye corners.
4. **State Management:** Updates the Blink FSM (checking for open→closed EAR transitions) and updates the 20-20-20 scheduler.
5. **Alert Generation:** Evaluates thresholds (BPM < 8, Distance < 45cm) and dispatches cross-platform native toasts.

---

## ⚙️ Engineering & Optimization Highlights

* **Model Quantization (Space/Time Complexity):** Exported a PyTorch MobileNetV2 model to ONNX and applied static INT8 quantization using a custom calibration data reader. This drastically reduced the model's memory footprint and accelerated CPU inference time while maintaining landmark regression accuracy.
* **Adaptive Resource Management:** Implemented an optical-flow-based motion scoring algorithm. If the user is absent or perfectly still, the system dynamically throttles the processing pipeline from 30 FPS down to 2 FPS, significantly reducing CPU time complexity and power consumption.
* **Robust Algorithmic Filtering:** Standard EAR thresholding often triggers false blinks when the head turns (yaw). I designed an FSM that cross-references head-yaw angles (tolerance ±30°), suppressing blink counts during extreme head movements to ensure signal fidelity.
* **On-Device Geometric Calibration:** Developed a calibration routine that computes the camera's specific focal length in pixels using a known physical reference (63mm IPD at a fixed distance), ensuring the distance estimation algorithm is accurate across different hardware setups.

---

## 🛠️ Tech Stack

* **Languages:** Python 3
* **Computer Vision & Math:** OpenCV, NumPy
* **Deep Learning & Inference:** PyTorch, ONNX, ONNX Runtime (INT8 Quantization)
* **System Integration:** Cross-platform native notifications (`notify-send`, `win10toast`, `osascript`)

---

## 📦 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/nixx-04/eye-health-monitor.git
cd eye-health-monitor
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Run the Real-Time Monitor
```bash
python src/monitor.py
```
