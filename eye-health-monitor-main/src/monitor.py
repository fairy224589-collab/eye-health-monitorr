# --------------------------------------------------------------
# file: src/monitor.py
# --------------------------------------------------------------
import time
from pathlib import Path
import numpy as np
import sys
import cv2
import onnxruntime as ort

from ear import overall_ear
from distance import estimate_distance
from blink_fsm import BlinkFSM
from utils import preprocess, motion_score, gamma_correct
from io_toasts import show_toast

# -----------------------------------------------------------------
# Constants (adjust once per device)
# -----------------------------------------------------------------
IPD_MM = 63.0                # mean inter‑pupil distance
INPUT_SIZE = 112
CONF_THRESH = 0.5

# -----------------------------------------------------------------
# ONNX Runtime session (INT8)
# -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_CANDIDATES = [
    PROJECT_ROOT / "models" / "model_int8.onnx",
    PROJECT_ROOT / "models" / "mobilenetv2_136_int8.onnx",
    PROJECT_ROOT / "models" / "mobilenetv2_136.onnx",
]
preferred_model = next((path for path in MODEL_CANDIDATES if path.exists()), MODEL_CANDIDATES[0])
MODEL_PATH = preferred_model
session = None
input_name = None

try:
    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
except Exception as exc:
    print(f"Warning: INT8 model not usable on this CPU backend: {exc}", file=sys.stderr)
    fallback_model = PROJECT_ROOT / "models" / "mobilenetv2_136.onnx"
    if fallback_model.exists() and fallback_model != MODEL_PATH:
        try:
            MODEL_PATH = fallback_model
            session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
            input_name = session.get_inputs()[0].name
            print(f"Loaded float32 fallback model: {MODEL_PATH}", file=sys.stderr)
        except Exception as fallback_exc:
            print(f"Warning: fallback model unavailable: {fallback_exc}", file=sys.stderr)
    if session is None:
        print(f"Warning: ONNX model unavailable at {MODEL_PATH}: {exc}", file=sys.stderr)

# -----------------------------------------------------------------
# Global objects
# -----------------------------------------------------------------
blink_fsm = BlinkFSM(window_sec=60,
                     healthy_min=10,
                     healthy_max=12,
                     yaw_tolerance=np.deg2rad(30))

# Adaptive‑FPS state
prev_gray = None
desired_fps = 30          # start full‑rate; will be lowered if needed
fps_last_change = 0.0
FPS_DEBOUNCE_SEC = 3.0

# 20‑20‑20 break scheduler state
focus_start = time.time()
rest_start = focus_start
in_break = False


def process_frame(bgr_frame: np.ndarray,
                  head_yaw: float = 0.0) -> dict:
    """
    Core pipeline:
      • pre‑process → ONNX INT8 inference → 68‑point landmarks
      • EAR, blink‑rate, distance, alerts
    Returns a dict that the caller can display or forward.
    """
    global prev_gray, desired_fps, fps_last_change

    t0 = time.time()

    #Adaptive Pooling : drops to 2 FPS if user absent
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    if prev_gray is not None:
        motion = motion_score(prev_gray, gray)
        # Debounce FPS changes - require stable motion for FPS_DEBOUNCE_SEC
        now = time.time()
        if motion < 5.0 and (now - fps_last_change) > FPS_DEBOUNCE_SEC:
            desired_fps = 2
            fps_last_change = now
        elif motion >= 5.0 and (now - fps_last_change) > FPS_DEBOUNCE_SEC:
            desired_fps = 30
            fps_last_change = now
    prev_gray = gray

    #Gamma correction if luminance is less than 60
    if gray.mean() < 60:
        bgr_frame = gamma_correct(bgr_frame, gamma=2.2)

    try:
        inp = preprocess(bgr_frame, INPUT_SIZE)          # (1,3,192,192) float32
    except Exception as e:
        print(f"Preprocess error: {e}", file=sys.stderr)
        return _error_result("preprocess failed")

    if session is None or input_name is None:
        print("ONNX inference skipped: model is not available.", file=sys.stderr)
        return _error_result("model not available")

    #ONNX INT8 Inference
    try:
        ort_inputs = {input_name: inp}
        outputs = session.run(None, ort_inputs)
        pred_136 = np.asarray(outputs[0]).reshape(-1)
        if pred_136.size != 136:
            raise ValueError(f"Unexpected landmark count: got {pred_136.size}, expected 136")
    except Exception as e:
        print(f"ONNX inference error: {e}", file=sys.stderr)
        return _error_result("inference failed")


    # Reshape to (68,2) – order matches Dlib 68‑point topology
    landmarks = pred_136.reshape(-1, 2)            # floating pixel coords

    #EAR
    try:
        ear = overall_ear(landmarks)
    except Exception as e:
        print(f"EAR computation error: {e}", file=sys.stderr)
        return _error_result("EAR computation failed")

    #Blink FSM update
    now = time.time()
    blink_fsm.update(ear, now, head_yaw=head_yaw)
    blink_rate_bpm = blink_fsm.bpm()

    #Distance estimation
    try:
        # Use outer eye corners as IPD proxy
        pupil_px = float(np.linalg.norm(landmarks[39] - landmarks[42]))
        distance_cm = estimate_distance(pupil_px)
    except (IndexError, Exception) as e:
        print(f"Distance estimation error: {e}", file=sys.stderr)
        distance_cm = float('inf')

    # Keep a short history for averaging
    # (here we just keep the last 10 readings at 2 Hz)
    # (Implementation omitted for brevity – you can add a deque if desired.)

    # -------------------------------------------------
    # 8️⃣  20‑20‑20 Scheduler (0.1 Hz)
    # -------------------------------------------------
    # Trigger a break after 20 min of continuous focus.
    # (Simple timestamp check; see src/monitor.py for full logic.)
    # For this snippet we just return a flag that the caller can test.
    _check_2020_scheduler(now)


    # -------------------------------------------------
    # 9️⃣  Build output dictionary
    # -------------------------------------------------
    alerts = {}
    if blink_rate_bpm < 8:
        alerts["alert_low_blink"] = True
    # distance‑close alert (threshold 45 cm)
    if distance_cm < 45:
        alerts["alert_distance_close"] = True
    # take‑break flag from scheduler
    alerts["alert_take_break"] = _is_in_break()

    elapsed_ms = (time.time() - t0) * 1000
    return {
        "ear": round(ear, 4),
        "blink_rate_bpm": round(blink_rate_bpm, 1),
        "distance_cm": round(distance_cm, 1),
        "alert_low_blink": alerts.get("alert_low_blink", False),
        "alert_distance_close": alerts.get("alert_distance_close", False),
        "alert_take_break": alerts.get("alert_take_break", False),
        "pipeline_ms": round(elapsed_ms, 1)
    }


def _check_2020_scheduler(current_time: float):
    """Implement 20‑20‑20 break scheduler."""
    global focus_start, rest_start, in_break

    if focus_start is None:
        focus_start = current_time
    if rest_start is None:
        rest_start = current_time
    if in_break is None:
        in_break = False

    elapsed_since_focus = current_time - focus_start
    if elapsed_since_focus >= 20 * 60:
        if not in_break:
            show_toast("🛌 20-20-20: Look 20 ft away for 20 s")
            in_break = True
            rest_start = current_time
            focus_start = current_time   # ← reset immediately on break trigger
    else:
        in_break = False


def _is_in_break() -> bool:
    """Return current break status for alert dict."""
    global in_break
    return in_break


def _error_result(message: str) -> dict:
    """Return a consistent error result dict."""
    return {
        "ear": 0.0,
        "blink_rate_bpm": 0.0,
        "distance_cm": float('inf'),
        "alert_low_blink": False,
        "alert_distance_close": False,
        "alert_take_break": False,
        "pipeline_ms": 0.0,
        "error": message
    }


# -----------------------------------------------------------------
# Optional CLI entry‑point for quick testing
# -----------------------------------------------------------------
if __name__ == "__main__":
    import sys
    cap = cv2.VideoCapture(0)
    print("--- EyeHealth Monitor (press 'q' to quit) ---")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out = process_frame(frame, head_yaw=0.0)   # head_yaw could come from a pose estimator
        # tiny on‑screen dump (feel free to replace with a proper UI)
        print(f"\rEAR:{out['ear']:.2f}  Blink:{out['blink_rate_bpm']:>3.0f} BPM  Dist:{out['distance_cm']:>4.1f} cm  "
              f"pipeline:{out['pipeline_ms']} ms  FPS:{desired_fps}", end="")
        key = cv2.waitKey(int(1000 / max(desired_fps, 5))) & 0xFF  # minimum FPS floor of 5
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
