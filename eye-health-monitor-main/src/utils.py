import cv2
import numpy as np

def preprocess(bgr_frame: np.ndarray, input_size: int = 192) -> np.ndarray:
    """
    Convert a BGR frame to the float32 tensor expected by the INT8 MobileNetV2.
    Returns shape (1, 3, input_size, input_size) with ImageNet normalisation.
    """
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (input_size, input_size))
    rgb = rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std
    return np.transpose(rgb, (2, 0, 1))[np.newaxis, :]   # (1,3,H,W)

def gamma_correct(frame: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """Simple gamma correction for low‑luminance frames."""
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255
                      for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(frame, table)

def motion_score(gray_prev: np.ndarray, gray_curr: np.ndarray) -> float:
    """Absolute‑difference mean – used for adaptive FPS."""
    return cv2.absdiff(gray_prev, gray_curr).mean()