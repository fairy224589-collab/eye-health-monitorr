import numpy as np

def eye_aspect_ratio(landmarks: np.ndarray, eye_idx: slice) -> float:
    """
    Compute EAR for a single eye.
    `landmarks` – (68, 2) array, (x, y) coordinates.
    `eye_idx`   – slice of the 68 points for the eye (e.g. 36:42 for left eye).
    Returns a floating point EAR value.
    """
    # Extract the six landmarks that define the eye
    p = landmarks[eye_idx]                     # shape (6, 2)

    # Vertical distances: |p2‑p6| + |p3‑p5|
    vert = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])

    # Horizontal distance: |p1‑p4|
    horiz = np.linalg.norm(p[0] - p[3])

    # Avoid divide‑by‑zero
    if horiz == 0:
        return 0.0

    return float(vert / (2.0 * horiz))


def overall_ear(landmarks: np.ndarray) -> float:
    """
    Average EAR of both eyes using Dlib 68‑point indices:
      left eye  = indices 36‑41
      right eye = indices 42‑47
    """
    left  = eye_aspect_ratio(landmarks, slice(36, 42))
    right = eye_aspect_ratio(landmarks, slice(42, 48))
    return (left + right) / 2.0