import numpy as np

# -----------------------------------------------------------------
# Calibrated focal length (pixels).  Set once per device, e.g. 650.
# -----------------------------------------------------------------
FOCAL_LENGTH_PX = 650.0          # <-- replace with your calibrated value

def estimate_distance(i_pixels: float | np.floating, focal_px: float = FOCAL_LENGTH_PX) -> float:
    """
    Distance Z in centimetres:
        Z = (f * 63.0 mm) / IPD_pixels   →  cm
    `i_pixels` – horizontal pixel spread between the two pupils (or any
                suitable proxy).  `focal_px` is the device‑specific focal length.
    Returns distance in centimetres.
    """
    if i_pixels == 0:
        return float('inf')
    Z_mm = (focal_px * 63.0) / float(i_pixels)
    return float(Z_mm / 10.0)    # mm → cm
