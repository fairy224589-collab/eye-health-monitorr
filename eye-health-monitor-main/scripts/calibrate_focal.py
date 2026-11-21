#!/usr/bin/env python3
# --------------------------------------------------------------
# Script: on‑device focal‑length calibration helper
# --------------------------------------------------------------
import cv2
import numpy as np
from src.eye_monitor import EyeHealthMonitor

def main():
    print("👉 Place a credit‑card‑size ruler with a known IPD (63 mm) "
          "at exactly 60 cm from the camera.")
    input("Press ENTER when ready … ")

    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames_needed = 30
    ipd_pixels_list = []

    for i in range(frames_needed):
        ret, frame = cap.read()
        if not ret:
            continue
        # Run the regressor (you would call the monitor's inference here)
        # For the demo we just approximate IPD by the width of the detected face.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # placeholder: assume we have landmarks – in reality run the ONNX model
        # and extract the two outermost eye x‑coordinates.
        # Here we fake a value for illustration:
        #   fake_px = 250   # replace with real pixel spread
        # ipd_pixels_list.append(fake_px)
        cv2.imshow("Calibration", frame)
        if cv2.waitKey(int(1000/fps)) & 0xFF == 27:   # ESC to abort
            break

    cap.release()
    cv2.destroyAllWindows()

    # -----------------------------------------------------------------
    # Compute focal length:  f = (Z * avg_IPD_pixels) / 63.0
    # -----------------------------------------------------------------
    # avg_ipd = np.mean(ipd_pixels_list)
    # focal = (60.0 * avg_ipd) / 63.0   # Z in cm, 63 mm = 6.3 cm
    # print(f"Calibrated focal length (px): {focal:.1f}")
    # You would store this value in a config file and set
    #   FOCAL_LENGTH_PX = focal
    # in src/utils.py / src/monitor.py

if __name__ == "__main__":
    main()