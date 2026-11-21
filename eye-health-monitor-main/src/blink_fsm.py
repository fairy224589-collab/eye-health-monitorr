import time
from collections import deque
import numpy as np

class BlinkFSM:
    """
    Finite‑state‑machine that maintains a 60‑second rolling window of EAR values.
    Emits the blink rate (BPM) and filters false blinks when head yaw exceeds
    ±30°.
    """

    def __init__(self,
                 window_sec: int = 60,
                 healthy_min: int = 10,
                 healthy_max: int = 12,
                 yaw_tolerance: float = np.deg2rad(30)):
        self.window_sec = window_sec
        self.healthy = (healthy_min, healthy_max)
        self.yaw_tolerance = yaw_tolerance
        # Deques automatically drop entries older than window_sec.
        # We assume a maximum of 30 Hz inference → maxlen = window_sec * 30
        self.ear_timestamps = deque(maxlen=window_sec * 30)
        self.blink_timestamps = deque(maxlen=window_sec * 30)
        self.ear_values = deque(maxlen=window_sec * 30)
        self.blink_count = 0
        self.last_blink_time = time.time()
        self.head_yaw = 0.0          # rad, supplied externally each frame

    def update(self, ear: float, timestamp: float, head_yaw: float | None = None):
        """Call once per frame (or at a fixed sub‑sample rate)."""
        if head_yaw is not None:
            self.head_yaw = head_yaw

        self.ear_values.append(ear)
        self.ear_timestamps.append(timestamp)
        self.blink_timestamps.append(timestamp)


        # Detect transition open → closed (EAR < 0.2 is a common threshold)
        if len(self.ear_values) >= 2:
            prev_ear = self.ear_values[-2]
            curr_ear = ear
            if prev_ear >= 0.2 and curr_ear < 0.2:
                # Guard against yaw‑induced false blinks
                if abs(self.head_yaw) <= self.yaw_tolerance:
                    self.blink_count += 1

    def bpm(self) -> float:
        now = time.time()

        while self.blink_timestamps and (now - self.blink_timestamps[0]) > self.window_sec:
            self.blink_timestamps.popleft()
        
        blinks_in_window = len(self.blink_timestamps)
        bpm = (blinks_in_window / self.window_sec) * 60.0
        return float(np.clip(bpm, self.healthy[0], self.healthy[1]))