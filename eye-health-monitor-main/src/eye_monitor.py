"""Compatibility wrapper for the legacy eye_monitor module name.

Old scripts imported `eye_monitor`, while the active implementation lives in
`src.monitor`. This shim keeps the legacy import path working without
duplicating runtime logic.
"""

from .monitor import *  # noqa: F401,F403


class EyeHealthMonitor:
    """Compatibility placeholder used by older calibration scripts."""

    def __init__(self, *args, **kwargs):
        pass

    def process_frame(self, *args, **kwargs):
        from .monitor import process_frame
        return process_frame(*args, **kwargs)
