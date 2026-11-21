import importlib
import platform
import subprocess


def show_toast(message: str, duration: int = 3000) -> None:
    """
    Show a native transient toast/notification.
    - Linux   : libnotify (`notify-send`)
    - Windows : win10toast (if installed) else a simple `msg`
    - macOS   : osascript `display notification`
    """
    sys_plat = platform.system()

    if sys_plat == "Linux":
        subprocess.run(["notify-send", "-t", str(duration), "EyeHealth", message],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    elif sys_plat == "Windows":
        # Preferred: win10toast package. Some users accidentally install a typo
        # like "qin10toast"; support both names and fall back cleanly.
        toaster_cls = None
        for package_name in ("win10toast", "qin10toast"):
            try:
                mod = importlib.import_module(package_name)
                toaster_cls = getattr(mod, "ToastNotifier", None)
                if toaster_cls is not None:
                    break
            except Exception:
                continue

        if toaster_cls is not None:
            try:
                toaster = toaster_cls()
                toaster.show_toast("EyeHealth", message, duration=duration / 1000)
            except Exception:
                pass
        else:
            # Very basic fallback (requires admin rights on some OSes)
            subprocess.run(["msg", "*", message],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    elif sys_plat == "Darwin":  # macOS
        script = (
            f'display notification "{message}" with title "EyeHealth"'
        )
        subprocess.run(["osascript", "-e", script],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    else:
        # Unknown platform – silently ignore
        pass