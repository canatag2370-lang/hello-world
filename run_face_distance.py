"""Convenience launcher that installs dependencies and runs face_distance.py."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

REQUIRED_PACKAGES = {
    "cv2": "opencv-python",
    "mediapipe": "mediapipe",
    "numpy": "numpy",
}


def ensure_packages() -> None:
    for module_name, package_name in REQUIRED_PACKAGES.items():
        if importlib.util.find_spec(module_name) is None:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])


def main() -> None:
    ensure_packages()
    script_path = Path(__file__).with_name("face_distance.py")
    subprocess.check_call([sys.executable, str(script_path)])


if __name__ == "__main__":
    main()
