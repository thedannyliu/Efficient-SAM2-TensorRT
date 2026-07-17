from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _command(command: list[str]) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _device_model() -> str | None:
    path = Path("/proc/device-tree/model")
    return path.read_bytes().rstrip(b"\x00").decode(errors="replace") if path.exists() else None


def probe_environment() -> dict[str, Any]:
    report: dict[str, Any] = {
        "host": platform.node(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "device_model": _device_model(),
        "nvidia_smi": _command(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]),
        "nvcc": _command(["nvcc", "--version"]),
        "ros_distro": os.environ.get("ROS_DISTRO"),
        "ros2": _command(["ros2", "--version"]),
    }
    try:
        import tensorrt

        report["tensorrt"] = tensorrt.__version__
    except ImportError:
        report["tensorrt"] = None
    try:
        import torch

        report["torch"] = torch.__version__
        report["torch_cuda"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            report["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        report["torch"] = None
    return report


def write_probe(path: str | Path) -> dict[str, Any]:
    report = probe_environment()
    Path(path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
