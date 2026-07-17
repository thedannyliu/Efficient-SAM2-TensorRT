from __future__ import annotations

import json
from pathlib import Path


def pin_environment(probe_path: str | Path, output_path: str | Path) -> dict[str, object]:
    probe = json.loads(Path(probe_path).read_text(encoding="utf-8"))
    model = str(probe.get("device_model") or "")
    if "thor" not in model.lower():
        raise ValueError(f"refusing to pin a non-Thor environment: {model or 'unknown'}")
    required = ("architecture", "device_model", "tensorrt", "torch_cuda", "ros_distro")
    missing = [key for key in required if not probe.get(key)]
    if missing:
        raise ValueError(f"probe is missing required values: {', '.join(missing)}")
    lock = {"schema_version": 1, **{key: probe[key] for key in required}}
    Path(output_path).write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return lock
