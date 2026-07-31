"""Capture the exact runtime used for a defensible WRSN experiment."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def main() -> None:
    try:
        import torch
    except Exception:
        torch = None

    cuda_available = bool(torch is not None and torch.cuda.is_available())
    gpu_names = []
    if cuda_available:
        gpu_names = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        git_commit = "not-a-git-checkout"

    report = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": {
            name: package_version(name)
            for name in (
                "numpy",
                "scipy",
                "pandas",
                "scikit-learn",
                "networkx",
                "matplotlib",
                "torch",
            )
        },
        "cuda_available": cuda_available,
        "cuda_runtime_version": (
            getattr(getattr(torch, "version", None), "cuda", None)
            if torch is not None else None
        ),
        "cudnn_version": (
            torch.backends.cudnn.version()
            if torch is not None and hasattr(torch.backends, "cudnn") else None
        ),
        "gpu_names": gpu_names,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "PYTHONHASHSEED",
                "CUBLAS_WORKSPACE_CONFIG",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "WRSN_ADAPTIVE_DEVICE",
                "WRSN_REQUIRE_TORCH",
                "WRSN_REQUIRE_CUDA",
                "WRSN_RUN_MODE",
                "WRSN_NUM_SEEDS",
                "WRSN_NUM_NODES",
                "WRSN_SIM_TIME",
            )
        },
        "git_commit": git_commit,
    }
    output = Path(os.environ.get(
        "WRSN_ENV_REPORT", "wrsn_runtime_environment.json"))
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[ENV] Saved {output}")
    if not cuda_available:
        print("[ENV] WARNING: CUDA is not available in this runtime.")


if __name__ == "__main__":
    main()
