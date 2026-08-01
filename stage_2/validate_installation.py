from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "MISSING"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--integration", action="store_true", help="Run a tiny real WRSN simulation.")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    print("Python:", sys.version)
    print("Executable:", sys.executable)
    print("Platform:", platform.platform())
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "networkx", "torch", "SALib", "ConfigSpace", "smac"):
        print(f"{package}: {version(package)}")

    missing = [name for name in ("SALib", "ConfigSpace", "smac") if version(name) == "MISSING"]
    if missing:
        raise SystemExit(
            "Missing Stage-2 packages: " + ", ".join(missing)
            + "\nInstall: python -m pip install -r requirements_stage2.txt"
        )

    import torch

    print("CUBLAS_WORKSPACE_CONFIG:", os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
    print("CUDA runtime:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA was required but is not available.")

    from .smoke_tests import main as smoke_main
    smoke_main()

    if args.integration:
        from .pipeline import WRSNEvaluator, load_settings

        settings = load_settings(Path(__file__).with_name("settings.json"))
        evaluator = WRSNEvaluator(
            settings,
            require_cuda=args.require_cuda,
            strict_determinism=True,
            quiet=True,
        )
        result = evaluator.evaluate(
            {"qifl_learning_rate": 0.004},
            101,
            tag="Stage2InstallationIntegration",
            num_nodes=8,
            simulation_time_s=20,
        )
        print("Integration status:", result["Status"])
        print("Adaptive device:", result["Adaptive_Compute_Device"])
        print("DQN Torch enabled:", result.get("DQN_Torch_Enabled_Flag"))
        print("QIFL Torch enabled:", result.get("QIFL_Torch_Enabled_Flag"))
        print("Strict determinism:", result.get("Config_Runtime_Torch_Strict_Determinism"))

        if str(result.get("Status")) != "SUCCESS":
            raise SystemExit("Integration run did not complete successfully.")
        if float(result.get("DQN_Torch_Enabled_Flag", 0.0)) != 1.0:
            raise SystemExit("Integration run did not use the PyTorch DQN.")
        if float(result.get("QIFL_Torch_Enabled_Flag", 0.0)) != 1.0:
            raise SystemExit("Integration run did not use the PyTorch QIFL oracle.")
        if float(result.get("Config_Runtime_Torch_Strict_Determinism", 0.0)) != 1.0:
            raise SystemExit("Integration run was not in strict deterministic mode.")
        if args.require_cuda:
            if str(result.get("Adaptive_Compute_Device")) != "cuda":
                raise SystemExit("Integration run did not use CUDA.")
            if float(result.get("CUDA_Available_Flag", 0.0)) != 1.0:
                raise SystemExit("Integration output did not confirm CUDA availability.")

    print("Stage-2 installation validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
