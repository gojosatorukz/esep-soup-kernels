"""Fail early when the measured Linux / Python 3.13 / T4 environment is unavailable."""
import argparse
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system-only", action="store_true", help="check hardware before installing dependencies")
    args = parser.parse_args()
    if platform.system() != "Linux" or sys.version_info[:2] != (3, 13):
        raise SystemExit("Use Linux with Python 3.13, matching the saved benchmark. macOS/CPU cannot run this workload.")
    if not shutil.which("nvidia-smi"):
        raise SystemExit("nvidia-smi is missing. Enable Kaggle GPU T4 x2 or install a compatible NVIDIA driver.")
    listing = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    if not listing or any("T4" not in name for name in listing):
        raise SystemExit("This reproduction targets Tesla T4 GPUs. Select Kaggle GPU T4 x2.")
    print("System check passed: Linux, Python 3.13, Tesla T4.")
    if args.system_only:
        return
    requirements = Path(__file__).resolve().parents[1] / "requirements.txt"
    for line in requirements.read_text().splitlines():
        if "==" not in line:
            continue
        package, expected = line.split("==", 1)
        package = package.split("[", 1)[0]
        try:
            actual = version(package)
        except PackageNotFoundError:
            raise SystemExit(f"Missing {package}; complete the README installation steps.") from None
        if actual.split("+", 1)[0] != expected:
            raise SystemExit(f"Expected {package} {expected}, found {actual}. Use a fresh environment.")
    import torch
    if torch.version.cuda != "12.8" or not torch.cuda.is_available():
        raise SystemExit("Use the PyTorch 2.11 CUDA 12.8 wheel from the README; an available CUDA GPU is required.")
    print("Dependency check passed: pinned package versions and CUDA 12.8.")


if __name__ == "__main__":
    main()
