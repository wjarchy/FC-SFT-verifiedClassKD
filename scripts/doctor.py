#!/usr/bin/env python3
"""Environment diagnostics (PRD §4.16 / M0)."""

from __future__ import annotations

import os
import platform
import shutil
import sys
from pathlib import Path


def main() -> int:
    ok = True
    print("=== FC Class KD Doctor ===")
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")

    try:
        import torch

        print(f"PyTorch: {torch.__version__}")
        cuda = torch.cuda.is_available()
        print(f"CUDA available: {cuda}")
        if cuda:
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name}, {props.total_memory / 1e9:.1f} GB")
            print(f"  BF16: {torch.cuda.is_bf16_supported()}")
    except ImportError:
        print("PyTorch: NOT INSTALLED")
        ok = False

    try:
        import transformers

        print(f"Transformers: {transformers.__version__}")
    except ImportError:
        print("Transformers: NOT INSTALLED")
        ok = False

    try:
        import bitsandbytes as bnb

        print(f"bitsandbytes: {bnb.__version__}")
    except ImportError:
        print("bitsandbytes: NOT INSTALLED (needed for QLoRA on GPU)")

    try:
        import datasets

        print(f"datasets: {datasets.__version__}")
    except ImportError:
        print("datasets: NOT INSTALLED")
        ok = False

    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    print(f"HF cache: {hf_home}")

    _, _, free = shutil.disk_usage(Path.cwd())
    print(f"Disk free: {free / 1e9:.1f} GB")

    try:
        from fc_class_kd import __version__

        print(f"fc_class_kd: {__version__}")
    except ImportError:
        print("fc_class_kd: NOT INSTALLED (run pip install -e .)")
        ok = False

    if ok:
        print("ENVIRONMENT_OK=true")
        return 0
    print("ENVIRONMENT_OK=false")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
