"""Capture versions and GPU provenance without collecting env vars or credentials."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[2]


def git_output(*args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def collect() -> dict:
    packages = {
        d.metadata["Name"]: d.version
        for d in importlib.metadata.distributions()
        if d.metadata.get("Name")
    }
    torch_info = {"installed": "torch" in {n.lower() for n in packages}}
    if torch_info["installed"]:
        try:
            import torch

            available = torch.cuda.is_available()
            torch_info.update(
                version=torch.__version__,
                cuda_build=torch.version.cuda,
                cuda_available=available,
                cudnn_version=torch.backends.cudnn.version(),
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                devices=[
                    {
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory,
                        "capability": list(torch.cuda.get_device_capability(i)),
                        "bf16_supported": torch.cuda.is_bf16_supported(),
                    }
                    for i in range(torch.cuda.device_count())
                ] if available else [],
            )
        except Exception as exc:
            # Type only: exception strings can include machine paths or other sensitive data.
            torch_info["import_error_type"] = type(exc).__name__
    driver = None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        if result.returncode == 0:
            driver = sorted(set(result.stdout.strip().splitlines()))
    except (OSError, subprocess.TimeoutExpired):
        pass
    java_info = {"available": False, "executable_override_used": bool(os.environ.get("NEU_ESC_JAVA"))}
    try:
        result = subprocess.run([os.environ.get("NEU_ESC_JAVA", "java"), "-version"],
                                capture_output=True, text=True, timeout=15, check=False)
        if result.returncode == 0:
            java_info.update(available=True, version_lines=(result.stderr or result.stdout).strip().splitlines()[:3])
    except (OSError, subprocess.TimeoutExpired):
        pass
    status = git_output("status", "--porcelain", "--untracked-files=all")
    return {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "packages": dict(sorted(packages.items(), key=lambda x: x[0].lower())),
        "torch": torch_info,
        "nvidia_driver_versions": driver,
        "java": java_info,
        "git": {"commit": git_output("rev-parse", "HEAD"), "clean": status == "" if status is not None else None},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()
    info = collect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(info, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    if args.require_cuda and not info["torch"].get("cuda_available"):
        print("BLOCKED: a working CUDA PyTorch runtime is required for full training.", file=sys.stderr)
        return 2
    if args.require_clean and (not info["git"]["commit"] or not info["git"]["clean"]):
        print("BLOCKED: use a committed, clean checkout for full training.", file=sys.stderr)
        return 2
    print(f"Captured environment: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
