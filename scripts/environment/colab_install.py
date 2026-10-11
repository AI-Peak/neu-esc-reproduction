"""Constrain and preserve an existing verified PyTorch GPU stack in Colab."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import sys

from capture import ROOT, collect

GPU_STACK = ("torch", "torchvision", "torchaudio", "triton")
IMPORTS = ("numpy", "pandas", "yaml", "torch", "transformers", "sentencepiece", "sklearn", "matplotlib", "tqdm")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Install only after reviewing the dry-run report")
    args = parser.parse_args()
    before = collect()
    if before["platform"]["system"] != "Linux" or not before["torch"].get("cuda_available"):
        print("BLOCKED: this recipe requires a verified Linux CUDA runtime.", file=sys.stderr)
        return 2
    if not before["git"]["commit"] or not before["git"]["clean"]:
        print("BLOCKED: checkout a fixed commit with a clean worktree first.", file=sys.stderr)
        return 2
    if args.evidence_dir.exists():
        print("Use a new evidence directory for each attempt; existing output is preserved.", file=sys.stderr)
        return 2
    args.evidence_dir.mkdir(parents=True)
    packages = before["packages"]
    stack = {name: version for name, version in packages.items()
             if name.lower() in GPU_STACK or name.lower().startswith("nvidia-")}
    constraints = args.evidence_dir / "gpu-stack.constraints.txt"
    constraints.write_text("".join(f"{n}=={v}\n" for n, v in sorted(stack.items())), encoding="utf-8")
    (args.evidence_dir / "environment.before.json").write_text(json.dumps(before, indent=2) + "\n", encoding="utf-8")
    report_path = args.evidence_dir / "install-plan.json"
    command = [sys.executable, "-m", "pip", "install", "--constraint", str(constraints),
               "--no-build-isolation", "--editable", f"{ROOT}[data,train,eval,dev]"]
    # Do not serialize the entire pip report: direct URLs may contain credentials.
    plan = subprocess.run([*command, "--dry-run", "--report", "-", "--quiet"],
                          capture_output=True, text=True, check=False)
    if plan.returncode:
        print("BLOCKED: constrained resolver failed. Review dependency compatibility; do not remove GPU pins.", file=sys.stderr)
        return 2
    report = json.loads(plan.stdout)
    proposed = [{"name": item["metadata"]["name"], "version": item["metadata"]["version"]}
                for item in report["install"]]
    report_path.write_text(json.dumps(proposed, indent=2) + "\n", encoding="utf-8")
    protected = {re.sub(r"[-_.]+", "-", name.lower()) for name in stack}
    changed_stack = [item["name"] for item in proposed
                     if re.sub(r"[-_.]+", "-", item["name"].lower()) in protected
                     or item["name"].lower() in GPU_STACK
                     or item["name"].lower().startswith("nvidia-")]
    if changed_stack:
        print("BLOCKED: installation would modify the GPU stack: " + ", ".join(changed_stack), file=sys.stderr)
        return 2
    print(json.dumps(proposed, indent=2))
    if not args.apply:
        print("PREPARED: review install-plan.json, then use --apply with a new evidence directory.")
        return 0
    result = subprocess.run(command, check=False)
    if result.returncode:
        return result.returncode
    if subprocess.run([sys.executable, "-m", "pip", "check"], check=False).returncode:
        print("BLOCKED: pip check failed; preserve evidence and resolve the environment.", file=sys.stderr)
        return 2
    current = {n: importlib.metadata.version(n) for n in stack}
    if current != stack:
        print("BLOCKED: GPU package versions changed unexpectedly.", file=sys.stderr)
        return 2
    code = "import importlib; [importlib.import_module(n) for n in " + repr(IMPORTS) + "]; import torch; assert torch.cuda.is_available(); x=torch.ones(2,device='cuda'); assert float((x+x).sum())==4"
    if subprocess.run([sys.executable, "-c", code], check=False).returncode:
        print("BLOCKED: fresh-process dependency/GPU probe failed.", file=sys.stderr)
        return 2
    freeze = subprocess.run([sys.executable, "-m", "pip", "list", "--format=json"],
                            capture_output=True, text=True, check=True)
    # Exact package pins only, never pip freeze's editable/direct credential-bearing URLs.
    installed = json.loads(freeze.stdout)
    pins = [f"{p['name']}=={p['version']}\n" for p in installed if p["name"] != "neu-esc-reproduction"]
    (args.evidence_dir / "requirements.verified.txt").write_text("".join(sorted(pins)), encoding="utf-8")
    (args.evidence_dir / "environment.after.json").write_text(json.dumps(collect(), indent=2) + "\n", encoding="utf-8")
    print("Verified in a fresh process. Restart the notebook runtime if any installed package was already imported, then rerun capture --require-cuda and CPU fixtures before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
