"""Resolve CPU metadata, then emit explicit hash-locked CPU wheel URLs for pip."""

from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    # The URLs/hashes in the committed input were read from the official CPU index.
    source = (ROOT / "requirements-cpu.in").read_text(encoding="utf-8")
    urls = [line for line in source.splitlines() if line.startswith("torch @ ")]
    versions = {re.search(r"torch-([^/]+?)%2Bcpu-", line).group(1) for line in urls}
    if len(urls) != 8 or len(versions) != 1:
        raise ValueError("Review all 8 Windows/Linux CPython 3.10-3.13 CPU wheels first")
    version = versions.pop() + "+cpu"
    with tempfile.TemporaryDirectory(prefix="neu-esc-cpu-lock-") as folder:
        temporary = Path(folder)
        req = temporary / "metadata.in"
        req.write_text(f"setuptools>=68\nwheel\ntorch=={version}\n", encoding="utf-8")
        output = temporary / "resolved.lock"
        command = ["uv", "pip", "compile", str(ROOT / "pyproject.toml"), str(req),
                   "--extra", "data", "--extra", "train", "--extra", "eval", "--extra", "dev",
                   "--universal", "--python-version", "3.10", "--torch-backend", "cpu",
                   "--generate-hashes", "--emit-index-url", "--output-file", str(output),
                   "--custom-compile-command", "python scripts/environment/lock_cpu.py", "--quiet"]
        subprocess.run(command, cwd=ROOT, check=True)
        lock = output.read_text(encoding="utf-8")
    wheel_lines = []
    for requirement in urls:
        url_and_hash, marker = requirement.removeprefix("torch @ ").split(" ; ", 1)
        url, digest = url_and_hash.split("#sha256=", 1)
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Invalid CPU wheel digest")
        wheel_lines.append(f"torch @ {url} ; {marker} \\\n    --hash=sha256:{digest}\n")
    pattern = r"^torch==[^\n]+\n(?:[ \t].*\n|\n)*"
    lock, replaced = re.subn(pattern, lambda _: "".join(wheel_lines), lock, count=1, flags=re.MULTILINE)
    if replaced != 1:
        raise ValueError("Unexpected resolver output; refuse to emit a partial CPU lock")
    if "--extra-index-url" in lock:
        raise ValueError("Do not expose the CPU index to unrelated packages")
    # Avoid machine-specific temp paths in annotations.
    lock = lock.replace(str(req), "CPU metadata input").replace(req.as_posix(), "CPU metadata input")
    (ROOT / "requirements-cpu.lock").write_text(lock, encoding="utf-8", newline="\n")
    print("CPU lock emitted from resolved metadata and audited CPU wheel hashes.")


if __name__ == "__main__":
    main()
