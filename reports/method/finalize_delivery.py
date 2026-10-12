"""Verify final data/config gate and list ONLY module 02-owned READY files."""
import json
from pathlib import Path
import sys
import zipfile
import hashlib

from neu_esc.training.config import canonical_digest, file_digest, load_config
from neu_esc.training.runtime import read_manifest


def main():
    root = Path.cwd()
    # Preserve upstream byte hashes before normalizing text for the LF Git snapshot.
    reference = root / "reports/method/reference"
    archive = root / "reports/method/reference-source.zip"
    ledger = reference / "upstream_hashes.json"
    if not archive.exists():
        sources = sorted(reference.glob("*.py"))
        source_hashes = {p.name.replace("__", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sources}
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as container:
            for path in sources:
                container.writestr(path.name.replace("__", "/"), path.read_bytes())
        ledger.write_text(json.dumps({"commit": "b5637cc0a8a7b5e88e794d180c11902405548eb5",
                                      "source_sha256": source_hashes,
                                      "archive": "reports/method/reference-source.zip",
                                      "python_copies": "LF normalized for portable Git checkout"}, indent=2) + "\n", encoding="utf-8")
    configs = sorted(root.glob("configs/experiments/*/*.yaml"))
    assert len(configs) == 22
    config_hashes = {}
    digests = set()
    for path in configs:
        config = load_config(path)
        _, digest = read_manifest(config)
        digests.add(digest)
        config_hashes[path.relative_to(root).as_posix()] = canonical_digest(config)
    assert len(digests) == 1
    source_directories = ["src/neu_esc/models", "src/neu_esc/losses", "src/neu_esc/training",
                          "configs/models", "configs/experiments", "tests/training", "reports/method"]
    files = set()
    excluded = {"delivery_inventory.json", "02-ready-allowlist.txt"}
    for directory in source_directories:
        files.update(p for p in (root / directory).rglob("*") if p.is_file()
                     and "__pycache__" not in p.parts and not p.name.endswith(".pyc")
                     and p.name not in excluded)
    files.update(root / p for p in ("docs/method_audit.md", "docs/training.md"))
    # .gitattributes uses eol=lf; inventory must describe the checkout bytes too.
    for path in files | {root / "handoffs/02-training.md"}:
        if path.suffix in (".py", ".md", ".json", ".yaml", ".txt"):
            raw = path.read_bytes()
            normalized = raw.replace(b"\r\n", b"\n")
            if raw != normalized:
                path.write_bytes(normalized)
    inventory = {"schema_version": "1.0", "module": "02", "status": "READY",
                 "P0_READY": True, "FULL_VARIANTS_READY": True, "benchmark_runs": 0,
                 "dataset_manifest_digest": next(iter(digests)), "resolved_config_hashes": config_hashes,
                 "files": {p.relative_to(root).as_posix(): file_digest(p) for p in sorted(files)}}
    for p in files:
        assert p.is_file()
    path = root / "reports/method/delivery_inventory.json"
    path.write_bytes((json.dumps(inventory, indent=2) + "\n").encode("utf-8"))
    allowlist = sorted([*inventory["files"], "reports/method/delivery_inventory.json",
                        "reports/method/02-ready-allowlist.txt", "handoffs/02-training.md"])
    (root / "reports/method/02-ready-allowlist.txt").write_bytes(("\n".join(allowlist) + "\n").encode("utf-8"))
    print(json.dumps({"config_count": len(configs), "files_ready": len(allowlist),
                      "manifest_digest": inventory["dataset_manifest_digest"]}))


if __name__ == "__main__":
    main()
