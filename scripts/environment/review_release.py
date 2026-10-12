"""Audit explicitly handed-off READY paths; never stage files or modify owners' output."""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8-sig"))


def review():
    hand01 = (ROOT / "handoffs/01-data.md").read_text(encoding="utf-8-sig")
    assert "Status: READY" in hand01
    data = re.findall(r"^- `([^`]+)`", hand01.split("## Allowlist READY cho module 04")[1], re.M)
    train = (ROOT / "reports/method/02-ready-allowlist.txt").read_text(encoding="utf-8-sig").splitlines()
    inventory02 = read("reports/method/delivery_inventory.json")
    assert inventory02["status"] == "READY"
    evaluation = read("runs/smoke/evaluation/verification.json")
    assert evaluation["status"] == "READY"
    evaluator = list(evaluation["owned_release_allowlist"])
    if "handoffs/03-evaluation.md" not in evaluator:
        evaluator.append("handoffs/03-evaluation.md")
    integration = [
        "scripts/integration/audit_pipeline.py", "scripts/integration/clean_install.py",
        "scripts/integration/report_gates.py", "scripts/integration/reproduce_validation_metadata.py",
        "tests/integration/test_boundaries.py", "docs/integration.md",
        "reports/integration/gates.json", "reports/integration/input_snapshot.json",
        "reports/integration/campaign_ledger.json", "reports/integration/fix-prompts.md",
        "handoffs/06-integration.md",
    ]
    snapshot06 = read("reports/integration/input_snapshot.json")["file_sha256"]
    ledger01 = read("reports/source/source_ledger.json")
    expected01 = dict(snapshot06)
    expected01.update({"reports/source/" + p: h for p, h in ledger01["evidence_sha256"].items()})
    expected03 = dict(evaluation["owned_sha256"])
    expected03["handoffs/03-evaluation.md"] = evaluation["handoff_sha256"]
    groups = {"01": (data, expected01), "02": (train, inventory02["files"]),
              "03": (evaluator, expected03), "06": (integration, snapshot06)}
    release = {}
    excluded = {}
    for owner, (paths, hashes) in groups.items():
        for name in paths:
            path = ROOT / name
            if not path.is_file() or path.is_symlink() or ROOT not in path.resolve().parents:
                raise ValueError(f"Unsafe or missing READY path: {name}")
            actual = digest(path)
            if name in hashes and actual != hashes[name]:
                raise ValueError(f"Owner hash mismatch: {name}")
            if "/tokenizer/" in name:
                excluded[name] = {"owner": owner, "sha256": actual,
                                  "reason": "downloaded tokenizer resource/cache; user excludes resources"}
                continue
            if re.match(r"(?:data/(?:raw|processed|resources|tools)|runs|checkpoints|secrets)/", name):
                raise ValueError(f"Forbidden READY path: {name}")
            content = path.read_bytes()
            if re.search(rb"(?:gh[pousr]_|hf_)[A-Za-z0-9]{24,}|-----BEGIN (?:RSA )?PRIVATE KEY-----|X-Amz-Signature=", content):
                raise ValueError(f"Credential-like bytes: {name}")
            if path.suffix == ".py":
                ast.parse(content.decode("utf-8-sig"), filename=name)
            if path.suffix == ".json":
                json.loads(content.decode("utf-8-sig"))
            if path.suffix in (".yaml", ".yml"):
                import yaml
                yaml.safe_load(content.decode("utf-8-sig"))
            if path.suffix == ".zip":
                with zipfile.ZipFile(path) as archive:
                    if any(not p.endswith(".py") for p in archive.namelist()):
                        raise ValueError(f"Only reviewed code-only reference archive is allowed: {name}")
            ignored = subprocess.run(["git", "check-ignore", "--quiet", "--no-index", name], cwd=ROOT)
            if ignored.returncode != 1:
                raise ValueError(f"READY path is ignored: {name}")
            release[name] = {"owner": owner, "sha256": actual,
                             "evidence": "handed_off_hash" if name in hashes else "explicit_READY_path_reviewed_and_frozen_by_04"}
    fix = ROOT / "reports/method/i06-02-verification.json"
    assert digest(fix) == "c20f24dbe2f95bdfe0539e2901eaba3a7fb1f97178dc1d5214a3c54570c249d4"
    receipt = read("reports/method/i06-02-verification.json")
    assert receipt["status"] == "PASS" and receipt["protected_files_unchanged"] and receipt["old_integration_artifacts_unchanged"]
    for case in receipt["variants"].values():
        smoke = case["smoke"]
        assert smoke["status"] == "PASS" and case["reproduction"]["exit_code"] == 0
        assert digest(ROOT / smoke["base"] / "run/evaluation/validation/metrics.json") == smoke["validation_metrics_sha256"]
    manifest = read("data/manifests/dataset.json")
    canonical = json.dumps({k: v for k, v in manifest.items() if k != "manifest_digest"},
                           sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    assert hashlib.sha256(canonical).hexdigest() == manifest["manifest_digest"] == inventory02["dataset_manifest_digest"]
    for name, expected in manifest["preprocessing"]["implementation_sha256"].items():
        assert digest(ROOT / "src/neu_esc/data" / name) == expected
    return {
        "schema_version": 1, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "READY_FOR_RELEASE_CANDIDATE", "I06-02": "FIXED_AND_RECHECKED",
        "scope": "ENCODER_ONLY_LLM_OUT_OF_SCOPE", "gpu": "PENDING", "benchmark_runs": 0,
        "dataset_manifest_digest": manifest["manifest_digest"],
        "dataset_equivalence": manifest["paper_equivalence_status"],
        "ready_paths": dict(sorted(release.items())), "excluded_ready_cache_paths": excluded,
        "historical_reports": "06 gates/input_snapshot predate I06-02 fix; preserve them as historical evidence, not current release certification",
        "reference_code_zip": "9 Python source members only; no data, model weights or checkpoint",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = review()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(f"Reviewed {len(result['ready_paths'])} READY module paths; excluded {len(result['excluded_ready_cache_paths'])} cached resources")


if __name__ == "__main__":
    main()
