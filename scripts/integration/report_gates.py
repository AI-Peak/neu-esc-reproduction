"""Materialize gates from completed receipts, never infer GPU/benchmark success."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def git(*args):
    result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
    return dict(argv=["git", *args], returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e2e", default="runs/smoke/integration/end-to-end-001/receipt.json")
    parser.add_argument("--install", default="runs/smoke/integration/clean-install-001/receipt.json")
    args = parser.parse_args()
    e2e, install = read(args.e2e), read(args.install)
    assert e2e["status"] == install["status"] == "PASS"
    dataset = read("runs/smoke/integration/real-data-verification.json")
    manifest = read("data/manifests/dataset.json")
    assert dataset["status"] == "PASS" and dataset["dataset_manifest_digest"] == manifest["manifest_digest"]
    tests = []
    for path in ("runs/smoke/integration/module-tests.xml", "runs/smoke/integration/integration-tests.xml"):
        node = ET.parse(ROOT / path).getroot()
        suites = list(node.iter("testsuite"))
        counts = {key: sum(int(n.attrib.get(key, 0)) for n in suites) for key in ("tests", "errors", "failures", "skipped")}
        assert counts["errors"] == counts["failures"] == 0
        tests.append(dict(path=path, sha256=digest(ROOT / path), **counts))
    executable = sorted({*list((ROOT / "src").rglob("*.py")),
                         *list((ROOT / "configs").rglob("*.yaml")),
                         *list((ROOT / "tests").rglob("*.py")),
                         *list((ROOT / "scripts").rglob("*.py")),
                         ROOT / "pyproject.toml", ROOT / "requirements-cpu.lock",
                         ROOT / "uv.lock", ROOT / "data/manifests/dataset.json"})
    hashes = {path.relative_to(ROOT).as_posix(): digest(path) for path in executable}
    inventory = read("reports/method/delivery_inventory.json")
    mismatches = [name for name, expected in inventory["files"].items()
                  if not (ROOT / name).is_file() or digest(ROOT / name) != expected]
    assert not mismatches, mismatches
    head = git("rev-parse", "HEAD")["stdout"].strip()
    tree = git("ls-tree", "-r", "--name-only", "HEAD")
    tracked = set(tree["stdout"].splitlines())
    critical = [name for name in hashes if name.startswith(("src/neu_esc/", "configs/"))]
    missing = [name for name in critical if name not in tracked]
    release_status = git("status", "--porcelain", "--untracked-files=all")
    evidence = dict(schema_version="1.0", checked_at_utc=datetime.now(timezone.utc).isoformat(),
                    head=head, git=release_status, missing_from_head=missing, file_sha256=hashes,
                    dataset_manifest_digest=manifest["manifest_digest"],
                    module02_inventory_sha256=digest(ROOT / "reports/method/delivery_inventory.json"),
                    module02_inventory_mismatches=mismatches)
    report_dir = ROOT / "reports/integration"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "input_snapshot.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8", newline="\n")
    validation = e2e["variants"]["mtl_smart"]["validation_aggregation"]
    assert validation == "aggregation requires nonempty method_mode"
    artifacts = read("runs/smoke/integration/end-to-end-001/validation-artifact-checks.json")
    assert all(all(check.values()) for check in artifacts.values())
    regeneration = read("runs/smoke/integration/regeneration-check.json")
    assert regeneration["status"] == "PASS" and all(regeneration["matching_artifacts"].values())
    planned = []
    for path in sorted((ROOT / "configs/experiments").glob("*/*.yaml")):
        import yaml
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        planned.append(dict(config=path.relative_to(ROOT).as_posix(), backbone=config["backbone"],
                            variant=config["variant"], seeds=[42, 43, 44, 45, 46]
                            if config["backbone"] in ("visobert", "phobert") else [42]))
    assert len(planned) == 22 and sum(len(p["seeds"]) for p in planned) == 70
    observed = []
    successful = set()
    for path in sorted((ROOT / "runs/encoder").rglob("provenance.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        eligible = (value.get("run_status") == "COMPLETED" and value.get("training_mode") == "full"
                    and value.get("benchmark_eligible") is True)
        key = (value.get("cohort_id"), value.get("seed"))
        complete_artifacts = all((path.parent / name).is_file() for name in
                                 ("predictions/test.jsonl", "evaluation/test/metrics.json", "checkpoints/best/COMPLETE.json"))
        if eligible and complete_artifacts and all(v is not None for v in key):
            successful.add(key)
        observed.append(dict(path=path.relative_to(ROOT).as_posix(), run_status=value.get("run_status"),
                             cohort_id=key[0], seed=key[1], eligible=eligible,
                             complete_test_artifacts=complete_artifacts))
    assert not successful, "Benchmark outputs arrived: audit their predictions/cohorts before issuing the old zero-run gate"
    ledger = dict(schema_version="1.0", llm="OUT_OF_SCOPE", planned_configs=planned,
                  planned_config_count=22, planned_successful_encoder_runs=70,
                  observed_encoder_runs=observed, completed_unique_cohort_seed_with_artifacts=len(successful),
                  smoke_or_retries_counted=False)
    (report_dir / "campaign_ledger.json").write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8", newline="\n")
    errors = [dict(id="I06-01", owner_module="04", severity="P1", blocks="COLAB_RELEASE_READY",
                   expected="Pinned clean commit includes READY data/training/evaluator/configs/integration files",
                   actual=f"HEAD {head} contains infrastructure only; {len(missing)} executable/config files absent",
                   reproduction="git show HEAD:src/neu_esc/training/train.py"),
              dict(id="I06-02", owner_module="02", severity="P2", blocks="VALIDATION_REPORT_READY",
                   expected="Persist validation metrics with method_mode/backbone/variant matching resolved config",
                   actual=validation,
                   reproduction="Read persisted validation metrics then call aggregate_cohort([metrics], expected_seeds=[42])")]
    gates = dict(schema_version="1.0", status="PARTIAL", checked_at_utc=evidence["checked_at_utc"],
                 scope=dict(priority="visobert/mtl_smart: sentiment+topic, paper_method", llm="OUT_OF_SCOPE",
                            llm_missing_is_blocker=False, encoder_configs=22, planned_successful_encoder_runs=70,
                            full_training_executed=False),
                 inputs=dict(head=head, clean=False, manifest_digest=manifest["manifest_digest"],
                             snapshot="reports/integration/input_snapshot.json",
                             snapshot_sha256=digest(report_dir / "input_snapshot.json")),
                 tests_actually_run=tests,
                 receipts=[dict(path=p, sha256=digest(ROOT / p)) for p in
                           (args.e2e, args.install, "runs/smoke/integration/real-data-verification.json",
                            "runs/smoke/integration/end-to-end-001/validation-artifact-checks.json",
                            "runs/smoke/integration/regeneration-check.json")],
                 gates={
                     "DATA_TECHNICAL_READY": dict(status="PASS", rows=manifest["row_counts"], unique_ids=dataset["n_unique_ids"], notice="Pinned GitHub alternative revision; HF/paper equivalence unverified"),
                     "EXACT_PAPER_DATA_EQUIVALENCE": dict(status="UNVERIFIED", blocks_technical_p0=False),
                     "P0_READY": dict(status="CPU_PASS/GPU_PENDING", cpu_pipeline="PASS", colab_release="BLOCKED", full_pretrained_gpu_backward="PENDING", errors=["I06-01"], validation_report_issue="I06-02"),
                     "FULL_VARIANTS_READY": dict(status="CPU_PASS/GPU_PENDING", configs=22, tiny_cpu_variants=list(e2e["variants"]), full_pretrained_gpu_variants="PENDING"),
                     "CLEAN_INSTALL_CPU": dict(status="PASS", source="Wheel built from current uncommitted bytes; fresh isolated venv and hash-locked CPU dependencies", release_verified=False),
                     "ID_LABEL_ALIGNMENT": dict(status="PASS", real_rows_checked=32966, real_test_diagnostic_count_per_task=6613, diagnostic_predictions_are_benchmark=False, permutation_and_negative_cli="PASS"),
                     "RESUME_CPU": dict(status="PASS", optimizer_boundary="PASS", exact_six_variant_state_comparison="PASS (module test receipt)", stale_seed_revision_manifest="REJECTED", missing_marker_corrupt_state="REJECTED", gpu="PENDING"),
                     "VALIDATION_REPORT_READY": dict(status="FAIL", errors=["I06-02"], checkpoint_selection_metric_affected=False, test_cli_report_affected=False),
                     "SAVED_ARTIFACT_REGENERATION": dict(status="PASS", p0_test_artifacts_byte_identical=12, validation_variants_recomputed=6, benchmark_eligible=False),
                     "COLAB_RELEASE_READY": dict(status="BLOCKED", errors=["I06-01"], commit_missing_files=missing),
                     "COLAB_RUNTIME_READY": dict(status="GPU_PENDING", notebook_present=(ROOT / "notebooks/colab_reproduce.ipynb").exists(), gpu_verified=False, drive_sync_verified=False, throughput_verified=False, owner_module="07"),
                     "LLM_READY": dict(status="NOT_APPLICABLE", reason="User removed LLM benchmark; module 05 not required"),
                     "BENCHMARK_REPORT_READY": dict(status="PENDING", successful_encoder_runs=len(successful), target=70, ledger="reports/integration/campaign_ledger.json", smoke_and_retries_counted=False, owner_module="07/08")},
                 errors=errors,
                 limitations=["No actual Colab runtime or GPU output supplied", "Tiny initialized BERT checks do not demonstrate full pretrained ViSoBERT performance or batch8/length128 VRAM fit", "No benchmark metrics produced", "No files owned by modules 01-04 were edited, staged or committed"])
    (report_dir / "gates.json").write_text(json.dumps(gates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": gates["status"], "errors": [e["id"] for e in errors], "head": head}))


if __name__ == "__main__":
    main()
