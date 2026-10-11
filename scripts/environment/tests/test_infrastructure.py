"""CPU tests for provenance and collection safeguards; no benchmark claims."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/environment" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_capture_keeps_tokens_and_env_out(monkeypatch):
    secret = "fixture-secret-never-log"
    monkeypatch.setenv("HF_TOKEN", secret)
    monkeypatch.setenv("GITHUB_TOKEN", secret)
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    info = load("capture").collect()
    assert secret not in json.dumps(info)
    assert info["schema_version"] == 1
    assert isinstance(info["packages"], dict)
    assert "python" in info and "torch" in info


def test_capture_refuses_overwrite(tmp_path):
    path = tmp_path / "environment.json"
    path.write_text("existing-run-evidence", encoding="utf-8")
    result = subprocess.run([sys.executable, str(ROOT / "scripts/environment/capture.py"),
                             "--output", str(path)], capture_output=True, text=True)
    assert result.returncode != 0
    assert path.read_text(encoding="utf-8") == "existing-run-evidence"


def test_external_markers_are_detected_before_import(tmp_path):
    path = tmp_path / "test_external.py"
    path.write_text("import pytest\npytestmark = pytest.mark.paid_api\nraise RuntimeError('must not import')\n", encoding="utf-8")
    assert load("offline_tests").has_external_marker(path)
    path.write_text("def test_cpu():\n    assert True\n", encoding="utf-8")
    assert not load("offline_tests").has_external_marker(path)


def test_gitignore_preserves_manifests_and_blocks_sensitive_files(tmp_path):
    # check-ignore works for nonexistent files without touching another module's files.
    ignored = ["data/raw/train.csv", "data/processed/test.jsonl", "runs/encoder/x/environment.json",
               ".env", "drive_credentials.json", "checkpoints/best/model.safetensors"]
    visible = ["data/manifests/dataset.json", "tests/data/fixtures/synthetic.jsonl", "uv.lock"]
    for path in ignored:
        result = subprocess.run(["git", "check-ignore", "--no-index", "--quiet", path], cwd=ROOT)
        assert result.returncode == 0, path
    for path in visible:
        result = subprocess.run(["git", "check-ignore", "--no-index", "--quiet", path], cwd=ROOT)
        assert result.returncode == 1, path


def test_colab_recipe_blocks_unverified_gpu_before_any_install(monkeypatch, tmp_path):
    capture = load("capture")
    monkeypatch.setitem(sys.modules, "capture", capture)
    installer = load("colab_install")
    monkeypatch.setattr(installer, "collect", lambda: {
        "platform": {"system": "Linux"}, "torch": {"cuda_available": False}})
    evidence = tmp_path / "no-gpu"
    monkeypatch.setattr(sys, "argv", ["colab_install", "--evidence-dir", str(evidence), "--apply"])
    assert installer.main() == 2
    assert not evidence.exists()


def test_colab_recipe_rejects_any_gpu_stack_mutation(monkeypatch, tmp_path):
    capture = load("capture")
    monkeypatch.setitem(sys.modules, "capture", capture)
    installer = load("colab_install")
    before = {"platform": {"system": "Linux"}, "torch": {"cuda_available": True},
              "git": {"commit": "a" * 40, "clean": True}, "packages": {"torch": "fixture-version"}}
    monkeypatch.setattr(installer, "collect", lambda: before)
    calls = []

    def dry_run_only(cmd, **kwargs):
        calls.append(cmd)
        assert "--dry-run" in cmd
        return subprocess.CompletedProcess(cmd, 0, json.dumps({"install": [
            {"metadata": {"name": "nvidia-cublas-cu99", "version": "fixture-version"}}]}), "")

    monkeypatch.setattr(installer.subprocess, "run", dry_run_only)
    evidence = tmp_path / "mutation"
    monkeypatch.setattr(sys, "argv", ["colab_install", "--evidence-dir", str(evidence), "--apply"])
    assert installer.main() == 2
    assert len(calls) == 1
    assert (evidence / "install-plan.json").is_file()


def test_synthetic_manifest_digest_and_byte_hashes():
    import hashlib

    fixtures = ROOT / "scripts/environment/fixtures"
    manifest = json.loads((fixtures / "manifest.example.json").read_text(encoding="utf-8"))
    digest = manifest.pop("manifest_digest")
    canonical = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(canonical).hexdigest() == digest
    assert hashlib.sha256((fixtures / "synthetic_raw.csv").read_bytes()).hexdigest() == manifest["raw_sha256"]["test"]
    assert hashlib.sha256((fixtures / "synthetic_test.jsonl").read_bytes()).hexdigest() == manifest["processed_sha256"]["test"]
    assert manifest["benchmark_eligible"] is False
