"""Pinned VnCoreNLP/RDRSegmenter with a portable Java source launcher (JDK 11+)."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import queue
import shutil
import subprocess
import threading
from pathlib import Path

from ._java_bridge import BRIDGE_SOURCE
from .common import DataError, atomic_bytes, digest, fetch_locked


def ensure_resources(root, lock_path, download=False):
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    if lock.get("tool") != "VnCoreNLP" or lock.get("version") != "1.2":
        raise DataError("Unsupported segmentation tool lock")
    directory = (Path(root) / lock["directory"]).resolve()
    for name, source in lock["files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise DataError("Invalid segmentation resource path")
        fetch_locked(source["url"], directory / name, source["sha256"], download=download)
    return lock, directory


class VnCoreSegmenter:
    def __init__(self, root, lock_path, timeout=120):
        self.lock, directory = ensure_resources(root, lock_path, download=False)
        # Materialize embedded source beside the model cache, making installed
        # wheels independent of setuptools package-data settings.
        bridge = directory / ".neu_esc_bridge" / "WsegBridge.java"
        bridge_bytes = BRIDGE_SOURCE.encode("utf-8")
        if bridge.exists() and bridge.read_bytes() != bridge_bytes:
            raise DataError("Unexpected modified Java bridge in resource cache; preserve and investigate")
        if not bridge.exists():
            atomic_bytes(bridge, bridge_bytes)
        java = os.environ.get("NEU_ESC_JAVA") or shutil.which("java")
        if not java:
            raise DataError("Segmentation requires JDK 11+; set NEU_ESC_JAVA or put java on PATH. No raw fallback.")
        self.timeout = timeout
        self.process = subprocess.Popen(
            [java, "-Xmx1g", "-Dfile.encoding=UTF-8", "--class-path", str(directory / "VnCoreNLP-1.2.jar"), str(bridge)],
            cwd=directory, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1,
        )
        self.responses = queue.Queue()
        self.errors = []

        def stdout_reader():
            for line in self.process.stdout:
                self.responses.put(line.rstrip("\r\n"))
            self.responses.put(None)

        def stderr_reader():
            for line in self.process.stderr:
                self.errors.append(line.rstrip())
                if len(self.errors) > 30:
                    self.errors.pop(0)

        self.threads = [threading.Thread(target=stdout_reader, daemon=True), threading.Thread(target=stderr_reader, daemon=True)]
        for thread in self.threads:
            thread.start()
        try:
            self._response("NEU_READY")
        except Exception:
            self.close()
            raise
        self.metadata = {
            "enabled": True, "backend": "vncorenlp", "tool": "VnCoreNLP/RDRSegmenter", "version": "1.2",
            "source_revision": self.lock["revision"], "resources_lock_digest": digest(self.lock),
            "resource_sha256": {name: source["sha256"] for name, source in self.lock["files"].items()},
            "bridge_sha256": hashlib.sha256(bridge_bytes).hexdigest(), "annotators": ["wseg"],
            "sentence_join": "single ASCII space; no split/sample changes",
            "normalization": "VnCoreNLP tokenizer/wordsegmenter internals; no extra Python normalization",
            "original_paper_tool_equivalence": "unverified",
        }

    def _response(self, prefix):
        # Only toolkit protocol messages are accepted, startup logs cannot become data.
        for _ in range(100):
            try:
                line = self.responses.get(timeout=self.timeout)
            except queue.Empty as error:
                raise DataError("Segmentation JVM timeout; " + "\n".join(self.errors)) from error
            if line is None:
                raise DataError("Segmentation JVM terminated; " + "\n".join(self.errors))
            if line == prefix or line.startswith(prefix + "\t"):
                return line
        raise DataError("Excessive unexpected segmentation JVM output")

    def segment(self, text):
        try:
            self.process.stdin.write(base64.b64encode(text.encode("utf-8")).decode("ascii") + "\n")
            self.process.stdin.flush()
            line = self._response("NEU_WSEG")
            result = base64.b64decode(line.split("\t", 1)[1], validate=True).decode("utf-8")
        except (BrokenPipeError, IndexError, UnicodeError) as error:
            raise DataError("Segmentation protocol failed") from error
        if not result.strip():
            raise DataError("Segmenter produced empty output; sample cannot be dropped")
        return result

    def close(self):
        if self.process.stdin:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)
        for thread in self.threads:
            thread.join(timeout=5)
        for stream in (self.process.stdout, self.process.stderr):
            stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
