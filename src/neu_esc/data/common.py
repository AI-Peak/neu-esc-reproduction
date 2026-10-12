"""Shared serialization, identity, label and source validation primitives."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
import urllib.request
from decimal import Decimal, InvalidOperation
from pathlib import Path

SCHEMA_VERSION = 1
PREPROCESSING_VERSION = "neu-esc-data-v1.0.0"
SPLITS = ("train", "validation", "test")
LABEL_MAPS = {
    "sentiment": {"0": "Neutral", "1": "Positive", "2": "Negative", "3": "Toxic"},
    "topic": {
        "0": "Spam", "1": "News", "2": "Academic", "3": "Other", "4": "Service",
        "5": "Jobs & Recruitment", "6": "Personal Affairs", "7": "Social Affairs",
        "8": "Help & Share", "9": "Club & Events",
    },
}
ID_ALGORITHM = "{split}:{source_row:06d}:{raw_sha256[:12]}"


class DataError(ValueError):
    """A data/provenance gate failed; never silently repair input."""


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def manifest_digest(manifest):
    return digest({key: value for key, value in manifest.items() if key != "manifest_digest"})


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def relative_path(path, base):
    return Path(os.path.relpath(Path(path).resolve(), Path(base).resolve())).as_posix()


def atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if Path(name).exists():
            Path(name).unlink()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def fetch_locked(url, destination, expected_sha256, download=True):
    """Reuse verified bytes; an existing corrupt source is never overwritten."""
    destination = Path(destination)
    if destination.exists():
        actual = sha256_file(destination)
        if actual != expected_sha256:
            raise DataError(f"Source hash mismatch at {destination}: {actual} != {expected_sha256}. Investigate; file preserved.")
        return destination
    if not download:
        raise DataError(f"Missing source {destination}; offline mode forbids downloading")
    request = urllib.request.Request(url, headers={"User-Agent": "neu-esc-reproduction-data/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        body = response.read()
    actual = hashlib.sha256(body).hexdigest()
    if actual != expected_sha256:
        raise DataError(f"Downloaded source hash mismatch for {url}: {actual} != {expected_sha256}")
    # Do not replace another process's newly created source.
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return fetch_locked(url, destination, expected_sha256, download=False)
    atomic_bytes(destination, body)
    return destination


def label_id(value, task, context):
    if value is None or isinstance(value, bool) or str(value).strip() == "":
        raise DataError(f"{context}: null {task} label")
    try:
        number = Decimal(str(value))
    except InvalidOperation as error:
        raise DataError(f"{context}: nonnumeric {task} label {value!r}") from error
    if not number.is_finite() or number != number.to_integral_value() or not 0 <= number < len(LABEL_MAPS[task]):
        raise DataError(f"{context}: invalid {task} label {value!r}")
    return int(number)


def read_source_csv(path, split, raw_hash):
    """UTF-8 strict, preserve source text and every CSV record in original order."""
    records = []
    try:
        with Path(path).open("r", encoding="utf-8-sig", errors="strict", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            if reader.fieldnames != ["text", "sentiment", "classification"]:
                raise DataError(f"{path}: expected columns text,sentiment,classification, got {reader.fieldnames}")
            for row_number, row in enumerate(reader):
                context = f"{split}:{row_number}"
                if None in row or any(value is None for value in row.values()):
                    raise DataError(f"{context}: malformed/missing CSV field")
                text = row["text"]
                if not text.strip() or text.strip().lower() in {"nan", "null", "none"}:
                    raise DataError(f"{context}: null/empty text")
                records.append({
                    "sample_id": f"{split}:{row_number:06d}:{raw_hash[:12]}",
                    "split": split, "source_row": row_number, "text": text,
                    "text_segmented": None,
                    "sentiment": label_id(row["sentiment"], "sentiment", context),
                    "topic": label_id(row["classification"], "topic", context),
                })
    except (UnicodeDecodeError, csv.Error) as error:
        raise DataError(f"{path}: invalid UTF-8/CSV: {error}") from error
    return records


def read_config(path):
    import yaml

    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != SCHEMA_VERSION:
        raise DataError("Unsupported data config schema_version")
    if config.get("label_maps") != LABEL_MAPS:
        raise DataError("Label mapping conflicts with audited contract; stop data gate")
    if set(config.get("sources", {})) != set(SPLITS):
        raise DataError("Config must contain train, validation and test sources")
    root = (path.parent / config.get("project_root", "../..")).resolve()
    if config.get("preprocessing", {}).get("cleaning") != "identity":
        raise DataError("Only identity cleaning is supported: source CSV is already curated")
    if config["preprocessing"].get("version") != PREPROCESSING_VERSION:
        raise DataError("Unrecognized preprocessing version")
    return config, root
