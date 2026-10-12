"""The data contract consumed by training, prediction, evaluation and LLMs."""

from __future__ import annotations

import json
from pathlib import Path

from .common import DataError, ID_ALGORITHM, LABEL_MAPS, SPLITS, manifest_digest, sha256_file


def load_split(manifest_path, split, text_variant="raw"):
    """Return a list of records; ``text`` is the explicitly selected variant.

    IDs, source rows, labels, ``text_raw`` and ``text_segmented`` remain available.
    Verify manifest/file hashes and full row integrity before returning any record.
    Missing segmentation raises DataError, never falls back to unsegmented text.
    File paths in the manifest are relative to its containing directory.
    """
    if split not in SPLITS:
        raise DataError(f"Unknown split {split!r}; expected {SPLITS}")
    if text_variant not in ("raw", "segmented"):
        raise DataError(f"Unknown text_variant {text_variant!r}")
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise DataError("Unsupported manifest schema_version")
    if manifest_digest(manifest) != manifest.get("manifest_digest"):
        raise DataError("Manifest digest mismatch")
    if manifest.get("label_maps") != LABEL_MAPS or manifest.get("id_algorithm") != ID_ALGORITHM:
        raise DataError("Manifest label mapping or ID algorithm conflicts with contract")
    filename = (path.parent / manifest["processed_files"][split]).resolve()
    if sha256_file(filename) != manifest["processed_sha256"][split]:
        raise DataError(f"Processed bytes hash mismatch: {filename}")
    records, seen = [], set()
    with filename.open(encoding="utf-8", errors="strict") as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            expected_id = f"{split}:{index:06d}:{manifest['raw_sha256'][split][:12]}"
            if row.get("sample_id") != expected_id or row.get("sample_id") in seen:
                raise DataError(f"Invalid, reordered or duplicate ID at {split}:{index}")
            if type(row.get("source_row")) is not int or row["source_row"] != index or row.get("split") != split:
                raise DataError(f"Invalid split/source_row at {split}:{index}")
            for task in LABEL_MAPS:
                label = row.get(task)
                if type(label) is not int or str(label) not in LABEL_MAPS[task]:
                    raise DataError(f"Invalid {task} label at {expected_id}")
            raw = row.get("text")
            if not isinstance(raw, str) or not raw.strip():
                raise DataError(f"Missing raw text at {expected_id}")
            selected = raw if text_variant == "raw" else row.get("text_segmented")
            if not isinstance(selected, str) or not selected.strip():
                raise DataError(f"Missing {text_variant} text at {expected_id}; PhoBERT segmentation is blocked")
            seen.add(row["sample_id"])
            records.append({**row, "text_raw": raw, "text": selected})
    if len(records) != manifest["row_counts"][split]:
        raise DataError(f"Row count mismatch in {split}")
    return records
