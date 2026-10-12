"""Explicit real-data CPU verification; uses no pretrained model or benchmark."""

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

from neu_esc.data.common import LABEL_MAPS, atomic_bytes, json_bytes, read_source_csv, sha256_file
from neu_esc.data.dataset import load_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifests/dataset.json")
    parser.add_argument("--output", default="runs/smoke/data/verification.json")
    args = parser.parse_args()
    path = Path(args.manifest).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected_counts = {"train": 23048, "validation": 3305, "test": 6613}
    assert manifest["row_counts"] == expected_counts
    assert manifest["label_maps"] == LABEL_MAPS
    all_ids, checks = set(), {}
    for split in expected_counts:
        raw_file = path.parent / manifest["raw_files"][split]
        assert sha256_file(raw_file) == manifest["raw_sha256"][split]
        original = read_source_csv(raw_file, split, manifest["raw_sha256"][split])
        raw = load_split(path, split, "raw")
        segmented = load_split(path, split, "segmented")
        assert len(original) == len(raw) == len(segmented) == expected_counts[split]
        for source, plain, seg in zip(original, raw, segmented):
            assert source["text"] == plain["text"] == seg["text_raw"]
            assert source["sample_id"] == plain["sample_id"] == seg["sample_id"]
            assert source["source_row"] == plain["source_row"] == seg["source_row"]
            assert source["sentiment"] == plain["sentiment"] == seg["sentiment"]
            assert source["topic"] == plain["topic"] == seg["topic"]
            assert plain["split"] == seg["split"] == split
            assert plain["text_segmented"] == seg["text"]
            assert isinstance(seg["text"], str) and seg["text"].strip()
            assert source["sample_id"] not in all_ids
            all_ids.add(source["sample_id"])
        checks[split] = {
            "rows": len(raw), "raw_sha256": sha256_file(raw_file),
            "processed_sha256": manifest["processed_sha256"][split],
            "first_sample_id": raw[0]["sample_id"], "last_sample_id": raw[-1]["sample_id"],
            "source_text_labels_ids_order_unchanged": True, "all_segmented_nonempty": True,
        }
    result = {
        "schema_version": 1, "status": "PASS", "benchmark_results": False,
        "dataset_manifest_digest": manifest["manifest_digest"],
        "n_unique_ids": len(all_ids), "splits": checks,
        "python": sys.version, "platform": platform.platform(),
        "segmentation": manifest["preprocessing"]["segmentation"],
        "limitations": ["HF/paper content equivalence unverified", "Linux/Colab not executed in this verification"],
    }
    atomic_bytes(args.output, json_bytes(result))
    print(json.dumps({"status": result["status"], "n_unique_ids": len(all_ids), "manifest_digest": manifest["manifest_digest"]}))


if __name__ == "__main__":
    main()
