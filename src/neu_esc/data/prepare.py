"""Prepare locked source CSVs into contract JSONL and publish manifest last."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .audit import compare_author_processed, summarize
from .common import (
    DataError, ID_ALGORITHM, LABEL_MAPS, PREPROCESSING_VERSION, SPLITS,
    atomic_bytes, canonical_bytes, digest, fetch_locked, json_bytes,
    manifest_digest, read_config, read_source_csv, relative_path, sha256_file,
)
from .segmentation import VnCoreSegmenter


def prepare(config_path, output=None, manifest_path=None, download=True):
    config, root = read_config(config_path)
    output = Path(output).resolve() if output is not None else root / "data/processed"
    manifest_path = Path(manifest_path).resolve() if manifest_path is not None else root / config["manifest_path"]
    audit_directory = root / config.get("audit_directory", "reports/source")
    # Protect shared namespaces and original data from accidental output collisions.
    raw_paths = {(root / source["path"]).resolve() for source in config["sources"].values()}
    target_paths = {(output / f"{split}.jsonl").resolve() for split in SPLITS} | {manifest_path.resolve()}
    if target_paths & raw_paths or manifest_path.parent.resolve() == output.resolve():
        raise DataError("Output must not collide with raw sources or share the JSONL directory with manifest")
    splits, raw_hashes, source_urls, raw_files = {}, {}, [], {}
    source_ledger = []
    for split in SPLITS:
        source = config["sources"][split]
        raw = fetch_locked(source["url"], root / source["path"], source["sha256"], download)
        raw_hashes[split] = sha256_file(raw)
        splits[split] = read_source_csv(raw, split, raw_hashes[split])
        if len(splits[split]) != source["rows"]:
            raise DataError(f"{split} has {len(splits[split])} records; expected {source['rows']}")
        raw_files[split] = relative_path(raw, manifest_path.parent)
        source_urls.append(source["url"])
        source_ledger.append({"split": split, "role": "unsegmented_curated_source", "path": source["path"],
                              "url": source["url"], "sha256": raw_hashes[split], "rows": len(splits[split])})
    segmentation = config["preprocessing"]["segmentation"]
    backend = segmentation["backend"]
    if backend not in ("none", "vncorenlp"):
        raise DataError("Unknown segmentation backend; no silent raw fallback")
    if backend == "none":
        metadata = {"enabled": False, "backend": "none", "status": "segmented variant blocked"}
    else:
        with VnCoreSegmenter(root, root / segmentation["resources_lock"]) as segmenter:
            metadata = segmenter.metadata
            for split in SPLITS:
                print(f"Segmenting {split}: {len(splits[split])} rows", flush=True)
                for row in splits[split]:
                    # Input is always the locked unsegmented source, never the author _processed CSV.
                    row["text_segmented"] = segmenter.segment(row["text"])
    alignment = {}
    for split, source in config.get("author_processed_sources", {}).items():
        raw = fetch_locked(source["url"], root / source["path"], source["sha256"], download)
        author = read_source_csv(raw, split, source["sha256"])
        alignment[split] = compare_author_processed(splits[split], author)
        if len(author) != len(splits[split]) or alignment[split]["label_mismatch_rows"]:
            raise DataError(f"Author processed CSV does not preserve row/label alignment for {split}; investigate")
        source_ledger.append({"split": split, "role": "author_presegmented_comparison_only", **source})
    payloads = {split: b"".join(canonical_bytes(row) + b"\n" for row in splits[split]) for split in SPLITS}
    manifest = {
        "schema_version": 1,
        "source_urls": source_urls,
        "source_revision": config["source_revision"],
        "source_provider": config["source_provider"],
        "paper_source": config.get("paper_source"),
        "hf_source": config.get("hf_source"),
        "source_license": config.get("source_license", "unverified"),
        "redistribution_status": config.get("redistribution_status", "not_authorized"),
        "raw_sha256": raw_hashes,
        "raw_files": raw_files,
        "processed_sha256": {split: hashlib.sha256(body).hexdigest() for split, body in payloads.items()},
        "processed_files": {split: relative_path(output / f"{split}.jsonl", manifest_path.parent) for split in SPLITS},
        "row_counts": {split: len(rows) for split, rows in splits.items()},
        "id_algorithm": ID_ALGORITHM,
        "label_maps": LABEL_MAPS,
        "preprocessing": {
            "version": PREPROCESSING_VERSION,
            "config_sha256": digest(config),
            "steps": ["decode UTF-8 strictly (strip encoding BOM only)", "identity text cleaning: preserve CSV text",
                      "parse integral numeric sentiment/classification labels; classification -> topic",
                      "derive sample ID from split, zero-based CSV row and raw SHA-256 prefix",
                      "optional pinned VnCoreNLP word/sentence segmentation into separate field"],
            "tool": "neu_esc.data.prepare", "tool_version": PREPROCESSING_VERSION,
            "implementation_sha256": {name: sha256_file(Path(__file__).with_name(name))
                                      for name in ("common.py", "prepare.py", "segmentation.py", "_java_bridge.py")},
            "segmentation": metadata,
        },
        "hf_equivalence_status": config["hf_equivalence_status"],
        "paper_equivalence_status": "unverified_alternative_revision",
        "assumptions": config.get("assumptions", []),
        "source_ledger": source_ledger,
        "full_training_source_notice": "GitHub alternative revision; HF/paper byte/content equivalence has not been established",
    }
    manifest["manifest_digest"] = manifest_digest(manifest)
    # Refuse a different cohort at an existing target, preserving previous usable data.
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_digest(previous) != previous.get("manifest_digest"):
            raise DataError("Existing manifest digest is corrupt; investigate before rerunning")
        if previous["manifest_digest"] != manifest["manifest_digest"]:
            raise DataError("Existing manifest belongs to a different data/config/code cohort; use a new --output and --manifest-path")
    for split, body in payloads.items():
        target = output / f"{split}.jsonl"
        if target.exists() and target.read_bytes() != body:
            raise DataError(f"Existing processed bytes differ at {target}; preserve and investigate")
    statistics, duplicates = summarize(splits)
    statistics["dataset_manifest_digest"] = manifest["manifest_digest"]
    statistics["author_processed_comparison"] = alignment
    # Guard against a source being modified during preparation.
    for split, source in config["sources"].items():
        if sha256_file(root / source["path"]) != raw_hashes[split]:
            raise DataError(f"Source changed during preparation: {split}")
    for split, body in payloads.items():
        target = output / f"{split}.jsonl"
        if not target.exists():
            atomic_bytes(target, body)
    atomic_bytes(audit_directory / "data_statistics.json", json_bytes(statistics))
    atomic_bytes(audit_directory / "duplicate_groups.jsonl", b"".join(canonical_bytes(g) + b"\n" for g in duplicates))
    atomic_bytes(manifest_path.parent / "label_maps.json", json_bytes(LABEL_MAPS))
    # Consumer can only accept data after this checksum-bearing manifest is published.
    if not manifest_path.exists():
        atomic_bytes(manifest_path, json_bytes(manifest))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--manifest-path", help="A new manifest path is required for a changed cohort")
    parser.add_argument("--offline", action="store_true", help="Only reuse existing verified source bytes")
    args = parser.parse_args()
    try:
        result = prepare(args.config, args.output, args.manifest_path, not args.offline)
    except (DataError, OSError) as error:
        parser.exit(2, f"Data gate failed: {error}\n")
    print(json.dumps({"row_counts": result["row_counts"], "manifest_digest": result["manifest_digest"],
                      "hf_equivalence_status": result["hf_equivalence_status"]}, ensure_ascii=True))


if __name__ == "__main__":
    main()
