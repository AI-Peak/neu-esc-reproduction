"""CPU-only synthetic gate tests: no encoder/tokenizer download or benchmark claims."""

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from neu_esc.data.audit import summarize
from neu_esc.data.common import (
    DataError, LABEL_MAPS, PREPROCESSING_VERSION, canonical_bytes, digest,
    fetch_locked, label_id, manifest_digest, read_source_csv, sha256_file,
)
from neu_esc.data.dataset import load_split
from neu_esc.data.prepare import prepare


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="neu_esc_data_test_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "configs/data/dataset.yaml"
        self.path.parent.mkdir(parents=True)
        self.config = {
            "schema_version": 1, "project_root": "../..",
            "manifest_path": "data/manifests/dataset.json", "audit_directory": "reports/source",
            "source_revision": "synthetic-test-only", "source_provider": "synthetic-fixture",
            "hf_equivalence_status": "not_applicable_synthetic", "label_maps": LABEL_MAPS,
            "preprocessing": {"version": PREPROCESSING_VERSION, "cleaning": "identity", "segmentation": {"backend": "none"}},
            "sources": {},
        }
        for split in ("train", "validation", "test"):
            self.source(split, [["sinh viên học Python", "0.0", "2.0"], ["sinh viên học Python", "3", "9"]])

    def source(self, split, rows):
        path = self.root / f"data/raw/{split}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["text", "sentiment", "classification"])
            writer.writerows(rows)
        self.config["sources"][split] = {
            "path": f"data/raw/{split}.csv", "url": "https://invalid.example/never-contacted",
            "sha256": sha256_file(path), "rows": len(rows),
        }
        self.save_config()

    def save_config(self):
        self.path.write_text(yaml.safe_dump(self.config), encoding="utf-8")

    @property
    def manifest(self):
        return self.root / "data/manifests/dataset.json"

    def test_restart_deterministic_ids_hashes_and_preserves_raw(self):
        before = {s: sha256_file(self.root / v["path"]) for s, v in self.config["sources"].items()}
        first = prepare(self.path, download=False)
        raw_records = load_split(self.manifest, "train", "raw")
        second = prepare(self.path, download=False)
        self.assertEqual(first, second)
        self.assertEqual(first["manifest_digest"], manifest_digest(first))
        self.assertEqual(first["raw_sha256"], before)
        self.assertEqual(first["row_counts"], {"train": 2, "validation": 2, "test": 2})
        self.assertEqual(first["label_maps"], LABEL_MAPS)
        self.assertNotEqual(raw_records[0]["sample_id"], raw_records[1]["sample_id"])
        self.assertEqual(raw_records, load_split(self.manifest, "train", "raw"))
        self.assertEqual(raw_records[0]["text"], "sinh viên học Python")
        for split, source in self.config["sources"].items():
            self.assertEqual(before[split], sha256_file(self.root / source["path"]))

    def test_exact_labels_and_invalid_failures(self):
        for task, k in (("sentiment", 4), ("topic", 10)):
            for label in range(k):
                self.assertEqual(label_id(str(label) + ".0", task, "fixture"), label)
            for invalid in (None, "", "NaN", "inf", "1.5", "-1", str(k), True, "positive"):
                with self.subTest(task=task, invalid=invalid), self.assertRaises(DataError):
                    label_id(invalid, task, "fixture")

    def test_null_empty_text_and_labels_fail_before_publication(self):
        for row in (["", "0", "0"], ["  ", "0", "0"], ["NaN", "0", "0"], ["x", "", "0"], ["x", "0", ""]):
            with self.subTest(row=row):
                self.source("train", [row])
                with self.assertRaises(DataError):
                    prepare(self.path, download=False)
                self.assertFalse(self.manifest.exists())

    def test_missing_segmented_variant_fails(self):
        prepare(self.path, download=False)
        with self.assertRaisesRegex(DataError, "Missing segmented"):
            load_split(self.manifest, "test", "segmented")

    def test_explicit_selected_text_and_ids_unchanged_when_segmented(self):
        class FixtureSegmenter:
            metadata = {"enabled": True, "backend": "synthetic_test_only"}
            def __init__(self, *_):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *_):
                pass
            def segment(self, text):
                return text.replace("sinh viên", "sinh_viên")
        self.config["preprocessing"]["segmentation"] = {"backend": "vncorenlp", "resources_lock": "fixture"}
        self.save_config()
        with patch("neu_esc.data.prepare.VnCoreSegmenter", FixtureSegmenter):
            result = prepare(self.path, download=False)
        for split in result["row_counts"]:
            raw = load_split(self.manifest, split, "raw")
            seg = load_split(self.manifest, split, "segmented")
            self.assertEqual(len(raw), len(seg))
            for a, b in zip(raw, seg):
                self.assertEqual((a["sample_id"], a["split"], a["source_row"], a["sentiment"], a["topic"]),
                                 (b["sample_id"], b["split"], b["source_row"], b["sentiment"], b["topic"]))
                self.assertEqual(b["text_raw"], a["text"])
                self.assertEqual(b["text"], "sinh_viên học Python")

    def test_segmentation_failure_does_not_publish(self):
        self.config["preprocessing"]["segmentation"] = {"backend": "vncorenlp", "resources_lock": "missing-lock.json"}
        self.save_config()
        with self.assertRaises((DataError, FileNotFoundError)):
            prepare(self.path, download=False)
        self.assertFalse(self.manifest.exists())

    def test_manifest_tampering_fails(self):
        result = prepare(self.path, download=False)
        result["row_counts"]["test"] = 123
        self.manifest.write_text(json.dumps(result), encoding="utf-8")
        with self.assertRaisesRegex(DataError, "Manifest digest"):
            load_split(self.manifest, "test", "raw")

    def test_processed_tampering_fails(self):
        prepare(self.path, download=False)
        path = self.root / "data/processed/test.jsonl"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(DataError, "Processed bytes hash"):
            load_split(self.manifest, "test", "raw")

    def test_bad_source_hash_preserves_existing_bytes(self):
        source = self.config["sources"]["train"]
        path = self.root / source["path"]
        original = path.read_bytes()
        with self.assertRaisesRegex(DataError, "Source hash mismatch"):
            fetch_locked(source["url"], path, "0" * 64)
        self.assertEqual(path.read_bytes(), original)

    def test_changed_cohort_refused_without_overwrite(self):
        prepare(self.path, download=False)
        original = self.manifest.read_bytes()
        self.config["assumptions"] = ["changed config"]
        self.save_config()
        with self.assertRaisesRegex(DataError, "different data/config/code cohort"):
            prepare(self.path, download=False)
        self.assertEqual(self.manifest.read_bytes(), original)

    def test_mapping_conflict_and_row_count_gate(self):
        self.config["label_maps"] = json.loads(json.dumps(LABEL_MAPS))
        self.config["label_maps"]["topic"]["3"] = "Neutral"
        self.save_config()
        with self.assertRaisesRegex(DataError, "mapping conflicts"):
            prepare(self.path, download=False)
        self.config["label_maps"] = LABEL_MAPS
        self.config["sources"]["train"]["rows"] += 1
        self.save_config()
        with self.assertRaisesRegex(DataError, "expected"):
            prepare(self.path, download=False)

    def test_invalid_encoding_and_headers_fail(self):
        path = self.root / "bad.csv"
        path.write_bytes(b"text,sentiment,classification\n\xff,0,0\n")
        with self.assertRaisesRegex(DataError, "invalid UTF-8"):
            read_source_csv(path, "test", "a" * 64)
        path.write_text("text,sentiment,topic\nx,0,0\n", encoding="utf-8")
        with self.assertRaisesRegex(DataError, "expected columns"):
            read_source_csv(path, "test", "a" * 64)

    def test_duplicates_audited_and_kept(self):
        prepare(self.path, download=False)
        splits = {s: load_split(self.manifest, s, "raw") for s in self.config["sources"]}
        report, evidence = summarize(splits)
        self.assertEqual(report["duplicates"]["exact"]["within"]["train"]["extra_rows"], 1)
        self.assertEqual(report["duplicates"]["exact"]["within"]["train"]["conflicting_label_groups"], 1)
        self.assertEqual(report["duplicates"]["exact"]["between"]["train__test"]["cross_split_row_pairs"], 4)
        self.assertTrue(evidence)
        self.assertTrue(all(len(rows) == 2 for rows in splits.values()))

    def test_reordered_rows_fail_even_with_recomputed_hash(self):
        result = prepare(self.path, download=False)
        path = self.root / "data/processed/test.jsonl"
        path.write_bytes(b"".join(reversed(path.read_bytes().splitlines(keepends=True))))
        result["processed_sha256"]["test"] = sha256_file(path)
        result["manifest_digest"] = manifest_digest(result)
        self.manifest.write_text(json.dumps(result), encoding="utf-8")
        with self.assertRaisesRegex(DataError, "reordered"):
            load_split(self.manifest, "test", "raw")

    def test_relative_paths_portable_and_no_machine_paths_in_digest(self):
        result = prepare(self.path, download=False)
        self.assertEqual(result["processed_files"]["test"], "../processed/test.jsonl")
        self.assertNotIn(str(self.root), canonical_bytes(result).decode("utf-8"))
        self.assertEqual(digest({"b": 2, "a": "Việt"}), digest({"a": "Việt", "b": 2}))


if __name__ == "__main__":
    unittest.main()
