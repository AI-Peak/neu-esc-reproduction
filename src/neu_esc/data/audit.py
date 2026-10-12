"""Counts and duplicate evidence; never drop/deduplicate/relabel records."""

from __future__ import annotations

import hashlib
import itertools
import unicodedata
from collections import Counter, defaultdict

from .common import LABEL_MAPS


def text_key(text, mode="exact"):
    if mode == "nfc_whitespace":
        text = " ".join(unicodedata.normalize("NFC", text).split())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def duplicate_audit(splits, mode):
    indexes = {}
    within, between, evidence = {}, {}, []
    for split, records in splits.items():
        index = defaultdict(list)
        for row in records:
            index[text_key(row["text"], mode)].append(row)
        indexes[split] = index
        groups = [group for group in index.values() if len(group) > 1]
        within[split] = {
            "duplicate_groups": len(groups), "rows_in_duplicate_groups": sum(len(g) for g in groups),
            "extra_rows": sum(len(g) - 1 for g in groups),
            "conflicting_label_groups": sum(len({(r['sentiment'], r['topic']) for r in g}) > 1 for g in groups),
        }
        for key, group in index.items():
            if len(group) > 1:
                evidence.append({"mode": mode, "kind": "within", "split": split, "text_sha256": key,
                                 "sample_ids": [r["sample_id"] for r in group],
                                 "label_pairs": sorted({(r['sentiment'], r['topic']) for r in group})})
    for a, b in itertools.combinations(splits, 2):
        shared = indexes[a].keys() & indexes[b].keys()
        between[f"{a}__{b}"] = {
            "shared_unique_texts": len(shared),
            "rows_left": sum(len(indexes[a][key]) for key in shared),
            "rows_right": sum(len(indexes[b][key]) for key in shared),
            "cross_split_row_pairs": sum(len(indexes[a][key]) * len(indexes[b][key]) for key in shared),
            "conflicting_label_texts": sum(len({(r['sentiment'], r['topic']) for r in indexes[a][key] + indexes[b][key]}) > 1 for key in shared),
        }
        for key in sorted(shared):
            group = indexes[a][key] + indexes[b][key]
            evidence.append({"mode": mode, "kind": "between", "splits": [a, b], "text_sha256": key,
                             "sample_ids": [r["sample_id"] for r in group],
                             "label_pairs": sorted({(r['sentiment'], r['topic']) for r in group})})
    return {"within": within, "between": between}, evidence


def summarize(splits):
    statistics = {"schema_version": 1, "retention_policy": "keep all source rows, labels and splits", "splits": {}}
    for split, records in splits.items():
        lengths = [len(row["text"].split()) for row in records]
        sentiment = Counter(row["sentiment"] for row in records)
        topic = Counter(row["topic"] for row in records)
        pairs = Counter((row["sentiment"], row["topic"]) for row in records)
        statistics["splits"][split] = {
            "row_count": len(records), "unique_sample_ids": len({r['sample_id'] for r in records}),
            "null_or_empty_text": sum(not r['text'].strip() for r in records),
            "invalid_sentiment": sum(str(r['sentiment']) not in LABEL_MAPS['sentiment'] for r in records),
            "invalid_topic": sum(str(r['topic']) not in LABEL_MAPS['topic'] for r in records),
            "sentiment_counts": {str(k): sentiment[k] for k in range(4)},
            "topic_counts": {str(k): topic[k] for k in range(10)},
            "label_pair_counts": [[pairs[(s, t)] for t in range(10)] for s in range(4)],
            "utf8_replacement_char_rows": sum('\ufffd' in r['text'] for r in records),
            "non_nfc_rows": sum(unicodedata.normalize('NFC', r['text']) != r['text'] for r in records),
            "underscore_rows_raw": sum('_' in r['text'] for r in records),
            "uppercase_rows": sum(any(c.isupper() for c in r['text']) for r in records),
            "whitespace_word_length": {"min": min(lengths, default=0), "max": max(lengths, default=0),
                                       "mean": sum(lengths) / len(lengths) if lengths else None},
            "segmented_rows": sum(isinstance(r['text_segmented'], str) for r in records),
        }
    evidence = []
    statistics["duplicates"] = {}
    for mode in ("exact", "nfc_whitespace"):
        counts, groups = duplicate_audit(splits, mode)
        statistics["duplicates"][mode] = counts
        evidence.extend(groups)
    return statistics, evidence


def compare_author_processed(raw_records, author_records):
    """Row/label alignment plus token preservation checks; no silent reuse."""
    label_mismatches, unsegment_mismatches, segmentation_mismatches = [], [], []
    for raw, author in zip(raw_records, author_records):
        row = raw["source_row"]
        if (raw["sentiment"], raw["topic"]) != (author["sentiment"], author["topic"]):
            label_mismatches.append(row)
        unsegmented = " ".join(author["text"].replace("_", " ").split())
        if " ".join(raw["text"].split()) != unsegmented:
            unsegment_mismatches.append(row)
        if raw["text_segmented"] is not None and raw["text_segmented"] != author["text"]:
            segmentation_mismatches.append(row)
    return {
        "raw_rows": len(raw_records), "author_processed_rows": len(author_records),
        "label_mismatch_rows": label_mismatches, "unsegment_normalized_text_mismatch_rows": unsegment_mismatches,
        "our_segmented_vs_author_mismatch_rows": segmentation_mismatches,
        "segmentation_compared": any(r["text_segmented"] is not None for r in raw_records),
        "author_segmentation_tool_and_dictionary": "not published in pinned repository",
    }
