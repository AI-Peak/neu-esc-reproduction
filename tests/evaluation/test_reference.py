import csv
import hashlib
import json
import unittest
from pathlib import Path

from neu_esc.evaluation.evaluate import metrics_from_counts


REFERENCE=Path(__file__).resolve().parents[2]/"reports"/"reference"


class ReferenceTests(unittest.TestCase):
    def test_table_shape_numeric_only_and_source_pages(self):
        with (REFERENCE/"table6.csv").open(encoding="utf-8",newline="") as stream:
            rows=list(csv.DictReader(stream))
        self.assertEqual(len(rows),21)
        self.assertEqual(sum(r["variant"]=="single_task_pair" for r in rows),5)
        self.assertEqual(sum(r["variant"].startswith("mtl") for r in rows),12)
        self.assertEqual(sum(r["variant"] in ("zero_shot","few_shot") for r in rows),4)
        for row in rows:
            self.assertEqual(row["source_page"],"18")
            for key,value in row.items():
                if key.endswith("_pct"):
                    self.assertTrue(0<=float(value)<=100)
                    self.assertNotIn("*",value)

    def test_figure_totals_traces_and_audit_hashes(self):
        audit=json.loads((REFERENCE/"figure3_audit.json").read_text(encoding="utf-8"))
        for task,trace,accuracy in [("sentiment",3619,3619/6613),("topic",2074,2074/6613)]:
            path=REFERENCE/f"figure3_{task}_counts.csv"
            with path.open(encoding="utf-8",newline="") as stream:
                rows=list(csv.reader(stream))
            matrix=[[int(v) for v in r[1:]] for r in rows[1:]]
            value=metrics_from_counts(matrix,task)
            self.assertEqual(value["n_samples"],6613)
            self.assertEqual(sum(matrix[i][i] for i in range(len(matrix))),trace)
            self.assertEqual(value["accuracy"],accuracy)
            self.assertEqual(audit["tasks"][task]["accuracy"],accuracy)
            self.assertEqual(audit["tasks"][task]["csv_sha256"],hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertTrue(audit["tasks"][task]["inconsistent_with_table6"])
        self.assertFalse(audit["reproduction_results"])
        for name,digest in audit["artifacts_sha256"].items():
            self.assertEqual(hashlib.sha256((REFERENCE/name).read_bytes()).hexdigest(),digest)


if __name__=="__main__":
    unittest.main()
