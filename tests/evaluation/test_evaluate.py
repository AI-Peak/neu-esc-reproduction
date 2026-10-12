import copy
import csv
import hashlib
import io
import json
import math
import random
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from neu_esc.evaluation.evaluate import (
    EvaluationError, LABELS, _manifest_digest, evaluate_records, main,
    metrics_from_counts, write_evaluation,
)


def expected_fixture():
    return [dict(sample_id=f"test:{i:06d}:fixture", split="test", sentiment=y, topic=y)
            for i, y in enumerate([0, 0, 1, 2])]


def prediction_fixture(expected=None, tasks=("sentiment",)):
    expected = expected_fixture() if expected is None else expected
    return [dict(sample_id=r["sample_id"], split=r["split"], task=t,
                 y_true=r[t], y_pred=p, status="ok", error=None)
            for t in tasks for r,p in zip(expected, [0, 1, 1, 1])]


class EvaluationTests(unittest.TestCase):
    def test_hand_calculated_scores_and_orientation(self):
        result = evaluate_records(prediction_fixture(), expected_fixture(), ["sentiment"])
        v = result["tasks"]["sentiment"]
        self.assertEqual(v["confusion_matrix"], [[1,1,0,0], [0,1,0,0], [0,1,0,0], [0,0,0,0]])
        self.assertEqual(v["n_samples"], 4)
        self.assertEqual(v["accuracy"], 0.5)
        self.assertAlmostEqual(v["macro_f1"], 7/24)
        self.assertAlmostEqual(v["weighted_f1"], 11/24)
        self.assertEqual(v["per_class"][0]["precision"], 1)
        self.assertEqual(v["per_class"][0]["recall"], 0.5)
        self.assertEqual(v["per_class"][2]["false_negative"], 1)
        self.assertEqual(v["confusion_matrix_normalized"][0], [0.5,0.5,0,0])
        self.assertEqual(v["confusion_matrix_normalized"][3], [0,0,0,0])

    def test_shuffle_ids_and_authoritative_order(self):
        expected = expected_fixture()
        records = prediction_fixture(tasks=("sentiment", "topic"))
        original = evaluate_records(records, expected, ["sentiment", "topic"])
        random.Random(42).shuffle(records)
        expected.reverse()
        shuffled = evaluate_records(records, iter(expected), ["sentiment", "topic"])
        self.assertEqual(original, shuffled)
        for task, k in (("sentiment",4),("topic",10)):
            self.assertEqual(len(shuffled["tasks"][task]["confusion_matrix"]), k)
            self.assertEqual(sum(map(sum, shuffled["tasks"][task]["confusion_matrix"])), 4)

    def test_all_one_class_keeps_absent_classes(self):
        expected = [dict(sample_id=str(i), split="test", sentiment=0) for i in range(3)]
        records = [dict(sample_id=r["sample_id"], split="test", task="sentiment",
                        y_true=0, y_pred=0, status="ok", error=None) for r in expected]
        v = evaluate_records(records, expected, ["sentiment"])["tasks"]["sentiment"]
        self.assertEqual(v["accuracy"], 1)
        self.assertEqual(v["macro_f1"], 0.25)
        self.assertEqual(v["weighted_f1"], 1)
        self.assertEqual(v["per_class"][3]["support"], 0)
        self.assertEqual(v["per_class"][3]["f1"], 0)

    def test_invalid_abstention_fixed_denominator(self):
        records = prediction_fixture()
        records[1].update(y_pred=None, status="invalid", error="timeout after 3 attempts")
        v = evaluate_records(records, expected_fixture(), ["sentiment"], True)["tasks"]["sentiment"]
        self.assertEqual(v["n_samples"], 4)
        self.assertEqual(v["accuracy"], 0.5)
        self.assertEqual(v["invalid_count"], 1)
        self.assertEqual(v["record_coverage"], 1)
        self.assertEqual(v["valid_prediction_coverage"], 0.75)
        self.assertEqual(v["confusion_matrix"][0][-1], 1)
        self.assertEqual(v["per_class"][0]["false_negative"], 1)
        self.assertAlmostEqual(v["macro_f1"], 1/3)
        self.assertEqual(v["weighted_f1"], 0.5)
        self.assertEqual(v["confusion_columns"], [0,1,2,3,"INVALID"])

    def test_all_invalid_is_zero_with_full_n(self):
        records = prediction_fixture(tasks=("sentiment", "topic"))
        for row in records:
            row.update(y_pred=None, status="invalid", error="parse failure")
        result = evaluate_records(records, expected_fixture(), ["sentiment", "topic"], True)
        for task, k in (("sentiment",4),("topic",10)):
            v = result["tasks"][task]
            self.assertEqual(v["invalid_count"], 4)
            self.assertEqual(v["accuracy"], 0)
            self.assertEqual(v["macro_f1"], 0)
            self.assertEqual(v["weighted_f1"], 0)
            self.assertEqual(len(v["confusion_matrix"][0]), k+1)

    def test_negative_integrity_cases(self):
        base = prediction_fixture()
        cases = []
        for field, value, pattern in [
            ("sample_id","extra", "extra prediction"),
            ("split","validation", "wrong split"),
            ("task","topic", "extra/wrong task"),
            ("y_true",1,"wrong labels"),
            ("y_pred",4,"invalid class"),
            ("y_pred",-1,"invalid class"),
            ("y_pred",0.0,"invalid class"),
            ("y_pred",True,"invalid class"),
            ("y_pred",None,"invalid class"),
            ("status","timeout","unknown status"),
            ("error","unexpected","error=null"),
        ]:
            rows = copy.deepcopy(base)
            rows[0][field] = value
            cases.append((field+repr(value), rows, False, pattern))
        cases.extend([
            ("duplicate", base+[base[0]], False, "duplicate prediction"),
            ("missing", base[:-1], False, "missing prediction"),
            ("missing with allow_invalid", base[:-1], True, "missing prediction"),
            ("empty predictions", [], False, "missing prediction"),
        ])
        for name, rows, invalid, pattern in cases:
            with self.subTest(name=name), self.assertRaisesRegex(EvaluationError, pattern):
                evaluate_records(rows, expected_fixture(), ["sentiment"], invalid)

    def test_wrong_truth_cannot_be_hidden_by_shuffle(self):
        records = prediction_fixture()
        records[0]["y_true"], records[2]["y_true"] = records[2]["y_true"], records[0]["y_true"]
        records.reverse()
        with self.assertRaisesRegex(EvaluationError, "wrong labels"):
            evaluate_records(records, expected_fixture(), ["sentiment"])

    def test_invalid_contract_failures(self):
        records = prediction_fixture()
        records[0].update(status="invalid", y_pred=None, error="timeout")
        with self.assertRaisesRegex(EvaluationError, "not allowed"):
            evaluate_records(records, expected_fixture(), ["sentiment"])
        for update, pattern in [({"y_pred":0},"y_pred=null"), ({"error":None},"nonempty error"),
                                ({"error":" "},"nonempty error")]:
            rows = copy.deepcopy(records)
            rows[0].update(update)
            with self.subTest(update=update), self.assertRaisesRegex(EvaluationError,pattern):
                evaluate_records(rows, expected_fixture(), ["sentiment"], True)

    def test_expected_input_integrity(self):
        expected = expected_fixture()
        cases = [([], "nonempty"), (expected+[expected[0]], "duplicate expected")]
        for field,value,pattern in [("split","validation","wrong split"),
                                    ("sentiment",99,"invalid class"),
                                    ("sample_id","", "sample_id")]:
            rows = copy.deepcopy(expected)
            rows[1][field] = value
            cases.append((rows, pattern))
        for rows, pattern in cases:
            with self.subTest(pattern=pattern), self.assertRaisesRegex(EvaluationError, pattern):
                evaluate_records(prediction_fixture(), rows, ["sentiment"])

    def test_task_list_integrity(self):
        for tasks in ([], ["sentiment","sentiment"], ["other"], "sentiment"):
            with self.subTest(tasks=tasks), self.assertRaises(EvaluationError):
                evaluate_records(prediction_fixture(), expected_fixture(), tasks)

    def test_missing_whole_mtl_task(self):
        with self.assertRaisesRegex(EvaluationError, "missing prediction"):
            evaluate_records(prediction_fixture(), expected_fixture(), ["sentiment", "topic"], True)

    def test_missing_required_fields(self):
        for key in ("error", "y_pred", "y_true", "status", "sample_id", "split", "task"):
            rows = prediction_fixture()
            del rows[0][key]
            with self.subTest(key=key), self.assertRaises(EvaluationError):
                evaluate_records(rows, expected_fixture(), ["sentiment"])

    def test_count_validation(self):
        for counts in ([[0]*4]*4, [[1]*3]*4, [[-1]*4]*4, [[True]*4]*4,
                       [[1.0]*4]*4, [[1]*4]*3, [[1]*4,[1]*4,[1]*3,[1]*4]):
            with self.subTest(counts=counts), self.assertRaises(EvaluationError):
                metrics_from_counts(counts, "sentiment")

    def test_writer_csv_json_consistent_and_refuses_overwrite(self):
        result = evaluate_records(prediction_fixture(), expected_fixture(), ["sentiment"])
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/"output"
            write_evaluation(result,out,False)
            loaded = json.loads((out/"metrics.json").read_text())
            self.assertEqual(loaded, result)
            with (out/"confusion_sentiment.csv").open(newline="") as stream:
                rows = list(csv.reader(stream))
            self.assertEqual([[int(v) for v in row[1:]] for row in rows[1:]],
                             result["tasks"]["sentiment"]["confusion_matrix"])
            with self.assertRaisesRegex(EvaluationError,"overwrite"):
                write_evaluation(result,out,False)

    def test_png_output(self):
        from PIL import Image
        records = prediction_fixture(tasks=("sentiment", "topic"))
        records[0].update(status="invalid",y_pred=None,error="fixture timeout")
        result = evaluate_records(records,expected_fixture(),["sentiment","topic"],True)
        with tempfile.TemporaryDirectory() as directory:
            write_evaluation(result,Path(directory)/"output")
            for path in (Path(directory)/"output").glob("*.png"):
                with Image.open(path) as image:
                    image.verify()
            self.assertEqual(len(list((Path(directory)/"output").glob("*.png"))),4)


def cli_fixture(directory):
    directory = Path(directory)
    run = directory/"fixture_run"
    (run/"predictions").mkdir(parents=True)
    prediction_path = run/"predictions"/"test.jsonl"
    prediction_path.write_text("".join(json.dumps(r)+"\n" for r in prediction_fixture(tasks=("sentiment","topic"))),encoding="utf-8")
    manifest = dict(schema_version="1.0", row_counts={"test":4}, note="fixture only")
    manifest["manifest_digest"] = hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
    manifest_path = directory/"dataset.fixture.json"
    manifest_path.write_text(json.dumps(manifest),encoding="utf-8")
    provenance = dict(run_id="fixture-run",cohort_id="fixture-cohort",seed=42,method_mode="paper_method",
                      tasks=["sentiment","topic"],dataset_manifest_digest=manifest["manifest_digest"])
    (run/"provenance.json").write_text(json.dumps(provenance),encoding="utf-8")
    return prediction_path,manifest_path,run/"evaluation"/"test"


class CliTests(unittest.TestCase):
    def call_cli(self, predictions, manifest, output, **kwargs):
        # Deliberate fixture loader at the data API boundary, never a benchmark.
        fake = types.ModuleType("neu_esc.data.dataset")
        fake.load_split = kwargs.get("loader", lambda manifest,split,text:expected_fixture())
        with patch.dict(sys.modules,{"neu_esc.data.dataset":fake}), redirect_stdout(io.StringIO()):
            return main(["--predictions",str(predictions),"--dataset-manifest",str(manifest),
                         "--output",str(output),"--no-png",*kwargs.get("extra",[])])

    def test_cli_api_boundary_metadata_hash_and_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            predictions,manifest,output = cli_fixture(directory)
            seen=[]
            def loader(path,split,text):
                seen.append((path,split,text))
                return iter(expected_fixture())
            self.assertEqual(self.call_cli(predictions,manifest,output,loader=loader),0)
            self.assertEqual(seen,[(manifest,"test","raw")])
            result=json.loads((output/"metrics.json").read_text())
            self.assertEqual(result["seed"],42)
            self.assertEqual(result["run_id"],"fixture-run")
            self.assertEqual(result["task_source"],"declared")
            self.assertEqual(result["predictions_sha256"],hashlib.sha256(predictions.read_bytes()).hexdigest())
            self.assertEqual(result["dataset_manifest_digest"],_manifest_digest(json.loads(manifest.read_text())))
            self.assertEqual(len(result["tasks"]),2)

    def test_cli_integrity_failures_emit_no_metrics(self):
        for case in ("missing-task","wrong-manifest","wrong-row-count","wrong-seed","blank-line","undeclared-tasks","wrong-task-override"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                predictions,manifest,output=cli_fixture(directory)
                extra=[]
                if case=="missing-task":
                    predictions.write_text("".join(json.dumps(r)+"\n" for r in prediction_fixture()),encoding="utf-8")
                elif case=="wrong-manifest":
                    value=json.loads(manifest.read_text()); value["note"]="changed"
                    manifest.write_text(json.dumps(value),encoding="utf-8")
                elif case=="wrong-row-count":
                    extra=[]
                elif case=="wrong-seed":
                    extra=["--seed","43"]
                elif case=="blank-line":
                    predictions.write_text(predictions.read_text()+"\n",encoding="utf-8")
                elif case=="undeclared-tasks":
                    (predictions.parent.parent/"provenance.json").unlink()
                elif case=="wrong-task-override":
                    extra=["--tasks","sentiment"]
                loader=(lambda *args:expected_fixture()[:-1]) if case=="wrong-row-count" else (lambda *args:expected_fixture())
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    self.call_cli(predictions,manifest,output,extra=extra,loader=loader)
                self.assertEqual(error.exception.code,2)
                self.assertFalse(output.exists())

    def test_cli_rejects_mutated_predictions_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            predictions,manifest,output=cli_fixture(directory)
            def loader(*args):
                predictions.write_text("changed during evaluation",encoding="utf-8")
                return expected_fixture()
            stderr=io.StringIO()
            with redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
                self.call_cli(predictions,manifest,output,loader=loader)
            self.assertEqual(error.exception.code,2)
            self.assertIn("predictions changed",stderr.getvalue())
            self.assertFalse(output.exists())


if __name__=="__main__":
    unittest.main()
