import copy
import math
import csv
import tempfile
from pathlib import Path
import unittest

from neu_esc.evaluation.aggregate import aggregate_cohort, combine_st_runs, paper_metrics_from_row, write_summary_csv
from neu_esc.evaluation.evaluate import EvaluationError, evaluate_records


def seed_result(seed,correct,task="sentiment",n=4):
    expected=[dict(sample_id=f"test:{i}",split="test",sentiment=0,topic=0) for i in range(n)]
    records=[dict(sample_id=r["sample_id"],split="test",task=task,y_true=0,
                  y_pred=0 if i<correct else 1,status="ok",error=None) for i,r in enumerate(expected)]
    result=evaluate_records(records,expected,[task])
    result.update(run_id=f"fixture-{task}-{seed}",cohort_id=f"fixture-cohort-{task}",seed=seed,
                  method_mode="paper_method",dataset_manifest_digest="fixture-manifest",
                  backbone="visobert",variant=f"st_{task}")
    return result


class AggregateTests(unittest.TestCase):
    def test_five_seeds_hand_calculated_mean_sample_std_delta(self):
        values=[seed_result(seed,i) for i,seed in enumerate(range(42,47))]
        paper={"sentiment":dict(accuracy=50,macro_f1=20,weighted_f1=60)}
        result=aggregate_cohort(values,paper_metrics=paper,expected_seeds=[42,43,44,45,46])
        summary=result["tasks"]["sentiment"]
        accuracy=summary["metrics"]["accuracy"]
        self.assertEqual(accuracy["mean"],0.5)
        self.assertAlmostEqual(accuracy["std"],math.sqrt(0.15625))
        self.assertEqual(accuracy["delta_pp"],0)
        self.assertEqual(accuracy["n"],5)
        self.assertEqual(accuracy["ddof"],1)
        self.assertEqual(summary["aggregate_confusion"]["n_samples"],20)
        self.assertEqual(summary["aggregate_confusion"]["counts"][0],[10,10,0,0])
        self.assertEqual(summary["figure3_comparison"]["seed"],42)
        self.assertEqual(summary["figure3_comparison"]["counts"][0],[0,4,0,0])
        self.assertEqual(len(summary["per_seed_confusion_matrices"]),5)
        # Mean per-seed F1 differs from F1 of a sum CM; never replace it.
        self.assertAlmostEqual(summary["metrics"]["weighted_f1"]["mean"],(0+0.4+2/3+6/7+1)/5)

    def test_one_seed_std_null_and_delta_pp(self):
        result=aggregate_cohort([seed_result(42,3)],paper_metrics={"sentiment":dict(accuracy=50,macro_f1=0,weighted_f1=0)})
        v=result["tasks"]["sentiment"]["metrics"]["accuracy"]
        self.assertIsNone(v["std"])
        self.assertIsNone(v["std_pct"])
        self.assertEqual(v["delta_pp"],25)
        self.assertEqual(result["status"],"observed_seeds_only")

    def test_full_test_sum_n_33065(self):
        values=[seed_result(seed,6000,n=6613) for seed in range(42,47)]
        result=aggregate_cohort(values,expected_seeds=range(42,47))
        summary=result["tasks"]["sentiment"]
        self.assertEqual(summary["aggregate_confusion"]["n_samples"],33065)
        self.assertEqual(summary["figure3_comparison"]["n_samples"],6613)

    def test_unavailable_42_is_not_replaced_with_best_seed(self):
        result=aggregate_cohort([seed_result(43,4)])
        value=result["tasks"]["sentiment"]["figure3_comparison"]
        self.assertEqual(value["status"],"not_available")
        self.assertIsNone(value["counts"])

    def test_same_cohort_gate_and_duplicates(self):
        for field,value,pattern in [("cohort_id","other","cohort_id"),
                                    ("method_mode","reference_code","method_mode"),
                                    ("dataset_manifest_digest","other","dataset_manifest_digest"),
                                    ("split","validation","split"),
                                    ("seed",42,"duplicate seed"),
                                    ("run_id","fixture-sentiment-42","duplicate run_id")]:
            other=seed_result(43,2); other[field]=value
            with self.subTest(field=field), self.assertRaisesRegex(EvaluationError,pattern):
                aggregate_cohort([seed_result(42,1),other])

    def test_aggregate_rejects_tampered_metrics_or_sample_counts(self):
        bad=seed_result(43,2); bad["tasks"]["sentiment"]["accuracy"]=0.9
        with self.assertRaisesRegex(EvaluationError,"metric/CM mismatch"):
            aggregate_cohort([seed_result(42,1),bad])
        with self.assertRaisesRegex(EvaluationError,"sample counts"):
            aggregate_cohort([seed_result(42,1),seed_result(43,2,n=5)])

    def test_expected_seed_coverage_fail(self):
        for declared in ([42,43],[43],[42,42]):
            with self.subTest(declared=declared), self.assertRaises(EvaluationError):
                aggregate_cohort([seed_result(42,1)],expected_seeds=declared)

    def test_st_pair_by_seed_from_two_separate_runs(self):
        sentiments=[seed_result(seed, i) for i,seed in enumerate(range(42,47))]
        topics=[seed_result(seed,4-i,"topic") for i,seed in enumerate(range(42,47))]
        topics.reverse()
        result=combine_st_runs(sentiments,topics,backbone="visobert",pairing_id="predeclared-fixture",expected_seeds=range(42,47))
        self.assertEqual(result["n_seeds"],5)
        self.assertEqual(result["variant"],"single_task_pair")
        self.assertEqual(result["cohort_ids"],dict(sentiment="fixture-cohort-sentiment",topic="fixture-cohort-topic"))
        self.assertEqual(result["tasks"]["sentiment"]["per_seed"][0]["accuracy"],0)
        self.assertEqual(result["tasks"]["topic"]["per_seed"][0]["accuracy"],1)

    def test_st_pair_mismatch_failures(self):
        for field,value in [("backbone","phobert"),("variant","mtl"),
                            ("seed",43),("method_mode","reference_code"),
                            ("dataset_manifest_digest","other")]:
            topic=seed_result(42,2,"topic"); topic[field]=value
            with self.subTest(field=field), self.assertRaises(EvaluationError):
                combine_st_runs([seed_result(42,1)],[topic],backbone="visobert",pairing_id="predeclared")
        with self.assertRaises(EvaluationError):
            combine_st_runs([seed_result(42,1)],[seed_result(42,2,"topic")],backbone="visobert",pairing_id="")

    def test_report_csv_percent_units_and_task_cohorts(self):
        reference={f"{task}_{metric}_pct":"50.00" for task in ("sentiment","topic")
                   for metric in ("accuracy","macro_f1","weighted_f1")}
        paper=paper_metrics_from_row(reference)
        summary=combine_st_runs([seed_result(42,3)],[seed_result(42,1,"topic")],
                                backbone="visobert",pairing_id="predeclared",paper_metrics=paper,
                                expected_seeds=[42])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"report.csv"
            write_summary_csv(summary,path,row_id="04",backbone="visobert",variant="single_task_pair")
            with path.open(newline="") as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),6)
            sentiment=next(r for r in rows if r["task"]=="sentiment" and r["metric"]=="accuracy")
            topic=next(r for r in rows if r["task"]=="topic" and r["metric"]=="accuracy")
            self.assertEqual(float(sentiment["mean_pct"]),75)
            self.assertEqual(float(sentiment["delta_pp"]),25)
            self.assertEqual(float(topic["mean_pct"]),25)
            self.assertEqual(float(topic["delta_pp"]),-25)
            self.assertEqual(sentiment["std_pct"],"")
            self.assertNotEqual(sentiment["cohort_id"],topic["cohort_id"])
            with self.assertRaises(FileExistsError):
                write_summary_csv(summary,path,row_id="04",backbone="visobert",variant="single_task_pair")


if __name__=="__main__":
    unittest.main()
