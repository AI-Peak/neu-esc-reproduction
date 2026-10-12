"""Create honest offline fixture outputs in this module's smoke namespace.

Uses the real data API and evaluator CLI; does not load benchmark data, train a
model, or invoke an LLM. Existing output is preserved rather than overwritten.
"""

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from neu_esc.data.common import ID_ALGORITHM, LABEL_MAPS, manifest_digest
from neu_esc.evaluation.aggregate import combine_st_runs, write_summary_csv
from neu_esc.evaluation.evaluate import evaluate_records


def _json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n",encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/smoke/evaluation/offline-v1", type=Path)
    args=parser.parse_args()
    root=args.output.resolve()
    if root.exists():
        parser.error(f"refusing to overwrite fixture output: {root}")
    root.mkdir(parents=True)
    raw=b"text,sentiment,classification\nfixture a,0,0\nfixture b,0,0\nfixture c,1,1\nfixture d,2,2\n"
    raw_hash=hashlib.sha256(raw).hexdigest()
    inputs=root/"inputs"
    inputs.mkdir()
    (inputs/"test.fixture.csv").write_bytes(raw)
    expected=[dict(sample_id=f"test:{i:06d}:{raw_hash[:12]}",split="test",source_row=i,
                   text=f"Fixture sample {i}; not NEU-ESC benchmark data.",text_segmented=None,
                   sentiment=y,topic=y) for i,y in enumerate([0,0,1,2])]
    processed=inputs/"test.fixture.jsonl"
    processed.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in expected),encoding="utf-8")
    manifest=dict(schema_version=1, source_urls=["fixture://evaluation/hand-calculated"],
                  source_revision="fixture-only-v1", raw_sha256={"test":raw_hash},
                  processed_sha256={"test":hashlib.sha256(processed.read_bytes()).hexdigest()},
                  processed_files={"test":processed.name}, row_counts={"test":4},
                  id_algorithm=ID_ALGORITHM, label_maps=LABEL_MAPS,
                  preprocessing={"steps":["fixture generation only"]},
                  hf_equivalence_status="not_applicable_fixture", assumptions=["not benchmark data"])
    manifest["manifest_digest"]=manifest_digest(manifest)
    manifest_path=inputs/"dataset.fixture.json"
    _json(manifest_path,manifest)
    base=[dict(sample_id=r["sample_id"],split="test",task=t,y_true=r[t],y_pred=p,status="ok",error=None)
          for t in ("sentiment","topic") for r,p in zip(expected,[0,1,1,1])]
    checks=[]
    for name,invalid in (("encoder",False),("llm_invalid",True)):
        run=root/name
        rows=copy.deepcopy(base)
        if invalid:
            for task in ("sentiment","topic"):
                next(r for r in rows if r["task"]==task and r["sample_id"]==expected[1]["sample_id"]).update(
                    y_pred=None,status="invalid",error="fixture timeout after bounded retries")
        rows.reverse()  # real API/CLI must still join by ID, never by order
        predictions=run/"predictions"/"test.jsonl"
        predictions.parent.mkdir(parents=True)
        predictions.write_text("".join(json.dumps(r)+"\n" for r in rows),encoding="utf-8")
        _json(run/"provenance.json",dict(run_id=f"smoke-evaluation-{name}",cohort_id=f"fixture-{name}",
              seed=42,method_mode="paper_method",backbone="fixture",variant="mtl",
              tasks=["sentiment","topic"],dataset_manifest_digest=manifest["manifest_digest"],
              run_status="FIXTURE_ONLY_NO_MODEL"))
        command=[sys.executable,"-m","neu_esc.evaluation.evaluate","--predictions",str(predictions),
                 "--dataset-manifest",str(manifest_path),"--output",str(run/"evaluation"/"test")]
        if invalid:
            command.append("--allow-invalid")
        result=subprocess.run(command,capture_output=True,text=True)
        checks.append(dict(name=name,command=command,exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr))
        if result.returncode:
            _json(root/"checks.json",dict(fixture_only=True,checks=checks))
            raise RuntimeError(result.stderr)
    seed_runs={"sentiment":[],"topic":[]}
    for task in seed_runs:
        for i,seed in enumerate(range(42,47)):
            rows=[dict(sample_id=r["sample_id"],split="test",task=task,y_true=r[task],
                       y_pred=r[task] if j<i else (r[task]+1)%len(LABEL_MAPS[task]),status="ok",error=None)
                  for j,r in enumerate(expected)]
            result=evaluate_records(rows,expected,[task])
            result.update(run_id=f"fixture-st-{task}-{seed}",cohort_id=f"fixture-st-{task}",
                          seed=seed,method_mode="paper_method",dataset_manifest_digest=manifest["manifest_digest"],
                          backbone="fixture",variant=f"st_{task}")
            seed_runs[task].append(result)
            _json(root/"seed_metrics"/f"{task}-{seed}.json",result)
    summary=combine_st_runs(seed_runs["sentiment"],seed_runs["topic"],backbone="fixture",
                            pairing_id="predeclared-fixture-st-row",expected_seeds=range(42,47))
    _json(root/"st_five_seed_summary.fixture.json",summary)
    write_summary_csv(summary,root/"st_five_seed_summary.fixture.csv",row_id="fixture",
                      backbone="fixture",variant="single_task_pair")
    # Demonstrate failure before any metric output on a missing record.
    missing=root/"negative"/"predictions"/"test.jsonl"
    missing.parent.mkdir(parents=True)
    missing.write_text("".join(json.dumps(r)+"\n" for r in base[:-1]),encoding="utf-8")
    target=root/"negative"/"evaluation"/"test"
    command=[sys.executable,"-m","neu_esc.evaluation.evaluate","--predictions",str(missing),
             "--dataset-manifest",str(manifest_path),"--output",str(target),"--tasks","sentiment","topic"]
    result=subprocess.run(command,capture_output=True,text=True)
    checks.append(dict(name="missing_record_rejected",command=command,exit_code=result.returncode,
                       stdout=result.stdout,stderr=result.stderr,metrics_output_exists=target.exists()))
    assert result.returncode==2 and not target.exists()
    _json(root/"checks.json",dict(fixture_only=True,benchmark_results=False,checks=checks))
    (root/"README.md").write_text(
        "# Evaluation fixtures only\n\nThese outputs use four synthetic samples, the real data load_split API, and evaluator CLI. No training or LLM API calls occurred. They are not NEU-ESC benchmark results. The negative coverage case must exit 2 and create no metrics.\n",encoding="utf-8")
    print(json.dumps({"output":str(root),"fixture_only":True,"checks":len(checks)}))


if __name__=="__main__":
    main()
