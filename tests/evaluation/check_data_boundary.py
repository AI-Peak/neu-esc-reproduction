"""Read the real prepared split and verify coverage gate without predictions.

This is an interface check, not a benchmark result. No predictions are generated.
"""

import argparse
import hashlib
import json
from pathlib import Path

from neu_esc.data.dataset import load_split
from neu_esc.evaluation.evaluate import EvaluationError, _manifest_digest, evaluate_records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest",type=Path,default=Path("data/manifests/dataset.json"))
    parser.add_argument("--output",type=Path,default=Path("runs/smoke/evaluation/real_data_boundary_check.json"))
    args=parser.parse_args()
    manifest_bytes=args.dataset_manifest.read_bytes()
    manifest=json.loads(manifest_bytes)
    digest=_manifest_digest(manifest)
    repo_root=Path(__file__).resolve().parents[2]
    implementation_paths=[repo_root/"src/neu_esc/data/dataset.py",repo_root/"src/neu_esc/data/common.py"]
    implementation_hashes={p.relative_to(repo_root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in implementation_paths}
    checks={}
    for split in ("validation","test"):
        expected=load_split(args.dataset_manifest,split,"raw")
        try:
            evaluate_records([],expected,["sentiment","topic"])
        except EvaluationError as exc:
            assert "missing prediction ID/task coverage" in str(exc)
            assert str(2*len(expected)) in str(exc)
            checks[split]=dict(n_samples=len(expected), expected_id_task_pairs=2*len(expected),
                               status="coverage_rejected_as_expected",error=str(exc),metrics_created=False)
        else:
            raise AssertionError("empty predictions must fail")
    assert args.dataset_manifest.read_bytes()==manifest_bytes
    assert implementation_hashes=={p.relative_to(repo_root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in implementation_paths}
    result=dict(schema_version="1.0",benchmark_results=False,predictions_generated=False,
                dataset_manifest_digest=digest,manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                source_revision=manifest.get("source_revision"),
                data_implementation_sha256=implementation_hashes,
                hf_equivalence_status=manifest.get("hf_equivalence_status"),
                paper_equivalence_status=manifest.get("paper_equivalence_status"),
                processed_sha256=manifest.get("processed_sha256"),checks=checks)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf-8") as stream:
        stream.write(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"output":str(args.output),"benchmark_results":False,"splits":list(checks)}))


if __name__=="__main__":
    main()
