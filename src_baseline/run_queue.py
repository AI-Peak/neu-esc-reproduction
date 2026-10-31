"""Run backbone x setup x seed jobs, one process per GPU (Kaggle T4 x2 -> two at a time).

    python src_baseline/run_queue.py --data-dir DATA --out-dir runs --seeds 42
    python src_baseline/run_queue.py --data-dir DATA --out-dir smoke --seeds 42 -- --epochs 1 --max-steps-per-epoch 30

Arguments after `--` are passed to every `neu_esc_kaggle.py train` call. Runs whose
metrics.json is already COMPLETED are skipped by the trainer, so a queue can be relaunched
after an interrupted session. Jobs start longest-first so both GPUs finish close together.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "neu_esc_kaggle.py"
BACKBONES = ("phobert", "visobert")
SETUPS = ("st_sentiment", "st_topic", "mtl", "mtl_smart", "mtl_mlm", "mtl_mlm_smart")
# Relative cost per run, used only for ordering: SMART adds two forward+backward passes.
COST = {"mtl_smart": 3.0, "mtl": 1.0, "st_sentiment": 0.6, "st_topic": 0.6, "mtl_mlm": 2.0, "mtl_mlm_smart": 4.2}


def visible_gpus():
    try:
        import torch
        return [str(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        return []


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    passthrough = []
    if "--" in argv:
        split = argv.index("--")
        argv, passthrough = argv[:split], argv[split + 1:]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, help="with --backbones/--setups: every combination")
    parser.add_argument("--jobs", nargs="+", metavar="BACKBONE:SETUP:SEED",
                        help="explicit job list instead of the seeds x backbones x setups grid")
    parser.add_argument("--backbones", nargs="+", default=list(BACKBONES), choices=BACKBONES)
    parser.add_argument("--setups", nargs="+", default=list(SETUPS), choices=SETUPS)
    parser.add_argument("--gpus", nargs="+", help="GPU indices; default: all visible (CPU if none)")
    parser.add_argument("--no-prefetch", action="store_true", help="skip caching models before launch")
    args = parser.parse_args(argv)

    gpus = args.gpus or visible_gpus() or [""]
    if args.jobs:
        jobs = []
        for spec in args.jobs:
            backbone, setup, seed = spec.split(":")
            if backbone not in BACKBONES or setup not in SETUPS:
                parser.error(f"bad job {spec}")
            jobs.append((backbone, setup, int(seed)))
    elif args.seeds:
        jobs = [(b, s, seed) for seed in args.seeds for b in args.backbones for s in args.setups]
    else:
        parser.error("give --seeds or --jobs")
    backbones = sorted({job[0] for job in jobs})
    jobs.sort(key=lambda job: -COST[job[1]])
    out_dir = Path(args.out_dir)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_prefetch:
        # Cache each model once so two processes never download the same files at once.
        subprocess.run([sys.executable, str(RUNNER), "prefetch", "--backbones", *backbones], check=True)

    lock = threading.Lock()
    results = []
    started = time.time()

    def worker(gpu):
        while True:
            with lock:
                if not jobs:
                    return
                backbone, setup, seed = jobs.pop(0)
            name = f"{backbone}-{setup}-s{seed}"
            command = [sys.executable, str(RUNNER), "train", "--backbone", backbone, "--setup", setup,
                       "--seed", str(seed), "--data-dir", args.data_dir, "--out-dir", args.out_dir,
                       *passthrough]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1")
            print(f"[{time.strftime('%H:%M:%S')}] gpu{gpu or '-cpu'} start {name}", flush=True)
            job_start = time.time()
            with open(log_dir / f"{name}.log", "a", encoding="utf-8") as stream:
                code = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
            minutes = (time.time() - job_start) / 60
            state = "ok" if code == 0 else f"FAILED (exit {code}, see logs/{name}.log)"
            print(f"[{time.strftime('%H:%M:%S')}] gpu{gpu or '-cpu'} done  {name} {minutes:.1f} min {state}",
                  flush=True)
            with lock:
                results.append((name, code, minutes))

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    failed = [name for name, code, _ in results if code != 0]
    print(f"queue finished in {(time.time() - started) / 3600:.2f} h; {len(results) - len(failed)} ok, "
          f"{len(failed)} failed{': ' + ', '.join(failed) if failed else ''}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
