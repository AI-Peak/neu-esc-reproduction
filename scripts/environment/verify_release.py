"""Verify a clean committed checkout using a fresh hash-locked CPU wheel install.

Run with a Python that has build/setuptools and uv on PATH. Outputs stay ignored.
This uses synthetic tiny models only; GPU and benchmark execution remain pending.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT), 'Checkout must be clean'
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    inventory = json.loads((ROOT / 'scripts/environment/release-inventory-20261012.json').read_text())
    for name, record in inventory['ready_paths'].items():
        assert sha(ROOT / name) == record['sha256'], name
    tracked = subprocess.check_output(['git', 'ls-files'], cwd=ROOT, text=True).splitlines()
    assert not any(name.startswith(('runs/', 'data/raw/', 'data/processed/', 'data/resources/', 'checkpoints/'))
                   or '/tokenizer/' in name for name in tracked)
    base = args.output.resolve()
    assert ROOT in base.parents, 'Output must be inside checkout ignored runs directory'
    assert base.relative_to(ROOT).parts[0] == 'runs'
    base.mkdir(parents=True, exist_ok=False)
    receipt = dict(status='RUNNING', commit=commit, checked_at_utc=datetime.now(timezone.utc).isoformat(),
                   clean_checkout=True, owner_hashes_verified=len(inventory['ready_paths']),
                   fixture_only=True, gpu='PENDING', benchmark_runs=0, commands=[])
    env = dict(os.environ)
    for name in ('PYTHONPATH', 'PYTHONHOME'):
        env.pop(name, None)
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled',
               CUDA_VISIBLE_DEVICES='', MPLBACKEND='Agg', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')

    def save():
        (base / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8', newline='\n')

    def run(name, argv, timeout=1800):
        print(f'CHECK {name}', flush=True)
        start = time.monotonic()
        result = subprocess.run(list(map(str, argv)), cwd=ROOT, env=env, capture_output=True,
                                text=True, encoding='utf-8', errors='replace', timeout=timeout)
        log = base / f'{name}.log'
        log.write_text(result.stdout + '\nSTDERR:\n' + result.stderr, encoding='utf-8', newline='\n')
        receipt['commands'].append(dict(name=name, argv=list(map(str, argv)), returncode=result.returncode,
                                        seconds=round(time.monotonic() - start, 2),
                                        log=log.relative_to(ROOT).as_posix(), log_sha256=sha(log)))
        save()
        if result.returncode:
            receipt['status'] = 'FAIL'
            save()
            raise RuntimeError(f'{name} failed: {log}\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}')
        print(f'PASS {name} ({receipt["commands"][-1]["seconds"]}s)', flush=True)
        return result.stdout

    run('build_wheel', [sys.executable, '-m', 'build', '--wheel', '--no-isolation', '--outdir', base / 'dist', ROOT])
    wheel = next((base / 'dist').glob('*.whl'))
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        for name in ('data/_java_bridge.py', 'training/train.py', 'evaluation/evaluate.py', 'models/encoder.py', 'losses/smart.py'):
            assert 'neu_esc/' + name in names, name
    receipt['wheel_sha256'] = sha(wheel)
    venv = base / '.venv'
    run('create_venv', [sys.executable, '-m', 'venv', venv])
    python = venv / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    run('install_hash_locked_cpu', [shutil.which('uv'), 'pip', 'install', '--python', python,
                                  '--require-hashes', '-r', ROOT / 'requirements-cpu.lock'])
    run('install_wheel', [python, '-m', 'pip', 'install', '--no-deps', wheel])
    run('pip_check', [python, '-m', 'pip', 'check'])
    run('assert_wheel_import', [python, '-c',
        'from pathlib import Path; import neu_esc,sys; p=Path(neu_esc.__file__).resolve(); '
        'assert Path(sys.prefix).resolve() in p.parents and "site-packages" in p.parts, p; print(p)'])
    run('pipeline_import_config_cli', [python, ROOT / 'scripts/environment/check_pipeline.py'])
    run('offline_cpu_tests', [python, ROOT / 'scripts/environment/offline_tests.py'])
    pipeline = base / 'pipeline'
    run('six_variant_cpu_pipeline', [python, ROOT / 'scripts/integration/audit_pipeline.py', '--output', pipeline])
    audit = json.loads((pipeline / 'receipt.json').read_text())
    assert audit['status'] == 'PASS' and len(audit['variants']) == 6
    for variant, record in audit['variants'].items():
        assert record['status'] == record['validation_aggregation'] == 'PASS', variant
        run('validation_' + variant, [python, ROOT / 'scripts/integration/reproduce_validation_metadata.py',
                                     '--run-dir', pipeline / variant])
    receipt['pipeline_variants'] = audit['variants']
    receipt['pipeline_receipt_sha256'] = sha(pipeline / 'receipt.json')
    receipt['execution_sha256'] = {name: sha(ROOT / name) for name in tracked
        if name.startswith(('src/', 'configs/', 'tests/', 'scripts/')) and not name.endswith(('.json', '.txt'))
        or name in ('pyproject.toml', 'requirements-cpu.lock', 'uv.lock', '.github/workflows/cpu.yml')}
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT), 'Tracked checkout changed during tests'
    receipt['status'] = 'PASS'
    save()
    print(json.dumps(dict(status='PASS', commit=commit, receipt=str(base / 'receipt.json'))), flush=True)


if __name__ == '__main__':
    main()
