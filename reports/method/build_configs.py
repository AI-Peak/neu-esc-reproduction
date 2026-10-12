"""Generate the 22 one-seed encoder configs. Does not schedule/execute training."""
import copy
import json
from pathlib import Path

import yaml

from neu_esc.training.config import validate_config


def main():
    audit = json.loads(Path("reports/method/backbones/audit.json").read_text(encoding="utf-8"))
    manifest_path = Path("data/manifests/dataset.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    base_assumptions = [
        "Dataset is the pinned GitHub alternative revision and current VnCoreNLP segmentation from module 01; equivalence to gated HF/paper revision remains unverified.",
        "Paper does not identify checkpoint revisions. These immutable upstream SHAs were audited on 2026-10-11; paper-time revisions remain unknown.",
        "Paper omits training hyperparameters. AdamW lr=2e-5, length=128, ST=15 epochs, MTL=20 epochs follow public scripts; other resolved settings are explicit assumptions.",
        "MTL classification sums mean CE with task weights 1/1 following paper; public 2-head trainer averages 0.5/0.5 instead.",
        "Physical batch 8, accumulation 16, effective batch 128 is a Colab memory assumption; microbatch means (MLM/SMART included) are sample-weighted. This differs from physical batch 128 with dropout.",
        "MTL uses first hidden token then Linear/SiLU/LayerNorm; public head dropout 0.2 is declared but unused, so effective extra head dropout=0.0.",
        "ST uses each backbone's native AutoModelForSequenceClassification head as in public base_trainer; it does not reuse the MTL shared projection.",
        "SMART uses independent continuous embedding adversaries, linf radius 1e-3, noise std 1e-5, step 1e-3, 1 step, gradient epsilon 1e-6; radius is not specified in paper.",
        "SMART weight 0.02 follows repo; batchmean symmetric KL replaces repo final sum reduction. This is a documented normalization assumption, not an exact public-code reproduction.",
        "SMART disables dropout for clean/adversarial regularizer forwards and excludes special/pad tokens. Supervised/MLM forwards keep native encoder dropout 0.1.",
        "MLM uses independent Bernoulli token selection at p=0.30 and selected-token 80/10/10, excluding special/pad IDs; random replacement also excludes special IDs.",
        "MLM raw CE is sum per sentence. Divide by mean selected tokens per sentence once, yielding sum CE / selected count; public trainer divides an already mean CE again.",
        "MLM head is random untied Linear/SiLU/LayerNorm/Linear(hidden,vocab_size), not pretrained MLM weights; original code hardcodes 64001, unsuitable for ViSoBERT/XLM-R.",
        "Validation checkpoint: max mean task macro-F1, fixed 4/10 labels, earliest epoch wins ties. No test selection, no early stopping, no class weighting.",
        "Epoch-linear scheduler ends at 0.2 for ST and 0.1 for MTL; full epochs are used despite inconsistent public script comments.",
        "fp32, deterministic algorithms, num_workers=0, num_threads=1, TF32 disabled; no clipping or warmup; AdamW weight decay=0.01 and betas=0.9/0.999 are saved defaults from public optimizer usage.",
    ]
    count = 0
    for slug, model in audit["backbones"].items():
        variants = ["st_sentiment", "st_topic"]
        if slug in ("xlm_r", "visobert", "phobert"):
            variants += ["mtl", "mtl_smart", "mtl_mlm", "mtl_mlm_smart"]
        for variant in variants:
            single = variant.startswith("st_")
            tasks = [variant[3:]] if single else ["sentiment", "topic"]
            config = {"schema_version": "1.0", "experiment_id": f"{slug}-{variant}",
                      "backbone": slug, **{k: model[k] for k in
                          ("model_id", "model_revision", "tokenizer_id", "tokenizer_revision")},
                      "variant": variant, "tasks": tasks, "method_mode": "paper_method",
                      "dataset_manifest": "data/manifests/dataset.json",
                      "dataset_manifest_digest": manifest["manifest_digest"] if manifest else None,
                      "text_variant": "segmented" if slug == "phobert" else "raw", "max_length": 128,
                      "seed": 42,
                      "training": {"mode": "full", "epochs": 15 if single else 20, "max_steps": None,
                                   "batch_size": 8, "gradient_accumulation_steps": 16, "effective_batch_size": 128,
                                   "eval_batch_size": 32, "optimizer": "adamw", "learning_rate": 2e-5,
                                   "adam_epsilon": 1e-8 if single else 5e-9, "betas": [0.9, 0.999],
                                   "weight_decay": 0.01, "scheduler": "linear_epoch",
                                   "scheduler_end_factor": 0.2 if single else 0.1, "warmup_steps": 0,
                                   "precision": "fp32", "gradient_clipping": None, "head_dropout": 0.0,
                                   "encoder_hidden_dropout": model["hidden_dropout"],
                                   "encoder_attention_dropout": model["attention_dropout"],
                                   "st_native_head_dropout": 0.1 if single else None,
                                   "head_layer_norm_eps": 1e-5, "pooling": "first_token",
                                   "classification_reduction": "mean", "task_weights": {t: 1.0 for t in tasks},
                                   "tokenizer_use_fast": False, "padding": "max_length", "num_workers": 0,
                                   "num_threads": 1, "deterministic": True, "checkpoint_every_steps": 50,
                                   "smoke_sample_limit": 8},
                      "smart": {"enabled": "smart" in variant, "epsilon": 1e-3, "steps": 1,
                                "step_size": 1e-3, "gradient_eps": 1e-6, "weight": 0.02, "noise_std": 1e-5,
                                "norm": "linf_per_token", "reduction": "batchmean",
                                "dropout_policy": "eval_for_regularizer", "exclude_special_pad": True,
                                "independent_task_noise": True},
                      "mlm": {"enabled": "mlm" in variant, "mask_rate": 0.3,
                              "selected_distribution": [0.8, 0.1, 0.1], "weight": 1.0,
                              "reduction": "sum_then_normalize", "denominator": "selected_tokens",
                              "input_policy": "separate_masked_forward", "exclude_special_pad": True,
                              "head_initialization": "random_untied", "empty_target_policy": "zero_skip_forward"},
                      "selection": {"metric": "macro_f1", "aggregation": "mean_tasks",
                                    "split": "validation", "tie_break": "earliest_epoch"},
                      "assumptions": copy.deepcopy(base_assumptions)}
            validate_config(config)
            path = Path(f"configs/experiments/{slug}/{variant}.yaml")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
            count += 1
            if slug == "visobert" and variant == "mtl_mlm_smart":
                tiny = copy.deepcopy(config)
                tiny.update(experiment_id="tiny-cpu-smoke", max_length=12)
                tiny["training"].update(mode="smoke", epochs=2, batch_size=2,
                                        gradient_accumulation_steps=2, effective_batch_size=4,
                                        eval_batch_size=2, learning_rate=0.001, checkpoint_every_steps=1)
                tiny["assumptions"].append("SMOKE ONLY: initialized tiny BERT and local 32-token vocabulary; not a pretrained ViSoBERT result.")
                path = Path("tests/training/fixtures/tiny.yaml")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(yaml.safe_dump(tiny, sort_keys=False), encoding="utf-8")
    assert count == 22
    print("Generated", count, "full templates; dataset_locked=", manifest is not None)


if __name__ == "__main__":
    main()
