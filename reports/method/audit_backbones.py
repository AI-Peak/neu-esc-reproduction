"""Read-only upstream audit; stores evidence only in module 02 owned directories.

Run from repo root. Downloads tokenizers/configs, never trains or downloads full weights.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import requests
import yaml
from transformers import AutoTokenizer

MODELS = {
    "mbert": ("google-bert/bert-base-multilingual-uncased", "7cbf9a625e29989f6b9c6c2fa68234c304f7e38f"),
    "velectra": ("FPTAI/velectra-base-discriminator-cased", "f594fee501933d92d8c13285c969495681d0cc00"),
    "xlm_r": ("FacebookAI/xlm-roberta-base", "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"),
    "visobert": ("uitnlp/visobert", "196a62afad9cbe4f52a54aabad828b13f0eec59a"),
    "phobert": ("vinai/phobert-base-v2", "86cd7fd4c148980922ac11a2cf5e257f2ba639e1"),
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    output = Path("reports/method/backbones")
    output.mkdir(parents=True, exist_ok=True)
    registry = {"audit_time_utc": datetime.now(timezone.utc).isoformat(), "backbones": {}}
    for slug, (model_id, revision) in MODELS.items():
        response = requests.get(f"https://huggingface.co/{model_id}/resolve/{revision}/config.json", timeout=60)
        response.raise_for_status()
        (output / f"{slug}.config.json").write_bytes(response.content)
        config = response.json()
        metadata = requests.get(f"https://huggingface.co/api/models/{model_id}/revision/{revision}", timeout=60)
        metadata.raise_for_status()
        (output / f"{slug}.metadata.json").write_text(json.dumps(metadata.json(), indent=2) + "\n", encoding="utf-8")
        item = {"model_id": model_id, "model_revision": revision, "tokenizer_id": model_id,
                "tokenizer_revision": revision, "config_sha256": sha(response.content),
                "encoder_hidden_size": config["hidden_size"], "vocab_size": config["vocab_size"],
                "model_type": config["model_type"], "max_position_embeddings": config["max_position_embeddings"],
                "hidden_dropout": config["hidden_dropout_prob"],
                "attention_dropout": config["attention_probs_dropout_prob"],
                "weights_loaded": False, "paper_revision_known": False}
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=False, trust_remote_code=False)
            directory = output / slug / "tokenizer"
            tokenizer.save_pretrained(directory)
            text = "Sinh_viên học_tập tốt ." if slug == "phobert" else "Sinh viên học tập tốt."
            tokens = tokenizer(text, max_length=128, padding="max_length", truncation=True)
            assert max(tokens["input_ids"]) < config["vocab_size"]
            item.update(tokenizer_verified=True, tokenizer_class=type(tokenizer).__name__,
                        special_token_ids=tokenizer.all_special_ids, pad_token_id=tokenizer.pad_token_id,
                        mask_token_id=tokenizer.mask_token_id,
                        tokenizer_files_sha256={p.name: sha(p.read_bytes()) for p in directory.iterdir() if p.is_file()})
        except Exception as exc:
            item.update(tokenizer_verified=False, tokenizer_error=f"{type(exc).__name__}: {exc}")
        registry["backbones"][slug] = item
        print(slug, "tokenizer_verified=", item["tokenizer_verified"])
    (output / "audit.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path("configs/models").mkdir(parents=True, exist_ok=True)
    for slug, item in registry["backbones"].items():
        Path(f"configs/models/{slug}.yaml").write_text(yaml.safe_dump(item, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
