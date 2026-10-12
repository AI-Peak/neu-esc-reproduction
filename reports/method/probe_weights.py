"""Probe a bounded prefix of pinned pretrained files; never train or save weights."""
import json
from pathlib import Path

import requests


def main():
    audit = json.loads(Path("reports/method/backbones/audit.json").read_text(encoding="utf-8"))
    result = {"full_weights_loaded": False, "backbones": {}}
    for slug, model in audit["backbones"].items():
        metadata = requests.get(f"https://huggingface.co/api/models/{model['model_id']}/revision/{model['model_revision']}?blobs=true", timeout=60)
        metadata.raise_for_status()
        siblings = {row["rfilename"]: row for row in metadata.json()["siblings"]}
        filename = next((name for name in ("model.safetensors", "pytorch_model.bin") if name in siblings), None)
        if filename is None:
            result["backbones"][slug] = {"access_verified": False, "error": "No single weight file found"}
            continue
        url = f"https://huggingface.co/{model['model_id']}/resolve/{model['model_revision']}/{filename}"
        try:
            # Only consume 16 bytes, including when the server ignores the range header.
            with requests.get(url, headers={"Range": "bytes=0-15"}, stream=True, timeout=60) as response:
                response.raise_for_status()
                prefix = next(response.iter_content(chunk_size=16))
                item = {"access_verified": bool(prefix), "http_status": response.status_code,
                        "model_id": model["model_id"], "revision": model["model_revision"],
                        "filename": filename, "consumed_bytes": len(prefix),
                        "repository_file_metadata": siblings[filename]}
        except Exception as exc:
            item = {"access_verified": False, "error": f"{type(exc).__name__}: {exc}"}
        result["backbones"][slug] = item
        print(slug, "weight_access=", item["access_verified"])
        Path("reports/method/weights_access.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
