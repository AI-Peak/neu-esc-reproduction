"""Install only the hash-locked segmentation resources (no pretrained encoder)."""

import argparse

from .common import read_config
from .segmentation import ensure_resources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    config, root = read_config(args.config)
    lock_path = root / config["preprocessing"]["segmentation"]["resources_lock"]
    lock, _ = ensure_resources(root, lock_path, download=not args.offline)
    print(f"Verified VnCoreNLP {lock['version']} at {lock['revision']}; {len(lock['files'])} locked files")


if __name__ == "__main__":
    main()
