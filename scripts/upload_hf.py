#!/usr/bin/env python3
"""Publish cards and data to the Hugging Face Hub (tknika org).

    python scripts/upload_hf.py cards            # model + dataset cards only
    python scripts/upload_hf.py heldout-test     # add the held-out test file
    python scripts/upload_hf.py model PATH       # upload a merged model folder

File names inside the dataset repo keep the names used at its first
release (pii-sintetikoa-v2-*.jsonl) so existing links keep working.
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
MODEL_V1 = "tknika/gliner2-PII-basque"
MODEL_V2 = "tknika/gliner2-PII-basque-v2"
DATASET_V2 = "tknika/pii-synthetic-basque-v2"


def upload_cards(api, message):
    api.upload_file(path_or_fileobj=ROOT / "cards" / "model-card-v2.md",
                    path_in_repo="README.md", repo_id=MODEL_V2,
                    commit_message=message)
    api.upload_file(path_or_fileobj=ROOT / "cards" / "model-card-v1.md",
                    path_in_repo="README.md", repo_id=MODEL_V1,
                    commit_message=message)
    api.upload_file(path_or_fileobj=ROOT / "cards" / "dataset-card-v2.md",
                    path_in_repo="README.md", repo_id=DATASET_V2,
                    repo_type="dataset", commit_message=message)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["cards", "heldout-test", "model"])
    ap.add_argument("path", nargs="?", type=Path)
    ap.add_argument("--message", default="Update card")
    args = ap.parse_args()
    api = HfApi()
    if args.what == "cards":
        upload_cards(api, args.message)
    elif args.what == "heldout-test":
        api.upload_file(
            path_or_fileobj=ROOT / "data/synthetic/v2-test-heldout-templates.jsonl",
            path_in_repo="pii-sintetikoa-v2-test-heldout-templates.jsonl",
            repo_id=DATASET_V2, repo_type="dataset",
            commit_message="Add held-out-template test split")
    else:
        if not args.path:
            ap.error("model needs the merged model folder")
        api.upload_folder(folder_path=args.path, repo_id=MODEL_V2,
                          ignore_patterns=["README.md", ".cache/*"],
                          commit_message=args.message)


if __name__ == "__main__":
    main()
