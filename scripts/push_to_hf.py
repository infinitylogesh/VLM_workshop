#!/usr/bin/env python3
"""Push a LoRA adapter directory to a PRIVATE Hugging Face repo.

Uses the cached HF token (user: infinitylogesh). Creates the repo if missing and
uploads the folder. A small README with provenance is added if none exists.
"""
import argparse
import os
import sys

from huggingface_hub import HfApi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="local adapter directory to upload")
    ap.add_argument("--repo", required=True, help="target repo id, e.g. infinitylogesh/vlm-workshop-...")
    ap.add_argument("--private", action="store_true", default=True)
    ap.add_argument("--public", dest="private", action="store_false")
    ap.add_argument("--note", default="", help="one-line provenance for the auto README")
    args = ap.parse_args()

    if not os.path.isdir(args.dir):
        print(f"[push] SKIP: {args.dir} does not exist", flush=True)
        sys.exit(0)  # non-fatal: never block the pipeline

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)

    readme = os.path.join(args.dir, "README.md")
    if not os.path.exists(readme):
        with open(readme, "w") as f:
            f.write(f"# {args.repo.split('/')[-1]}\n\n{args.note}\n\n"
                    "LoRA adapter from the VLM SFT+RL+distillation workshop.\n")

    api.upload_folder(folder_path=args.dir, repo_id=args.repo, repo_type="model")
    print(f"[push] uploaded {args.dir} -> https://huggingface.co/{args.repo} (private={args.private})", flush=True)


if __name__ == "__main__":
    main()
