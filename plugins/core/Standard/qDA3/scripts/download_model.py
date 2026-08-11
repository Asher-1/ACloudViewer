#!/usr/bin/env python3
"""Download DA3-BASE weights from HuggingFace into models/DA3-BASE."""
import argparse
from huggingface_hub import snapshot_download

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="depth-anything/DA3-BASE")
    ap.add_argument("--out", default="models/DA3-BASE")
    a = ap.parse_args()
    p = snapshot_download(repo_id=a.repo, local_dir=a.out)
    print("downloaded to", p)

if __name__ == "__main__":
    main()
