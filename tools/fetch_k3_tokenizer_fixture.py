"""Downloads ONLY tiktoken.model + tokenizer_config.json (a few MB, no weights) from the real
Kimi K3 Hugging Face repo, for manually validating src/kimi_k3/tokenizer.rs against the real
~163,840-entry vocab — the sibling of `fetch_kimi_tokenizer_fixture.py`'s Kimi Linear 48B version.

Not part of `cargo test`: it needs network access, so it's a dev-time check, same spirit as that
script.

Usage:
    pip install huggingface_hub
    python3 tools/fetch_k3_tokenizer_fixture.py                # -> tests/fixtures/k3/
    python3 tools/fetch_k3_tokenizer_fixture.py --dest DIR
"""

import argparse
import os

from huggingface_hub import hf_hub_download

REPO = "moonshotai/Kimi-K3"
FILES = ["tiktoken.model", "tokenizer_config.json"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=REPO, help=f"HF repo id (default: {REPO})")
    ap.add_argument(
        "--dest",
        default=os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "k3"),
        help="destination directory (default: tests/fixtures/k3)",
    )
    args = ap.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    for name in FILES:
        path = hf_hub_download(repo_id=args.repo, filename=name, local_dir=args.dest)
        size = os.path.getsize(path)
        print(f"{name}: {size / 1e6:.1f} MB -> {path}")


if __name__ == "__main__":
    main()
