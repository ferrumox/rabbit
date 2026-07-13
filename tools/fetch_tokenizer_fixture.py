"""Downloads ONLY tokenizer.json + config.json (a few MB, no weights) from the real GLM-5.2
Hugging Face repo, for manually validating src/tokenizer.rs against the real vocab/merges —
as opposed to tests/oracle/make_glm_oracle.py's tiny random model (vocab_size=256), which
exists to validate the forward pass, not the tokenizer.

Not part of `cargo test`: it needs network access and the real ~150k-entry vocab, so it's a
dev-time check, same spirit as the reference implementation's tests/test_tok.c (see rabbit-plan.md, Fase 2).

Usage:
    pip install huggingface_hub
    python3 tools/fetch_tokenizer_fixture.py                # -> tests/fixtures/
    python3 tools/fetch_tokenizer_fixture.py --dest DIR
"""

import argparse
import os

from huggingface_hub import hf_hub_download

REPO = "zai-org/GLM-5.2"
FILES = ["tokenizer.json", "config.json"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=REPO, help=f"HF repo id (default: {REPO})")
    ap.add_argument(
        "--dest",
        default=os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures"),
        help="destination directory (default: tests/fixtures)",
    )
    args = ap.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    for name in FILES:
        path = hf_hub_download(repo_id=args.repo, filename=name, local_dir=args.dest)
        size = os.path.getsize(path)
        print(f"{name}: {size / 1e6:.1f} MB -> {path}")


if __name__ == "__main__":
    main()
