"""Generates text->ids ground truth from the REAL Qwen 3.8 tokenizer (`tokenizers` library, not
rabbit's own port), for `tests/qwen38_tokenizer_fixture.rs` to check `src/tokenizer.rs` +
`src/qwen38/tokenizer.rs` against.

Same shape and spirit as `tools/gen_tokenizer_cases.py` (GLM-5.2's), with cases chosen for the
places Qwen's `tokenizer.json` genuinely differs from GLM's: `ignore_merges: false`, a
`pre_tokenizer` that splits every digit into its own piece and groups combining marks with their
base letter, and ChatML special tokens.

The tokenizer.json is COPIED from the local checkpoint rather than downloaded (it already lands
there long before the 213 MXFP4 shards do):

    mkdir -p tests/fixtures/qwen38
    cp /mnt/data/qwen38-max-mxfp4/tokenizer.json tests/fixtures/qwen38/

Then, since this host's system pip is unusable, run the generator in Docker:

    docker run --rm -v "$PWD:/w" -w /w python:3.12-slim \
        bash -c "pip install -q tokenizers && python3 tools/gen_qwen38_tokenizer_cases.py"
"""

import json
import os

from tokenizers import Tokenizer

HERE = os.path.dirname(__file__)
FIXTURES = os.path.join(HERE, "..", "tests", "fixtures", "qwen38")

CASES = [
    # plain ASCII, whitespace-prefixed words (GPT2/ByteLevel convention)
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    # contractions: pretokenizer alternative 1, identical to cl100k's
    "I'll say don't, can't, we're, and they've.",
    # DIGITS: Qwen splits every digit on its own (`\p{N}`), GLM takes up to three (`\p{N}{1,3}`).
    # The single most likely place a wrong pre-tokenizer shows up in real prompts.
    "In 2024 there were 123456 events, or 3.14159 pi.",
    "1 12 123 1234 12345",
    "0x1F 2^32 99.99% -7",
    # punctuation runs + newlines/tabs: alternative 4
    "Hi, world!\n\nNext paragraph.\tTabbed.",
    # whitespace edges: alternatives 5/6/7
    "end   ",
    "a  b\n\n\nc",
    "   leading",
    # precomposed accents (NFC already) — Spanish/German text, the common real case here
    "café naïve résumé Zürich",
    "¿Qué tal? ¡Hola! Año 1998, español.",
    # MIXED normalization: "cafe" + U+0301 (a DECOMPOSED combining acute) followed by a
    # PRECOMPOSED "naïve" (U+00EF). Qwen's `[\p{L}\p{M}]+` keeps the mark with its base letter, and
    # this is also the one case rabbit's port knowingly differs on, since the file's NFC normalizer
    # isn't implemented — `tests/qwen38_tokenizer_fixture.rs` reports that as a known difference
    # instead of a failure, and asserts it still round-trips byte-for-byte.
    "café naïve",
    # CJK and Japanese
    "你好世界，欢迎使用 Qwen 3.8！",
    "こんにちは、世界",
    # emoji / astral-plane codepoints (4-byte UTF-8 through the byte-level map)
    "🎉🚀 emoji test 😀",
    # code-like text
    "def foo(x: int) -> bool:\n    return x > 0  # comment",
    "fn main() { let v: Vec<i32> = (0..10).collect(); }",
    # ChatML special/added tokens mixed with normal text, including the think block
    "<|im_start|>user\nHola<|im_end|>\n",
    "<|im_start|>assistant\n<think>\nrazonando sobre 42\n</think>\nlista<|im_end|>",
    "<|endoftext|>",
    # empty and single-char edges
    "",
    "x",
    " ",
]


def main():
    path = os.path.join(FIXTURES, "tokenizer.json")
    tok = Tokenizer.from_file(path)
    cases = []
    for text in CASES:
        ids = tok.encode(text, add_special_tokens=False).ids
        decoded = tok.decode(ids, skip_special_tokens=False)
        cases.append({"text": text, "ids": ids, "decoded": decoded})

    out_path = os.path.join(FIXTURES, "tokenizer_cases.json")
    json.dump(cases, open(out_path, "w"), ensure_ascii=False, indent=1)
    print(f"wrote {len(cases)} cases -> {out_path}")


if __name__ == "__main__":
    main()
