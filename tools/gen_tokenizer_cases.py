"""Generates text->ids ground truth from the REAL GLM-5.2 tokenizer (`tokenizers` library,
not rabbit's own port), for `tests/tokenizer_fixture.rs` to check `src/tokenizer.rs` against.

Same spirit as colibri's `tests/test_tok.c` (`TEXT\tID,ID,..` cases from an HF oracle) but as
JSON, since that's what the Rust side already speaks fluently.

Usage:
    pip install tokenizers
    python3 tools/fetch_tokenizer_fixture.py       # downloads tests/fixtures/tokenizer.json
    python3 tools/gen_tokenizer_cases.py            # -> tests/fixtures/tokenizer_cases.json
"""

import json
import os

from tokenizers import Tokenizer

HERE = os.path.dirname(__file__)
FIXTURES = os.path.join(HERE, "..", "tests", "fixtures")

CASES = [
    # plain ASCII, whitespace-prefixed words (GPT2/ByteLevel convention)
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    # contractions: exercises pretokenizer alternative 1
    "I'll say don't, can't, we're, and they've.",
    # numbers: alternative 3 (1-3 digit runs)
    "In 2024 there were 123456 events, or 3.14159 pi.",
    # punctuation runs + trailing newline: alternative 4
    "Hi, world!\n\nNext paragraph.\tTabbed.",
    # whitespace edge cases: alternative 5/6
    "end   ",
    "a  b\n\n\nc",
    # accented Latin / non-ASCII single-byte-adjacent codepoints
    "café naïve résumé Zürich",
    # CJK (each char its own \p{L} pretoken run in this pattern, byte-level BPE underneath)
    "你好世界，欢迎使用GLM-5.2！",
    "こんにちは、世界",
    # emoji / astral-plane codepoints (4-byte UTF-8, byte-level fallback)
    "🎉🚀 emoji test 😀",
    # mixed code-like text with symbols
    "def foo(x: int) -> bool:\n    return x > 0  # comment",
    # GLM-5.2 special/added tokens mixed with normal text
    "<|user|>Hello<|assistant|>Hi there<|endoftext|>",
    "<think>reasoning about 42</think>final answer",
    "<|system|>You are helpful.<|observation|>result: 7",
    # empty and single-char edges
    "",
    "x",
    " ",
]


def main():
    tok = Tokenizer.from_file(os.path.join(FIXTURES, "tokenizer.json"))
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
