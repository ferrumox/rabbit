"""Generates text->ids ground truth from the REAL Kimi Linear tokenizer (`tiktoken` library,
not rabbit's own port), for `tests/kimi_tokenizer_fixture.rs` to check
`src/kimi_linear/tokenizer.rs` against — the sibling of `gen_tokenizer_cases.py`'s GLM-5.2
version.

The pre-tokenizer pattern (`PAT_STR` below) is copied verbatim from the real
`tokenization_kimi.py`'s `TikTokenTokenizer.pat_str` (fetched from
moonshotai/Kimi-Linear-48B-A3B-Instruct) — not reproduced from memory, since a single character
different here would silently produce a "ground truth" that doesn't match the real tokenizer at
all. The 258-entry special-token map is reconstructed the same way `TikTokenTokenizer.__init__`
does: 17 real names from `tokenizer_config.json`'s `added_tokens_decoder`, the rest fall back to
`<|reserved_token_{id}|>`.

Usage:
    pip install tiktoken
    python3 tools/fetch_kimi_tokenizer_fixture.py    # downloads tests/fixtures/kimi/{tiktoken.model,tokenizer_config.json}
    python3 tools/gen_kimi_tokenizer_cases.py         # -> tests/fixtures/kimi/tokenizer_cases.json
"""

import json
import os

import tiktoken
from tiktoken.load import load_tiktoken_bpe

HERE = os.path.dirname(__file__)
FIXTURES = os.path.join(HERE, "..", "tests", "fixtures", "kimi")

PAT_STR = "|".join(
    [
        r"""[\p{Han}]+""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+""",
    ]
)
NUM_RESERVED_SPECIAL_TOKENS = 256

ORDINARY_CASES = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "I'll say don't, can't, we're, and they've.",
    "In 2024 there were 123456 events, or 3.14159 pi.",
    "Hi, world!\n\nNext paragraph.\tTabbed.",
    "end   ",
    "a  b\n\n\nc",
    "café naïve résumé Zürich",
    "你好世界，欢迎使用Kimi Linear！",
    "こんにちは、世界",
    "🎉🚀 emoji test 😀",
    "def foo(x: int) -> bool:\n    return x > 0  # comment",
    "",
    "x",
    " ",
    "Mixed 中文 and English 混合 text here",
]
SPECIAL_CASES = [
    "<|im_user|>Hello<|im_middle|>Hi there<|im_end|>",
    "[BOS]hello[EOS]",
]


def main():
    mergeable_ranks = load_tiktoken_bpe(os.path.join(FIXTURES, "tiktoken.model"))
    num_base_tokens = len(mergeable_ranks)

    added_tokens_decoder = json.load(open(os.path.join(FIXTURES, "tokenizer_config.json")))["added_tokens_decoder"]
    special_tokens_mapping = {int(k): v["content"] for k, v in added_tokens_decoder.items()}
    special_tokens = {
        special_tokens_mapping.get(i, f"<|reserved_token_{i}|>"): i
        for i in range(num_base_tokens, num_base_tokens + NUM_RESERVED_SPECIAL_TOKENS + 2)
    }

    enc = tiktoken.Encoding(name="kimi", pat_str=PAT_STR, mergeable_ranks=mergeable_ranks, special_tokens=special_tokens)

    ordinary_cases = []
    for text in ORDINARY_CASES:
        ids = enc.encode_ordinary(text)
        ordinary_cases.append({"text": text, "ids": ids, "decoded": enc.decode(ids)})

    allowed = set(special_tokens.keys())
    special_cases = []
    for text in SPECIAL_CASES:
        ids = enc.encode(text, allowed_special=allowed)
        special_cases.append({"text": text, "ids": ids, "decoded": enc.decode(ids)})

    out_path = os.path.join(FIXTURES, "tokenizer_cases.json")
    json.dump({"ordinary_cases": ordinary_cases, "special_cases": special_cases}, open(out_path, "w"), ensure_ascii=False, indent=1)
    print(f"wrote {len(ordinary_cases) + len(special_cases)} cases -> {out_path}")


if __name__ == "__main__":
    main()
