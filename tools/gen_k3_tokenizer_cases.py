"""Generates text->ids ground truth from the REAL Kimi K3 tokenizer (`tiktoken` library, not
rabbit's own port), for `tests/k3_tokenizer_fixture.rs` to check `src/kimi_k3/tokenizer.rs`
against — the sibling of `gen_kimi_tokenizer_cases.py`'s Kimi Linear 48B version.

The pre-tokenizer pattern (`PAT_STR` below) is copied verbatim from the real
`tokenization_kimi.py`'s `TikTokenTokenizer.pat_str` (fetched from moonshotai/Kimi-K3,
2026-07-27) — confirmed character-for-character identical to Kimi Linear 48B's own, same real
source file backs both. **One real difference from `gen_kimi_tokenizer_cases.py`**: this uses
EXACTLY `NUM_RESERVED_SPECIAL_TOKENS = 256` (not that script's `+2`-adjusted 258) — re-read from
`tokenization_kimi.py`'s own `num_reserved_special_tokens = 256` class attribute this session,
and confirmed against K3's real `tokenizer_config.json`: its `added_tokens_decoder`'s highest key
is `num_base_tokens + 255`, safely inside a plain 256-entry range (see `src/kimi_k3/tokenizer.rs`'s
module doc for the full reasoning).

Usage:
    pip install tiktoken
    python3 tools/fetch_k3_tokenizer_fixture.py    # downloads tests/fixtures/k3/{tiktoken.model,tokenizer_config.json}
    python3 tools/gen_k3_tokenizer_cases.py         # -> tests/fixtures/k3/tokenizer_cases.json
"""

import json
import os

import tiktoken
from tiktoken.load import load_tiktoken_bpe

HERE = os.path.dirname(__file__)
FIXTURES = os.path.join(HERE, "..", "tests", "fixtures", "k3")

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
    "你好世界，欢迎使用Kimi K3！",
    "こんにちは、世界",
    "🎉🚀 emoji test 😀",
    "def foo(x: int) -> bool:\n    return x > 0  # comment",
    "",
    "x",
    " ",
    "Mixed 中文 and English 混合 text here",
]
SPECIAL_CASES = [
    "<|open|>message role=\"user\"<|sep|>Hello<|close|>message<|sep|><|end_of_msg|>",
    "[BOS]hello[EOS]",
]


def main():
    mergeable_ranks = load_tiktoken_bpe(os.path.join(FIXTURES, "tiktoken.model"))
    num_base_tokens = len(mergeable_ranks)

    added_tokens_decoder = json.load(open(os.path.join(FIXTURES, "tokenizer_config.json")))["added_tokens_decoder"]
    special_tokens_mapping = {int(k): v["content"] for k, v in added_tokens_decoder.items()}
    special_tokens = {
        special_tokens_mapping.get(i, f"<|reserved_token_{i}|>"): i
        for i in range(num_base_tokens, num_base_tokens + NUM_RESERVED_SPECIAL_TOKENS)
    }

    enc = tiktoken.Encoding(name="k3", pat_str=PAT_STR, mergeable_ranks=mergeable_ranks, special_tokens=special_tokens)

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
