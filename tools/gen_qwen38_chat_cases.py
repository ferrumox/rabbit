"""Renders Qwen 3.8's REAL `chat_template.jinja` with Jinja2 to produce ground truth for
`tests/qwen38_chat_template_fixture.rs`, which checks `src/qwen38/chat_template.rs`'s hand port
against it string-for-string.

The Jinja environment mirrors what `transformers` itself uses to compile chat templates
(`ImmutableSandboxedEnvironment` with `trim_blocks`/`lstrip_blocks` on, plus a `raise_exception`
global and a `tojson` filter), so whitespace comes out exactly as it does in production.

Only the text-only subset rabbit implements is covered: no `tools`, no `tool_calls`, no image/video
content parts. `reasoning_effort` stays at the template's own default (`xhigh`) because rabbit's
public API exposes no override; the `medium`/`low` texts are pinned by unit tests in the Rust module.

Usage (the checkpoint's chat_template.jinja must already be in tests/fixtures/qwen38/):

    mkdir -p tests/fixtures/qwen38
    cp /mnt/data/qwen38-max-mxfp4/chat_template.jinja tests/fixtures/qwen38/
    docker run --rm -v "$PWD:/w" -w /w python:3.12-slim \
        bash -c "pip install -q jinja2 && python3 tools/gen_qwen38_chat_cases.py"
"""

import json
import os

from jinja2.sandbox import ImmutableSandboxedEnvironment

HERE = os.path.dirname(__file__)
FIXTURES = os.path.join(HERE, "..", "tests", "fixtures", "qwen38")

# (name, messages, enable_thinking) — `add_generation_prompt` is True except where the history
# already ends with an assistant turn, matching what rabbit's `render_messages` does.
CASES = [
    ("user_only_thinking", [{"role": "user", "content": "hola"}], True),
    ("user_only_no_thinking", [{"role": "user", "content": "hola"}], False),
    (
        "system_and_user_thinking",
        [{"role": "system", "content": "sos conciso"}, {"role": "user", "content": "hola"}],
        True,
    ),
    (
        "system_and_user_no_thinking",
        [{"role": "system", "content": "sos conciso"}, {"role": "user", "content": "hola"}],
        False,
    ),
    (
        "multi_turn_thinking",
        [
            {"role": "system", "content": "sos conciso"},
            {"role": "user", "content": "hola"},
            {"role": "assistant", "content": "buenas"},
            {"role": "user", "content": "y ahora?"},
        ],
        True,
    ),
    (
        "multi_turn_no_thinking",
        [
            {"role": "user", "content": "hola"},
            {"role": "assistant", "content": "buenas"},
            {"role": "user", "content": "y ahora?"},
        ],
        False,
    ),
    # content that needs trimming, and multi-line content
    (
        "whitespace_and_multiline",
        [
            {"role": "system", "content": "  sos conciso  "},
            {"role": "user", "content": "  linea 1\nlinea 2  "},
        ],
        True,
    ),
    # a history ending in an assistant turn: no generation prompt
    (
        "trailing_assistant",
        [{"role": "user", "content": "hola"}, {"role": "assistant", "content": "buenas"}],
        False,
    ),
    # an empty system message: the template falls back to instructions-only when thinking is on
    (
        "empty_system_thinking",
        [{"role": "system", "content": "   "}, {"role": "user", "content": "hola"}],
        True,
    ),
]


def raise_exception(message):
    raise RuntimeError(message)


def main():
    template_path = os.path.join(FIXTURES, "chat_template.jinja")
    # Default (lenient) Undefined on purpose, exactly like `transformers`' own template compiler:
    # the template probes `message.tool_calls` / `message.reasoning_content` on plain dicts that
    # don't carry them, and those must read as falsy rather than raise.
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = raise_exception
    env.filters["tojson"] = lambda v, **kw: json.dumps(v, ensure_ascii=False)
    template = env.from_string(open(template_path).read())

    out = []
    for name, messages, thinking in CASES:
        add_generation_prompt = messages[-1]["role"] != "assistant"
        rendered = template.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=thinking,
            # declared so StrictUndefined doesn't trip on the template's own `is defined` guards
            tools=None,
            add_vision_id=False,
        )
        out.append(
            {
                "name": name,
                "messages": messages,
                "enable_thinking": thinking,
                "add_generation_prompt": add_generation_prompt,
                "expected": rendered,
            }
        )

    out_path = os.path.join(FIXTURES, "chat_cases.json")
    json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=1)
    print(f"wrote {len(out)} cases -> {out_path}")


if __name__ == "__main__":
    main()
