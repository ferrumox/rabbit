//! Integration test: `src/qwen38/chat_template.rs`'s hand port against the REAL
//! `chat_template.jinja` rendered by Jinja2 itself (`tools/gen_qwen38_chat_cases.py`, in the same
//! `ImmutableSandboxedEnvironment` transformers uses), string-for-string.
//!
//! Why an oracle and not just unit tests: a chat template is 170 lines of Jinja whose output differs
//! from a plausible hand translation by single newlines and block ordering — and the failure mode
//! isn't an error, it's a model that reasons in the wrong place or never emits a stop token. The
//! unit tests in the module pin the intended shape; this pins that the shape is the RIGHT one.
//!
//! Needs `tests/fixtures/qwen38/chat_template.jinja` (copy it from the checkpoint) and
//! `tests/fixtures/qwen38/chat_cases.json`. Skips, not fails, when absent.

use rabbit::chat::Role;
use std::path::Path;

fn role_of(name: &str) -> Role {
    match name {
        "system" => Role::System,
        "user" => Role::User,
        "assistant" => Role::Assistant,
        other => panic!("chat_cases.json: unexpected role {other:?} (this port covers no tool roles)"),
    }
}

#[test]
fn render_messages_matches_the_real_jinja_template() {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen38");
    let cases_path = fixtures.join("chat_cases.json");
    if !cases_path.is_file() {
        eprintln!(
            "SKIP render_messages_matches_the_real_jinja_template: {} not found — \
             `cp <checkpoint>/chat_template.jinja tests/fixtures/qwen38/` then run \
             tools/gen_qwen38_chat_cases.py (see its docstring for the Docker one-liner).",
            cases_path.display()
        );
        return;
    }

    let cases: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&cases_path).unwrap()).unwrap();
    let cases = cases.as_array().expect("chat_cases.json: expected a JSON array");
    assert!(!cases.is_empty(), "chat_cases.json: no cases found");

    let mut mismatches = Vec::new();
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let think = case["enable_thinking"].as_bool().unwrap();
        let messages: Vec<(Role, String)> = case["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| (role_of(m["role"].as_str().unwrap()), m["content"].as_str().unwrap().to_string()))
            .collect();
        let expected = case["expected"].as_str().unwrap();

        let got = rabbit::qwen38::chat_template::render_messages(&messages, think);
        if got != expected {
            mismatches.push(format!("case {name}:\n  expected {expected:?}\n  got      {got:?}"));
        }
    }

    assert!(mismatches.is_empty(), "{} case(s) differ from the real Jinja template:\n{}", mismatches.len(), mismatches.join("\n"));
    eprintln!("qwen38 chat template: {}/{} cases match the real Jinja render exactly", cases.len(), cases.len());
}

/// The incremental path (`render_turn`, what `--chat` uses) has no Jinja counterpart — Jinja always
/// renders a whole conversation — so this checks the property that matters instead: turn by turn,
/// the concatenated prompt stays a well-formed ChatML stream, with the model's own reply closed by
/// the NEXT turn's leading `<|im_end|>` and exactly one block left open for it to continue.
///
/// It deliberately does NOT assert equality with `render_messages` on the same history: the two
/// legitimately differ, because the incremental stream keeps the model's real reasoning text inline
/// while a stateless re-render (no `reasoning_content` from the client) emits an empty `<think>`
/// block instead — the real template behaves the same way.
#[test]
fn render_turn_concatenation_stays_well_formed_chatml() {
    let mut stream = String::new();
    stream.push_str(&rabbit::qwen38::chat_template::render_turn("hola", true, true, Some("sos conciso")));
    assert_eq!(stream.matches("<|im_start|>").count(), stream.matches("<|im_end|>").count() + 1);

    // the model's reply, as it actually arrives: reasoning, the closing </think>, then the answer.
    stream.push_str("pienso un poco\n</think>\n\nbuenas");
    stream.push_str(&rabbit::qwen38::chat_template::render_turn("y ahora?", false, true, Some("sos conciso")));

    assert_eq!(
        stream.matches("<|im_start|>").count(),
        stream.matches("<|im_end|>").count() + 1,
        "still exactly one open block after a continuation turn: {stream:?}"
    );
    assert_eq!(stream.matches("<|im_start|>system").count(), 1, "the system message must not repeat");
    assert!(
        stream.contains("buenas<|im_end|>\n<|im_start|>user\ny ahora?"),
        "the continuation turn must close the assistant turn before opening the user's: {stream:?}"
    );
    assert!(stream.ends_with("<|im_start|>assistant\n<think>\n"));
}
