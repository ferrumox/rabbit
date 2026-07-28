//! Kimi K3's real chat template — a genuinely different format from Kimi Linear 48B's simple
//! `<|im_role|>role<|im_middle|>content<|im_end|>` ChatML-style template. K3 has no
//! `chat_template.jinja` file at all; the real reference (`tokenization_kimi.py`'s
//! `TikTokenTokenizer.apply_chat_template` + `encoding_k3.py`'s `build_chat_segments`, both
//! fetched from `moonshotai/Kimi-K3` 2026-07-27, not guessed) builds the prompt PROGRAMMATICALLY
//! in Python as a custom "XTML" structure: `<|open|>tag attr="val"<|sep|>...<|close|>tag<|sep|>`,
//! with messages wrapped in `<|open|>message role="..."<|sep|>...<|close|>message<|sep|>
//! <|end_of_msg|>` and assistant turns carrying a structural `<think>...</think><response>...
//! </response>` channel pair (both tags always present in thinking mode, even with empty
//! reasoning — confirmed via `_render_assistant_segments`'s comment: "the <think> channel is
//! structural").
//!
//! Restricted to the same simple text-only subset `crate::chat::render_turn`/`render_messages`
//! already implement for GLM-5.2/Kimi Linear 48B: no tool-calling (`tools`/`tool_calls`
//! branches), no image content, no `response_format`/`tool_choice` system-message injections —
//! none of rabbit's other chat templates support those either, so this isn't a narrower cut
//! relative to what's already there. The control tokens (`<|open|>`/`<|close|>`/`<|sep|>`/
//! `<|end_of_msg|>`) are encoded via `kimi_linear::tokenizer::Tokenizer::encode`'s existing
//! "any literal special-token string is always recognized" behavior — this module only needs to
//! produce the right STRING, matching the real `apply_chat_template(tokenize=False)` path
//! (`"".join(segment.text for segment in segments)`), not track which segments are
//! `allow_special` itself.
//!
//! **The `thinking_effort` system preamble** (`_internal_system_message("thinking-effort", ...)`)
//! is emitted by the real code UNCONDITIONALLY at the start of every `build_chat_segments` call
//! when `thinking=True` and a `thinking_effort` is set — `apply_chat_template` itself defaults
//! `thinking_effort` to `"max"` (`kwargs.setdefault("thinking_effort", "max")`) unless the caller
//! overrides it. Since rabbit has no such override surface, this always fires when `think` is
//! true, hardcoded to `"max"` — matching the real default's actual observable behavior. Emitted
//! once at conversation start (`render_turn`'s `first`, `render_messages` unconditionally),
//! BEFORE any message content, same relative order as the real code (tool-declare, then this
//! preamble, then the message loop).
//!
//! **No `[BOS]`-equivalent prefix** — same as Kimi Linear 48B (confirmed: `apply_chat_template`
//! never calls `build_inputs_with_special_tokens`, goes straight to `_encode_chat_segments`).
//! **The real stop token is `<|end_of_msg|>`** (id `163586`, confirmed against the real
//! checkpoint's `config.json`'s `eos_token_id` — matches; `kimi_k3::config::Cfg` already reads
//! this generically via `cfg.base.stop_ids`, no change needed there).

use crate::chat::Role;

const OPEN: &str = "<|open|>";
const CLOSE: &str = "<|close|>";
const SEP: &str = "<|sep|>";
const END_OF_MSG: &str = "<|end_of_msg|>";

fn escape_attr(value: &str) -> String {
    value.replace('&', "&amp;").replace('"', "&quot;")
}

fn open_tag(tag: &str, attrs: &[(&str, &str)]) -> String {
    let mut out = String::new();
    out.push_str(OPEN);
    out.push_str(tag);
    for (k, v) in attrs {
        out.push(' ');
        out.push_str(k);
        out.push_str("=\"");
        out.push_str(&escape_attr(v));
        out.push('"');
    }
    out.push_str(SEP);
    out
}

fn close_tag(tag: &str) -> String {
    format!("{CLOSE}{tag}{SEP}")
}

/// A plain user/system message: `<|open|>message role="..."<|sep|>{content}<|close|>message
/// <|sep|><|end_of_msg|>` — matches `build_chat_segments`'s `role == "user"`/`role == "system"`
/// branches with no `name` attribute (rabbit's `(Role, String)` messages carry no name).
fn render_user_or_system(role: &str, content: &str) -> String {
    let mut out = String::new();
    out.push_str(&open_tag("message", &[("role", role)]));
    out.push_str(content);
    out.push_str(&close_tag("message"));
    out.push_str(END_OF_MSG);
    out
}

/// A complete assistant turn: the structural `<think>` channel (empty body — rabbit stores no
/// separate reasoning content) when `think`, then `<response>{content}</response>`, matching
/// `_render_assistant_segments` with `tool_calls` absent.
fn render_assistant(content: &str, think: bool) -> String {
    let mut out = String::new();
    out.push_str(&open_tag("message", &[("role", "assistant")]));
    if think {
        out.push_str(&open_tag("think", &[]));
        out.push_str(&close_tag("think"));
    }
    out.push_str(&open_tag("response", &[]));
    out.push_str(content);
    out.push_str(&close_tag("response"));
    out.push_str(&close_tag("message"));
    out.push_str(END_OF_MSG);
    out
}

/// `_internal_system_message("thinking-effort", ...)`, hardcoded to `thinking_effort=max` — see
/// this module's doc for why that's the real default, not a simplification.
fn thinking_effort_preamble() -> String {
    let body = "`thinking_effort` guides on how much to think in your thinking channel (not \
                including the response channel), supported values include `low`, `medium`, \
                `high`, and `max`.\nNow the system is invoked with `thinking_effort=max`.";
    let mut out = String::new();
    out.push_str(&open_tag("message", &[("role", "system"), ("type", "thinking-effort")]));
    out.push_str(body);
    out.push_str(&close_tag("message"));
    out.push_str(END_OF_MSG);
    out
}

/// The open, UN-closed assistant turn appended when the caller wants the model to continue
/// (`add_generation_prompt`): `<|open|>message role="assistant"<|sep|><|open|>think<|sep|>` (or
/// `...response<|sep|>` when `!think`) — the model's own output is expected to close both tags
/// and the message itself, ending in `<|end_of_msg|>`.
fn generation_prompt(think: bool) -> String {
    let mut out = String::new();
    out.push_str(&open_tag("message", &[("role", "assistant")]));
    out.push_str(&open_tag(if think { "think" } else { "response" }, &[]));
    out
}

pub fn render_turn(user_msg: &str, first: bool, think: bool, system: Option<&str>) -> String {
    let mut out = String::new();
    if first {
        if think {
            out.push_str(&thinking_effort_preamble());
        }
        if let Some(sys) = system {
            out.push_str(&render_user_or_system("system", sys));
        }
    }
    out.push_str(&render_user_or_system("user", user_msg));
    out.push_str(&generation_prompt(think));
    out
}

pub fn render_messages(messages: &[(Role, String)], think: bool) -> String {
    let mut out = String::new();
    if think {
        out.push_str(&thinking_effort_preamble());
    }
    for (role, content) in messages {
        match role {
            Role::System => out.push_str(&render_user_or_system("system", content)),
            Role::User => out.push_str(&render_user_or_system("user", content)),
            Role::Assistant => out.push_str(&render_assistant(content, think)),
        }
    }
    if !matches!(messages.last(), Some((Role::Assistant, _))) {
        out.push_str(&generation_prompt(think));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_turn_with_thinking_and_system_prompt_emits_preamble_then_system_then_user_then_open_think() {
        let out = render_turn("hi", true, true, Some("be nice"));
        assert_eq!(
            out,
            "<|open|>message role=\"system\" type=\"thinking-effort\"<|sep|>`thinking_effort` guides on how much to think in your thinking channel (not including the response channel), supported values include `low`, `medium`, `high`, and `max`.\nNow the system is invoked with `thinking_effort=max`.<|close|>message<|sep|><|end_of_msg|>\
             <|open|>message role=\"system\"<|sep|>be nice<|close|>message<|sep|><|end_of_msg|>\
             <|open|>message role=\"user\"<|sep|>hi<|close|>message<|sep|><|end_of_msg|>\
             <|open|>message role=\"assistant\"<|sep|><|open|>think<|sep|>"
        );
    }

    #[test]
    fn first_turn_without_thinking_skips_the_preamble_and_opens_response_directly() {
        let out = render_turn("hi", true, false, None);
        assert_eq!(out, "<|open|>message role=\"user\"<|sep|>hi<|close|>message<|sep|><|end_of_msg|><|open|>message role=\"assistant\"<|sep|><|open|>response<|sep|>");
        assert!(!out.contains("think"));
    }

    #[test]
    fn later_turn_never_repeats_the_system_prompt_or_the_preamble() {
        let out = render_turn("again", false, true, Some("be nice"));
        assert!(!out.contains("system"), "system prompt/preamble must only render on the first turn");
        assert_eq!(out, "<|open|>message role=\"user\"<|sep|>again<|close|>message<|sep|><|end_of_msg|><|open|>message role=\"assistant\"<|sep|><|open|>think<|sep|>");
    }

    #[test]
    fn render_messages_closes_every_past_turn_with_an_empty_think_channel_and_leaves_the_last_assistant_slot_open() {
        let messages = vec![(Role::User, "hi".to_string()), (Role::Assistant, "hello".to_string()), (Role::User, "how are you".to_string())];
        let out = render_messages(&messages, true);
        assert_eq!(
            out,
            "<|open|>message role=\"system\" type=\"thinking-effort\"<|sep|>`thinking_effort` guides on how much to think in your thinking channel (not including the response channel), supported values include `low`, `medium`, `high`, and `max`.\nNow the system is invoked with `thinking_effort=max`.<|close|>message<|sep|><|end_of_msg|>\
             <|open|>message role=\"user\"<|sep|>hi<|close|>message<|sep|><|end_of_msg|>\
             <|open|>message role=\"assistant\"<|sep|><|open|>think<|sep|><|close|>think<|sep|><|open|>response<|sep|>hello<|close|>response<|sep|><|close|>message<|sep|><|end_of_msg|>\
             <|open|>message role=\"user\"<|sep|>how are you<|close|>message<|sep|><|end_of_msg|>\
             <|open|>message role=\"assistant\"<|sep|><|open|>think<|sep|>"
        );
    }

    #[test]
    fn render_messages_does_not_reopen_an_already_trailing_assistant_turn() {
        let messages = vec![(Role::User, "hi".to_string()), (Role::Assistant, "hello".to_string())];
        let out = render_messages(&messages, false);
        assert!(out.ends_with("<|end_of_msg|>"), "must not append a second open assistant tag when the last turn is already assistant");
    }

    #[test]
    fn escape_attr_matches_the_real_attribute_escaping_rule() {
        // `_escape_attr_value` (encoding_k3.py): only `&` and `"` get escaped, in that order --
        // no public render_turn/render_messages call site currently threads dynamic content
        // through an XTML attribute (rabbit's (Role, String) messages carry no `name`), but
        // `open_tag`'s attrs mechanism follows the real spec regardless, so this tests the
        // escaping rule directly rather than through an unreachable public path.
        assert_eq!(escape_attr("a & b"), "a &amp; b");
        assert_eq!(escape_attr("say \"hi\""), "say &quot;hi&quot;");
        assert_eq!(escape_attr("a & \"b\""), "a &amp; &quot;b&quot;");
    }

    #[test]
    fn message_body_content_is_not_escaped() {
        // Only ATTRIBUTE values go through escape_attr -- message content is plain text, same
        // as the real `_append_text`/`_text` path (no escaping call anywhere in it).
        let out = render_turn("hi", true, false, Some("a & b \"quoted\""));
        assert!(out.contains("a & b \"quoted\""));
    }
}
