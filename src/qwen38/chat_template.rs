//! Qwen 3.8's chat template, ported from the REAL `chat_template.jinja` that ships in the
//! checkpoint (read off disk 2026-08-13 — unlike GLM-5.2, whose template had to be
//! reverse-engineered from colibrì's C source, and unlike Kimi K3, which ships no template file at
//! all). Plain ChatML: `<|im_start|>{role}\n{content}<|im_end|>\n`, with a `<think>` block on
//! assistant turns.
//!
//! Restricted to the same text-only subset `crate::chat::render_turn`/`render_messages` implement
//! for every other family here: no tool declarations (`tools`), no `tool_calls`/`tool` messages, no
//! image/video content parts. Those branches exist in the real template and are deliberately out of
//! scope — rabbit's HTTP surface has no way to express them.
//!
//! Three behaviors are worth stating outright, because each one silently changes output if missed:
//!
//! **1. The reasoning-effort system message.** The template injects an extra system message
//! whenever thinking is on, with text depending on `reasoning_effort` (default `'xhigh'`):
//! `xhigh` and `low` each have their own sentence, and **`medium` deliberately has none** (the
//! Jinja leaves `reasoning_instructions` empty for it). rabbit has no CLI surface for effort, so
//! `render_turn`/`render_messages` use the template's own default, `xhigh` — same choice
//! `kimi_k3::chat_template` makes with `thinking_effort=max`. If a system prompt is also present,
//! the instruction goes FIRST, separated by a blank line, inside the same system message.
//!
//! **2. `think` flips the generation prompt, not just the preamble.** With thinking on the prompt
//! ends `<|im_start|>assistant\n<think>\n` and the model reasons; with it off the template
//! pre-closes the block — `<|im_start|>assistant\n<think>\n\n</think>\n\n` — so the model answers
//! directly. This is the same "wrong tag and the model never stops" hazard `chat.rs`'s GLM template
//! doc warns about, just with a different shape.
//!
//! **3. A continuation turn must close the model's own assistant turn.** `generate_reply` breaks on
//! a stop id WITHOUT forwarding it, so after a reply the KV cache holds `...<|im_start|>assistant\n
//! <think>\n{reply}` with no `<|im_end|>`. ChatML needs that closer before the next
//! `<|im_start|>user`, so `render_turn` emits `<|im_end|>\n` first on every non-first turn. GLM's
//! and Kimi Linear's templates need no equivalent (their role tags self-delimit), which is exactly
//! why this is easy to miss.
//!
//! Prior assistant turns in `render_messages` render as `<think>\n\n</think>\n\n{content}`: the
//! template's `preserve_thinking` defaults to true, and an OpenAI-style client sends no
//! `reasoning_content`, so the real template emits an EMPTY think block there — reproduced here
//! rather than "cleaned up", since the whole point is to match what the model was trained on.

use crate::chat::Role;

const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";

/// `reasoning_effort`'s three accepted values, verbatim from the template's own validation list
/// (`('xhigh', 'medium', 'low')`, anything else raises). Only reachable through this module's
/// `*_with_effort` functions today — see this module's doc for why the public entry points hardcode
/// the template's own default.
///
/// `Medium`/`Low` are only constructed by this module's own tests today — kept (rather than
/// dropped down to a single hardcoded string) because they pin real, verified template behavior,
/// including the easily-missed fact that `medium` emits NO instructions at all. Wiring a
/// `--reasoning-effort` flag through `chat.rs` would then be a one-line change here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
enum Effort {
    XHigh,
    Medium,
    Low,
}

/// The template's default when thinking is enabled: `reasoning_effort|default('xhigh')`.
const DEFAULT_EFFORT: Effort = Effort::XHigh;

/// `reasoning_instructions`, verbatim per effort. `Medium` yields `None`: the Jinja sets the
/// variable only for `xhigh` and `low`, so a medium-effort conversation carries NO extra system
/// message at all.
fn reasoning_instructions(effort: Effort) -> Option<&'static str> {
    match effort {
        Effort::XHigh => Some(
            "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.",
        ),
        Effort::Medium => None,
        Effort::Low => Some(
            "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.",
        ),
    }
}

/// One complete `<|im_start|>{role}\n{content}<|im_end|>\n` block. `content` is trimmed, matching
/// the template's `render_content(...)|trim` on every message.
fn message(role: &str, content: &str) -> String {
    format!("{IM_START}{role}\n{}{IM_END}\n", content.trim())
}

/// The leading system message, or `""` when there's neither a system prompt nor reasoning
/// instructions to carry. Mirrors the template's no-tools branch exactly, including the case where
/// a system message exists but trims to empty (then only the instructions are emitted).
fn system_block(system: Option<&str>, instructions: Option<&str>) -> String {
    let content = system.map(str::trim).unwrap_or("");
    match (content.is_empty(), instructions) {
        (false, Some(instr)) => message("system", &format!("{instr}\n\n{content}")),
        (false, None) => message("system", content),
        (true, Some(instr)) => message("system", instr),
        (true, None) => String::new(),
    }
}

/// A completed assistant turn with an empty think block — see this module's doc for why the block is
/// present and empty rather than absent.
fn assistant_turn(content: &str) -> String {
    format!("{IM_START}assistant\n<think>\n\n</think>\n\n{}{IM_END}\n", content.trim())
}

/// `add_generation_prompt`: the open assistant turn the model continues from. With `think`, the
/// `<think>` block is left OPEN for it to reason in; without, it's pre-closed so the model answers
/// immediately.
fn generation_prompt(think: bool) -> String {
    let opener = if think { "<think>\n" } else { "<think>\n\n</think>\n\n" };
    format!("{IM_START}assistant\n{opener}")
}

pub fn render_turn(user_msg: &str, first: bool, think: bool, system: Option<&str>) -> String {
    render_turn_with_effort(user_msg, first, think, system, DEFAULT_EFFORT)
}

fn render_turn_with_effort(user_msg: &str, first: bool, think: bool, system: Option<&str>, effort: Effort) -> String {
    let mut out = String::new();
    if first {
        out.push_str(&system_block(system, if think { reasoning_instructions(effort) } else { None }));
    } else {
        // Close the assistant turn the model itself left open — see this module's doc, point 3.
        out.push_str(IM_END);
        out.push('\n');
    }
    out.push_str(&message("user", user_msg));
    out.push_str(&generation_prompt(think));
    out
}

pub fn render_messages(messages: &[(Role, String)], think: bool) -> String {
    render_messages_with_effort(messages, think, DEFAULT_EFFORT)
}

fn render_messages_with_effort(messages: &[(Role, String)], think: bool, effort: Effort) -> String {
    let instructions = if think { reasoning_instructions(effort) } else { None };
    // The template only ever emits a system message from `messages[0]`, and raises if a later one
    // appears; here a later system message is rendered in place instead of panicking — this API
    // returns a String, and a mis-ordered history is the caller's problem, not a reason to abort a
    // request.
    let leading_system = match messages.first() {
        Some((Role::System, content)) => Some(content.as_str()),
        _ => None,
    };
    let mut out = system_block(leading_system, instructions);

    for (i, (role, content)) in messages.iter().enumerate() {
        match role {
            Role::System if i == 0 => {} // already rendered by `system_block`
            Role::System => out.push_str(&message("system", content)),
            Role::User => out.push_str(&message("user", content)),
            Role::Assistant => out.push_str(&assistant_turn(content)),
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

    const XHIGH: &str = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";

    #[test]
    fn first_turn_with_thinking_emits_the_xhigh_system_message_then_user_then_an_open_think() {
        let out = render_turn("hola", true, true, None);
        assert_eq!(
            out,
            format!("<|im_start|>system\n{XHIGH}<|im_end|>\n<|im_start|>user\nhola<|im_end|>\n<|im_start|>assistant\n<think>\n")
        );
    }

    /// A system prompt and the reasoning instructions share ONE system message, instructions first,
    /// separated by a blank line — not two messages, and not the other order.
    #[test]
    fn system_prompt_and_reasoning_instructions_share_one_message_instructions_first() {
        let out = render_turn("hola", true, true, Some("  sos conciso  "));
        assert_eq!(
            out,
            format!(
                "<|im_start|>system\n{XHIGH}\n\nsos conciso<|im_end|>\n<|im_start|>user\nhola<|im_end|>\n<|im_start|>assistant\n<think>\n"
            )
        );
    }

    /// Thinking off: no reasoning instructions at all (so with no system prompt there's no system
    /// message either), and the think block arrives PRE-CLOSED.
    #[test]
    fn thinking_off_drops_the_preamble_and_pre_closes_the_think_block() {
        let out = render_turn("hola", true, false, None);
        assert_eq!(out, "<|im_start|>user\nhola<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");

        // ...but an explicit system prompt still renders, without instructions prepended.
        let with_sys = render_turn("hola", true, false, Some("sos conciso"));
        assert_eq!(
            with_sys,
            "<|im_start|>system\nsos conciso<|im_end|>\n<|im_start|>user\nhola<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    /// The subtle one: a continuation turn must first close the assistant turn the model left open
    /// (the stop token never reaches the KV cache), and must not repeat the system message.
    #[test]
    fn continuation_turn_closes_the_open_assistant_turn_and_repeats_no_system_message() {
        let out = render_turn("y ahora?", false, true, Some("sos conciso"));
        assert_eq!(out, "<|im_end|>\n<|im_start|>user\ny ahora?<|im_end|>\n<|im_start|>assistant\n<think>\n");
        assert!(!out.contains("system"), "the system message belongs to the first turn only");
        assert!(out.starts_with("<|im_end|>\n"), "without this closer the next user turn nests inside the assistant's");
    }

    /// `medium` is the odd one out: the template defines no instruction text for it, so a
    /// medium-effort conversation with no system prompt emits no system message at all.
    #[test]
    fn medium_effort_emits_no_reasoning_instructions() {
        assert_eq!(reasoning_instructions(Effort::Medium), None);
        let out = render_turn_with_effort("hola", true, true, None, Effort::Medium);
        assert_eq!(out, "<|im_start|>user\nhola<|im_end|>\n<|im_start|>assistant\n<think>\n");

        let low = render_turn_with_effort("hola", true, true, None, Effort::Low);
        assert!(low.starts_with("<|im_start|>system\nReasoning effort is set to low."), "got {low}");
    }

    #[test]
    fn render_messages_renders_a_full_conversation_with_empty_think_blocks_on_past_turns() {
        let messages = vec![
            (Role::System, "sos conciso".to_string()),
            (Role::User, "hola".to_string()),
            (Role::Assistant, "buenas".to_string()),
            (Role::User, "y ahora?".to_string()),
        ];
        assert_eq!(
            render_messages(&messages, true),
            format!(
                "<|im_start|>system\n{XHIGH}\n\nsos conciso<|im_end|>\n\
                 <|im_start|>user\nhola<|im_end|>\n\
                 <|im_start|>assistant\n<think>\n\n</think>\n\nbuenas<|im_end|>\n\
                 <|im_start|>user\ny ahora?<|im_end|>\n\
                 <|im_start|>assistant\n<think>\n"
            )
        );
    }

    /// A history already ending in an assistant message gets no generation prompt appended (the
    /// caller is continuing that turn), same rule every other family's `render_messages` follows.
    #[test]
    fn render_messages_appends_no_generation_prompt_after_a_trailing_assistant_turn() {
        let messages = vec![(Role::User, "hola".to_string()), (Role::Assistant, "buenas".to_string())];
        let out = render_messages(&messages, false);
        assert!(out.ends_with("buenas<|im_end|>\n"), "got {out}");
        assert_eq!(out.matches("<|im_start|>assistant").count(), 1);
    }

    /// Every rendered block is closed: as many `<|im_end|>`s as `<|im_start|>`s, except for the one
    /// open assistant turn the generation prompt deliberately leaves for the model.
    #[test]
    fn blocks_are_balanced_apart_from_the_open_generation_prompt() {
        let messages = vec![
            (Role::System, "s".to_string()),
            (Role::User, "u1".to_string()),
            (Role::Assistant, "a1".to_string()),
            (Role::User, "u2".to_string()),
        ];
        for think in [true, false] {
            let out = render_messages(&messages, think);
            assert_eq!(
                out.matches(IM_START).count(),
                out.matches(IM_END).count() + 1,
                "exactly one unclosed block expected (think={think}): {out}"
            );
        }
    }
}
