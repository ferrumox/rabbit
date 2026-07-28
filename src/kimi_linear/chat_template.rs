//! Kimi Linear's real chat template (`chat_template.jinja`, fetched from
//! `moonshotai/Kimi-Linear-48B-A3B-Instruct` this session — not guessed), restricted to the
//! same simple text-only subset `crate::chat::render_turn`/`render_messages` already implement
//! for GLM-5.2: no tool-calling, no image content (the real template's `tools`/`tool_calls`/
//! image branches are out of scope — GLM-5.2's own `chat.rs` has no tool-calling support
//! either, so this isn't a narrower cut relative to what rabbit already does).
//!
//! Confirmed from the real template + `tokenizer_config.json`: **no `[BOS]`-equivalent prefix**
//! (no `add_bos_token` setting, and the template itself never references `{{ bos_token }}`) —
//! unlike GLM-5.2's `[gMASK]<sop>`. Every message renders
//! `<|im_{role}|>{role_name}<|im_middle|>{content}<|im_end|>`; an open, UN-closed
//! `<|im_assistant|>assistant<|im_middle|>` is appended when the caller wants the model to
//! continue (matching the template's `add_generation_prompt` branch) — the model's own output
//! ends with `<|im_end|>` (`eos_token_id: 163586` in the real checkpoint's
//! `generation_config.json`, matching this exact token), so nothing else needs to close it.
//!
//! **No `<think>`/`</think>` tag exists in Kimi Linear's special-token vocabulary at all**
//! (confirmed via the real `tokenizer_config.json`'s `added_tokens_decoder` — none of its 17
//! named entries are a think tag, unlike GLM-5.2's own reasoning-toggle convention). The
//! `think` parameter both functions below accept, for a uniform call signature with
//! `crate::chat::render_turn`/`render_messages`, is simply ignored.

use crate::chat::Role;

pub fn render_turn(user_msg: &str, first: bool, _think: bool, system: Option<&str>) -> String {
    let mut out = String::new();
    if first && let Some(sys) = system {
        out.push_str("<|im_system|>system<|im_middle|>");
        out.push_str(sys);
        out.push_str("<|im_end|>");
    }
    out.push_str("<|im_user|>user<|im_middle|>");
    out.push_str(user_msg);
    out.push_str("<|im_end|><|im_assistant|>assistant<|im_middle|>");
    out
}

pub fn render_messages(messages: &[(Role, String)], _think: bool) -> String {
    let mut out = String::new();
    for (role, content) in messages {
        let tag = match role {
            Role::System => "<|im_system|>system<|im_middle|>",
            Role::User => "<|im_user|>user<|im_middle|>",
            Role::Assistant => "<|im_assistant|>assistant<|im_middle|>",
        };
        out.push_str(tag);
        out.push_str(content);
        out.push_str("<|im_end|>");
    }
    if !matches!(messages.last(), Some((Role::Assistant, _))) {
        out.push_str("<|im_assistant|>assistant<|im_middle|>");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_turn_with_system_prompt_opens_system_then_user_then_open_assistant() {
        let out = render_turn("hi", true, false, Some("be nice"));
        assert_eq!(out, "<|im_system|>system<|im_middle|>be nice<|im_end|><|im_user|>user<|im_middle|>hi<|im_end|><|im_assistant|>assistant<|im_middle|>");
    }

    #[test]
    fn later_turn_never_repeats_the_system_prompt() {
        let out = render_turn("again", false, false, Some("be nice"));
        assert!(!out.contains("im_system"), "system prompt must only render on the first turn");
        assert_eq!(out, "<|im_user|>user<|im_middle|>again<|im_end|><|im_assistant|>assistant<|im_middle|>");
    }

    #[test]
    fn think_flag_has_no_effect_no_think_tag_exists_for_this_architecture() {
        let with_think = render_turn("hi", true, true, None);
        let without_think = render_turn("hi", true, false, None);
        assert_eq!(with_think, without_think);
        assert!(!with_think.contains("think"));
    }

    #[test]
    fn render_messages_closes_every_past_turn_and_leaves_the_last_assistant_slot_open() {
        let messages = vec![(Role::User, "hi".to_string()), (Role::Assistant, "hello".to_string()), (Role::User, "how are you".to_string())];
        let out = render_messages(&messages, false);
        assert_eq!(
            out,
            "<|im_user|>user<|im_middle|>hi<|im_end|>\
             <|im_assistant|>assistant<|im_middle|>hello<|im_end|>\
             <|im_user|>user<|im_middle|>how are you<|im_end|>\
             <|im_assistant|>assistant<|im_middle|>"
        );
    }

    #[test]
    fn render_messages_does_not_reopen_an_already_trailing_assistant_turn() {
        let messages = vec![(Role::User, "hi".to_string()), (Role::Assistant, "hello".to_string())];
        let out = render_messages(&messages, false);
        assert!(out.ends_with("<|im_end|>"), "must not append a second open assistant tag when the last turn is already assistant");
    }
}
