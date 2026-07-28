//! Integration test: `src/kimi_linear/tokenizer.rs` against the REAL Kimi Linear tokenizer
//! (163,584-entry base vocab + 258 special/reserved ids) — the sibling of `tokenizer_fixture.rs`'s
//! GLM-5.2 version.
//!
//! Needs `tests/fixtures/kimi/{tiktoken.model,tokenizer_config.json}` (`tools/
//! fetch_kimi_tokenizer_fixture.py`) and `tests/fixtures/kimi/tokenizer_cases.json` (`tools/
//! gen_kimi_tokenizer_cases.py`, ground truth from the real `tiktoken` Python library against
//! the real `tokenization_kimi.py` pre-tokenizer pattern — not rabbit's own port, an independent
//! oracle). Skips (not fails) when either is absent, same policy as `tokenizer_fixture.rs`.

use rabbit::kimi_linear::tokenizer::Tokenizer;
use std::path::Path;

#[test]
fn encode_and_decode_match_the_real_tokenizer_on_diverse_text() {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/kimi");
    let model_path = fixtures.join("tiktoken.model");
    let cases_path = fixtures.join("tokenizer_cases.json");
    if !model_path.is_file() || !cases_path.is_file() {
        eprintln!(
            "SKIP encode_and_decode_match_the_real_tokenizer_on_diverse_text: fixtures not found \
             in {} — run `python3 tools/fetch_kimi_tokenizer_fixture.py && \
             python3 tools/gen_kimi_tokenizer_cases.py` first (needs huggingface_hub + tiktoken).",
            fixtures.display()
        );
        return;
    }

    let tok = Tokenizer::load(&fixtures).expect("failed to load tests/fixtures/kimi");
    let cases: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&cases_path).unwrap()).unwrap();

    let mut encode_mismatches = Vec::new();
    let mut decode_mismatches = Vec::new();
    let mut total = 0usize;

    for group in ["ordinary_cases", "special_cases"] {
        let cases = cases[group].as_array().unwrap_or_else(|| panic!("tokenizer_cases.json: missing {group}"));
        for case in cases {
            total += 1;
            let text = case["text"].as_str().expect("case.text");
            let expected_ids: Vec<i32> = case["ids"].as_array().expect("case.ids").iter().map(|v| v.as_i64().unwrap() as i32).collect();

            let got_ids = tok.encode(text);
            if got_ids != expected_ids {
                encode_mismatches.push((text.to_string(), expected_ids.clone(), got_ids.clone()));
            }

            let decoded = tok.decode(&got_ids);
            if decoded != text.as_bytes() {
                decode_mismatches.push((text.to_string(), String::from_utf8_lossy(&decoded).to_string()));
            }
        }
    }

    assert!(encode_mismatches.is_empty(), "{} encode mismatches (text, expected, got): {:#?}", encode_mismatches.len(), encode_mismatches);
    assert!(decode_mismatches.is_empty(), "{} decode round-trip mismatches (text, got): {:#?}", decode_mismatches.len(), decode_mismatches);
    eprintln!("kimi tokenizer fixture: {total}/{total} cases matched (encode + decode round-trip)");
}

#[test]
fn special_tokens_resolve_to_the_documented_kimi_ids() {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/kimi");
    if !fixtures.join("tiktoken.model").is_file() {
        eprintln!("SKIP special_tokens_resolve_to_the_documented_kimi_ids: tests/fixtures/kimi/tiktoken.model not found.");
        return;
    }
    let tok = Tokenizer::load(&fixtures).unwrap();

    // spot-checked against the real checkpoint's tokenizer_config.json's added_tokens_decoder.
    assert_eq!(tok.id_of("[BOS]"), Some(163584));
    assert_eq!(tok.id_of("[EOS]"), Some(163585));
    assert_eq!(tok.id_of("<|im_end|>"), Some(163586));
    assert_eq!(tok.id_of("<|im_user|>"), Some(163587));
    assert_eq!(tok.id_of("<|im_assistant|>"), Some(163588));
    // an unnamed reserved id must still resolve, via the placeholder-name fallback.
    assert!(tok.id_of("<|reserved_token_163700|>").is_some());

    let ids = tok.encode("<|im_user|>hi<|im_assistant|>");
    assert_eq!(ids.first(), Some(&163587));
    assert_eq!(ids.last(), Some(&163588));
}
