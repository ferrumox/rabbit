//! Integration test: `src/tokenizer.rs` against the REAL GLM-5.2 tokenizer (154,820-entry
//! vocab, 321,649 merges) — as opposed to `tests/oracle`'s tiny random model, which never
//! exercises tokenization at all (vocab_size=256, no BPE). This is the one part of the engine
//! we can validate against real, non-synthetic data without the full 370GB/FP8 weights.
//!
//! Needs `tests/fixtures/tokenizer.json` (`tools/fetch_tokenizer_fixture.py`) and
//! `tests/fixtures/tokenizer_cases.json` (`tools/gen_tokenizer_cases.py`, ground truth from
//! the real `tokenizers` Python library — not rabbit's own port, an independent oracle).
//! Skips (not fails) when either is absent, same policy as `tests/teacher_forcing.rs`.

use rabbit::tokenizer::Tokenizer;
use std::path::Path;

#[test]
fn encode_and_decode_match_the_real_tokenizer_on_diverse_text() {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let tok_path = fixtures.join("tokenizer.json");
    let cases_path = fixtures.join("tokenizer_cases.json");
    if !tok_path.is_file() || !cases_path.is_file() {
        eprintln!(
            "SKIP encode_and_decode_match_the_real_tokenizer_on_diverse_text: fixtures not found \
             in {} — run `python3 tools/fetch_tokenizer_fixture.py && python3 tools/gen_tokenizer_cases.py` \
             first (needs huggingface_hub + tokenizers).",
            fixtures.display()
        );
        return;
    }

    let tok = Tokenizer::load(&tok_path).expect("failed to load tests/fixtures/tokenizer.json");
    let cases: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&cases_path).unwrap()).unwrap();
    let cases = cases.as_array().expect("tokenizer_cases.json: expected a JSON array");
    assert!(!cases.is_empty(), "tokenizer_cases.json: no cases found");

    let mut encode_mismatches = Vec::new();
    let mut decode_mismatches = Vec::new();

    for case in cases {
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

    assert!(encode_mismatches.is_empty(), "{} encode mismatches (text, expected, got): {:#?}", encode_mismatches.len(), encode_mismatches);
    assert!(decode_mismatches.is_empty(), "{} decode round-trip mismatches (text, got): {:#?}", decode_mismatches.len(), decode_mismatches);
    eprintln!("tokenizer fixture: {}/{} cases matched (encode + decode round-trip)", cases.len(), cases.len());
}

#[test]
fn special_tokens_resolve_to_the_documented_glm52_ids() {
    let tok_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tokenizer.json");
    if !tok_path.is_file() {
        eprintln!("SKIP special_tokens_resolve_to_the_documented_glm52_ids: tests/fixtures/tokenizer.json not found.");
        return;
    }
    let tok = Tokenizer::load(&tok_path).unwrap();

    // spot-checked against tests/fixtures/tokenizer.json's added_tokens list.
    assert_eq!(tok.id_of("<|endoftext|>"), Some(154820));
    assert_eq!(tok.id_of("<|user|>"), Some(154827));
    assert_eq!(tok.id_of("<|assistant|>"), Some(154828));
    assert_eq!(tok.id_of("<|observation|>"), Some(154829));

    let ids = tok.encode("<|user|>hi<|assistant|>");
    assert_eq!(ids.first(), Some(&154827));
    assert_eq!(ids.last(), Some(&154828));
}
