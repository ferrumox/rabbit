//! Integration test: `src/qwen38/tokenizer.rs` + `src/tokenizer.rs` against the REAL Qwen 3.8
//! tokenizer (248,044-entry vocab, 247,587 merges, 33 added tokens), with ground truth from the
//! Python `tokenizers` library — an independent oracle, not rabbit's own port. The sibling of
//! `tests/tokenizer_fixture.rs` (GLM-5.2's), and the check that actually proves the two real
//! differences from GLM's file (`ignore_merges: false`, a different `pre_tokenizer` regex) are
//! handled: with either one wrong, most of these cases produce different ids while still decoding
//! back to the same text, so nothing but an oracle comparison would catch it.
//!
//! Needs `tests/fixtures/qwen38/tokenizer.json` (copy it from the checkpoint dir) and
//! `tests/fixtures/qwen38/tokenizer_cases.json` (`tools/gen_qwen38_tokenizer_cases.py`, run in
//! Docker — see its docstring). Skips, not fails, when either is absent: same policy as every other
//! fixture-backed test here.

use std::path::Path;

/// The one case whose ids are EXPECTED to differ: the file declares an `NFC` normalizer, which this
/// port deliberately doesn't implement (see `qwen38::tokenizer`'s module doc), so decomposed input
/// is tokenized as-is instead of being composed first. It must still round-trip byte-for-byte,
/// which is asserted separately below — the risk of skipping NFC is a worse split, never corruption.
fn is_known_nfc_difference(text: &str) -> bool {
    text.chars().any(|c| matches!(c, '\u{0300}'..='\u{036F}'))
}

#[test]
fn encode_and_decode_match_the_real_qwen38_tokenizer() {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen38");
    let tok_path = fixtures.join("tokenizer.json");
    let cases_path = fixtures.join("tokenizer_cases.json");
    if !tok_path.is_file() || !cases_path.is_file() {
        eprintln!(
            "SKIP encode_and_decode_match_the_real_qwen38_tokenizer: fixtures not found in {} — \
             `mkdir -p tests/fixtures/qwen38 && cp <checkpoint>/tokenizer.json tests/fixtures/qwen38/` \
             then run tools/gen_qwen38_tokenizer_cases.py (see its docstring for the Docker one-liner).",
            fixtures.display()
        );
        return;
    }

    // Through the architecture's own loader, so the pre_tokenizer/ignore_merges verification runs
    // against the real file too, not just against synthetic unit-test fixtures.
    let tok = rabbit::qwen38::tokenizer::load(&fixtures).expect("real Qwen 3.8 tokenizer.json must load");
    assert!(!tok.ignore_merges(), "the real file sets ignore_merges: false");

    let cases: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&cases_path).unwrap()).unwrap();
    let cases = cases.as_array().expect("tokenizer_cases.json: expected a JSON array");
    assert!(!cases.is_empty(), "tokenizer_cases.json: no cases found");

    let mut encode_mismatches = Vec::new();
    let mut decode_mismatches = Vec::new();
    let mut known_nfc = 0usize;

    for case in cases {
        let text = case["text"].as_str().expect("case.text");
        let expected_ids: Vec<i32> =
            case["ids"].as_array().expect("case.ids").iter().map(|v| v.as_i64().unwrap() as i32).collect();

        let got_ids = tok.encode(text);
        if got_ids != expected_ids {
            if is_known_nfc_difference(text) {
                known_nfc += 1;
            } else {
                encode_mismatches.push((text.to_string(), expected_ids.clone(), got_ids.clone()));
            }
        }

        // Round-trip is required of EVERY case, including the NFC one.
        let decoded = tok.decode(&got_ids);
        if decoded != text.as_bytes() {
            decode_mismatches.push((text.to_string(), String::from_utf8_lossy(&decoded).to_string()));
        }
    }

    assert!(
        encode_mismatches.is_empty(),
        "{} encode mismatches vs the `tokenizers` oracle (text, expected, got): {:#?}",
        encode_mismatches.len(),
        encode_mismatches
    );
    assert!(
        decode_mismatches.is_empty(),
        "{} decode round-trip mismatches (text, got): {:#?}",
        decode_mismatches.len(),
        decode_mismatches
    );
    eprintln!(
        "qwen38 tokenizer fixture: {}/{} cases matched the oracle exactly ({} known NFC difference(s), all round-tripping)",
        cases.len() - known_nfc,
        cases.len(),
        known_nfc
    );
}

/// The ChatML ids `qwen38::chat_template` will depend on, spot-checked against the real file's
/// `added_tokens` — and against `config.json`/`generation_config.json`, where `<|im_end|>` (248046)
/// is the stop token that only `generation_config.json` lists (see `qwen38::config`'s module doc).
#[test]
fn special_tokens_resolve_to_the_real_qwen38_ids() {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen38");
    if !fixtures.join("tokenizer.json").is_file() {
        eprintln!("SKIP special_tokens_resolve_to_the_real_qwen38_ids: tests/fixtures/qwen38/tokenizer.json not found.");
        return;
    }
    let tok = rabbit::qwen38::tokenizer::load(&fixtures).unwrap();

    assert_eq!(tok.id_of("<|endoftext|>"), Some(248044));
    assert_eq!(tok.id_of("<|im_start|>"), Some(248045));
    assert_eq!(tok.id_of("<|im_end|>"), Some(248046));

    let ids = tok.encode("<|im_start|>user\nhi<|im_end|>");
    assert_eq!(ids.first(), Some(&248045));
    assert_eq!(ids.last(), Some(&248046));
    assert_eq!(tok.decode(&ids), "<|im_start|>user\nhi<|im_end|>".as_bytes());
}
