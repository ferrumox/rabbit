//! OpenAI-compatible HTTP server (`rabbit --serve`) — see `rabbit-plan.md`'s Fase 11 plan for
//! the design rationale behind every choice below (single-threaded blocking accept loop, no
//! server-side conversation state, `tiny_http` for a minimal sync stack with no async
//! runtime).
//!
//! Deliberately out of scope for v1 (see the plan): legacy `POST /v1/completions`, an explicit
//! admission queue (the OS TCP backlog already serves that role given the single-threaded,
//! always-serialized handling below), and `--kv-slots` multi-session (there is no per-request
//! session identity to attach a slot to — every request is stateless, see `handle_chat_completions`).
//!
//! Every handler here takes `Request` BY VALUE and is responsible for calling exactly one of
//! `request.respond(...)` (normal JSON) or `request.into_writer()` (SSE streaming) itself —
//! `tiny_http::Request` only supports responding by consuming itself, so passing it around by
//! `&mut` and trying to "respond later" doesn't type-check (and isn't worked around with any
//! unsafe trick here; every function below either owns `request` or borrows it briefly to read
//! before handing ownership on).

use crate::chat::{self, GenEvent, KvState, Role, Session};
use serde::Deserialize;
use serde_json::{Value, json};
use std::io::{Read, Write};
use tiny_http::{Header, Method, Request, Response};

pub struct ServeConfig {
    pub host: String,
    pub port: u16,
    pub api_key: Option<String>,
    pub model_id: String,
    /// Server-wide default for GLM-5.2's reasoning block — matches the CLI's own `--think`
    /// flag. Not exposed per-request in v1 (OpenAI's Chat Completions shape has no field to
    /// map it from); every request served by one process gets the same setting.
    pub think: bool,
    /// Ceiling for a per-request `max_tokens` override, taken from the CLI's own `--max-tokens`
    /// (the same value `session.max_tokens` is initialized from). A request asking for more
    /// than this gets clamped down to it rather than rejected — matches colibrì's own
    /// `da260c3` fix (real client SDKs send large default `max_tokens` regardless of what the
    /// server can actually do; rejecting broke trivial requests).
    pub max_tokens_cap: usize,
}

/// Constant-time byte comparison — an API key check using `==`/`memcmp` short-circuits on the
/// first mismatched byte, letting a network-observable timing difference leak how many leading
/// bytes of a guess were correct. Length is compared normally first (leaking length isn't
/// meaningful for a fixed-format secret) before the constant-time body.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// `Ok(Some(n))` — set `session.max_tokens` to `n` (already clamped to `cap`); `Ok(None)` —
/// leave the session's current value untouched (no override requested); `Err` — the request
/// asked for something nonsensical (zero tokens).
fn clamp_max_tokens(requested: Option<usize>, cap: usize) -> Result<Option<usize>, ApiError> {
    match requested {
        Some(0) => Err(ApiError::bad_request("max_tokens must be a positive integer")),
        Some(n) => Ok(Some(n.min(cap))),
        None => Ok(None),
    }
}

/// Chat messages are text, not file uploads — 1 MiB is generous headroom for even a long
/// multi-turn history while keeping a hostile/broken client from parking the single request
/// loop on an enormous body.
const MAX_BODY_BYTES: usize = 1 << 20;

#[derive(Debug)]
struct ApiError {
    status: u16,
    kind: &'static str,
    message: String,
}

impl ApiError {
    fn new(status: u16, kind: &'static str, message: impl Into<String>) -> ApiError {
        ApiError { status, kind, message: message.into() }
    }
    fn bad_request(message: impl Into<String>) -> ApiError {
        ApiError::new(400, "invalid_request_error", message)
    }
    fn body(&self) -> String {
        json!({"error": {"message": self.message, "type": self.kind}}).to_string()
    }
}

impl From<std::io::Error> for ApiError {
    fn from(e: std::io::Error) -> Self {
        ApiError::new(500, "server_error", e.to_string())
    }
}

impl From<Box<dyn std::error::Error>> for ApiError {
    fn from(e: Box<dyn std::error::Error>) -> Self {
        ApiError::new(500, "server_error", e.to_string())
    }
}

pub fn serve(mut session: Session, cfg: ServeConfig) -> Result<(), Box<dyn std::error::Error>> {
    let addr = format!("{}:{}", cfg.host, cfg.port);
    let server = tiny_http::Server::http(&addr).map_err(|e| format!("bind {addr}: {e}"))?;
    eprintln!("rabbit serve — listening on http://{addr}");
    if cfg.api_key.is_some() {
        eprintln!("  API key required (Authorization: Bearer <key>)");
    }

    // Single-threaded, always-serialized: only one forward pass can make CPU progress at a
    // time regardless of how many connections are accepted (one Model/ExpertCaches resident,
    // tens of GB) — accepting requests concurrently would buy nothing but complexity here. The
    // kernel's own TCP accept backlog is the admission queue; see the Fase 11 plan.
    for request in server.incoming_requests() {
        handle(request, &mut session, &cfg);
    }
    Ok(())
}

fn respond_json(request: Request, status: u16, body: String) {
    let header = Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).expect("static header is valid");
    let response = Response::from_string(body).with_status_code(status).with_header(header);
    if let Err(e) = request.respond(response) {
        eprintln!("failed to send response: {e}");
    }
}

fn respond_err(request: Request, e: ApiError) {
    let (status, body) = (e.status, e.body());
    respond_json(request, status, body);
}

fn handle(request: Request, session: &mut Session, cfg: &ServeConfig) {
    if let Some(key) = &cfg.api_key {
        let want = format!("Bearer {key}");
        let authorized = request
            .headers()
            .iter()
            .any(|h| h.field.equiv("Authorization") && constant_time_eq(h.value.as_str().as_bytes(), want.as_bytes()));
        if !authorized {
            respond_err(request, ApiError::new(401, "invalid_request_error", "Invalid API key"));
            return;
        }
    }

    let method = request.method().clone();
    let url = request.url().to_string();
    match (method, url.as_str()) {
        (Method::Get, "/v1/models") => handle_models(request, cfg),
        (Method::Post, "/v1/chat/completions") => handle_chat_completions(request, session, cfg),
        _ => respond_err(request, ApiError::new(404, "invalid_request_error", "not found")),
    }
}

fn read_body(request: &mut Request) -> Result<Vec<u8>, ApiError> {
    if let Some(len) = request.body_length()
        && len > MAX_BODY_BYTES
    {
        return Err(ApiError::new(413, "invalid_request_error", "request body too large"));
    }
    let mut buf = Vec::new();
    request.as_reader().take((MAX_BODY_BYTES + 1) as u64).read_to_end(&mut buf)?;
    if buf.len() > MAX_BODY_BYTES {
        return Err(ApiError::new(413, "invalid_request_error", "request body too large"));
    }
    Ok(buf)
}

fn handle_models(request: Request, cfg: &ServeConfig) {
    let body = json!({"object": "list", "data": [{"id": cfg.model_id, "object": "model", "created": 0, "owned_by": "rabbit"}]});
    respond_json(request, 200, body.to_string());
}

#[derive(Deserialize)]
struct ChatMessageJson {
    role: String,
    content: String,
}

#[derive(Deserialize, Default)]
struct ChatRequest {
    #[serde(default)]
    messages: Vec<ChatMessageJson>,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    #[serde(default)]
    stream: bool,
}

fn parse_messages(msgs: &[ChatMessageJson]) -> Result<Vec<(Role, String)>, ApiError> {
    if msgs.is_empty() {
        return Err(ApiError::bad_request("messages must not be empty"));
    }
    msgs.iter()
        .map(|m| {
            let role = match m.role.as_str() {
                "system" => Role::System,
                "user" => Role::User,
                "assistant" => Role::Assistant,
                other => return Err(ApiError::bad_request(format!("unknown role: {other}"))),
            };
            Ok((role, m.content.clone()))
        })
        .collect()
}

fn handle_chat_completions(mut request: Request, session: &mut Session, cfg: &ServeConfig) {
    let raw = match read_body(&mut request) {
        Ok(b) => b,
        Err(e) => return respond_err(request, e),
    };
    let req: ChatRequest = match serde_json::from_slice(&raw) {
        Ok(r) => r,
        Err(e) => return respond_err(request, ApiError::bad_request(format!("invalid JSON: {e}"))),
    };
    let messages = match parse_messages(&req.messages) {
        Ok(m) => m,
        Err(e) => return respond_err(request, e),
    };

    // Per-request overrides fall back to the session's own CLI-provided defaults. Safe to
    // mutate `session` directly (not a hidden/lingering state hazard): the server is
    // single-threaded and every request sets ALL of these explicitly before use, so there's no
    // stale-value carryover between requests.
    session.sampling.temperature = req.temperature.unwrap_or(session.sampling.temperature);
    session.sampling.nucleus = req.top_p.unwrap_or(session.sampling.nucleus);
    match clamp_max_tokens(req.max_tokens, cfg.max_tokens_cap) {
        Ok(Some(n)) => session.max_tokens = n,
        Ok(None) => {}
        Err(e) => return respond_err(request, e),
    }

    let prompt_text = chat::render_messages(&messages, cfg.think);
    let prompt_ids: Vec<usize> = session.tokenizer.encode(&prompt_text).into_iter().map(|id| id as usize).collect();
    let mut kv = KvState::new(&session.model);
    let id = format!("chatcmpl-{:016x}", request_id());
    let created = unix_time();

    if req.stream {
        stream_chat_completion(request, session, &mut kv, &prompt_ids, &id, created, cfg);
        return;
    }

    let mut hit_stop = true;
    let result = chat::generate_reply(session, &mut kv, &prompt_ids, 0, |ev| {
        if let GenEvent::Token { index, max, .. } = ev
            && index >= max
        {
            hit_stop = false;
        }
    });
    let (text, _pos, n) = match result {
        Ok(v) => v,
        Err(e) => return respond_err(request, ApiError::from(e)),
    };

    // Not a violation of "stateless across requests" (Fase 11): the usage cache isn't
    // conversation memory visible to API clients, it's an internal performance-tuning file —
    // same distinction colibrì's own `run_serve` makes when it accumulates `.coli_usage`.
    if session.usage_cache_enabled
        && let Err(e) = session.caches.save_usage(&session.model_dir)
    {
        eprintln!("usage cache: failed to save: {e}");
    }

    let finish_reason = if hit_stop { "stop" } else { "length" };
    let body = json!({
        "id": id, "object": "chat.completion", "created": created, "model": cfg.model_id,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": prompt_ids.len(), "completion_tokens": n, "total_tokens": prompt_ids.len() + n},
    });
    respond_json(request, 200, body.to_string());
}

/// Streams one SSE `chat.completion.chunk` per generated token. `tiny_http`'s normal
/// `Request::respond` only flushes once, at the end of the whole body (see its `raw_print`
/// implementation) — no good for real token-by-token delivery. `Request::into_writer` hands
/// back the raw `BufWriter<TcpStream>` instead, so the status line, headers, and each chunk are
/// written and flushed by hand.
fn stream_chat_completion(request: Request, session: &mut Session, kv: &mut KvState, prompt_ids: &[usize], id: &str, created: u64, cfg: &ServeConfig) {
    let mut writer = request.into_writer();

    let head_result = (|| -> std::io::Result<()> {
        writer.write_all(b"HTTP/1.1 200 OK\r\n")?;
        writer.write_all(b"Content-Type: text/event-stream\r\n")?;
        writer.write_all(b"Cache-Control: no-cache\r\n")?;
        writer.write_all(b"Connection: close\r\n")?;
        writer.write_all(b"Transfer-Encoding: chunked\r\n\r\n")
    })();
    if let Err(e) = head_result {
        eprintln!("stream: failed to write headers: {e}");
        return;
    }

    fn write_chunk(w: &mut dyn Write, payload: &str) -> std::io::Result<()> {
        write!(w, "{:x}\r\n", payload.len())?;
        w.write_all(payload.as_bytes())?;
        w.write_all(b"\r\n")?;
        w.flush()
    }

    let role_chunk = json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": cfg.model_id,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": Value::Null}]});
    if write_chunk(&mut *writer, &format!("data: {role_chunk}\n\n")).is_err() {
        return;
    }

    // Decoded token bytes can land mid-UTF-8-character (BPE tokens don't align to codepoint
    // boundaries) — buffer until the accumulated bytes are valid UTF-8 before emitting a
    // delta, or the SSE stream would contain invalid JSON string escapes on any multi-byte
    // output (routine for non-ASCII replies).
    let mut byte_buf: Vec<u8> = Vec::new();
    let mut hit_stop = true;
    let mut io_failed = false;

    let gen_result = chat::generate_reply(session, kv, prompt_ids, 0, |ev| {
        if io_failed {
            return;
        }
        if let GenEvent::Token { bytes, index, max, .. } = ev {
            if index >= max {
                hit_stop = false;
            }
            byte_buf.extend_from_slice(bytes);
            if let Ok(s) = std::str::from_utf8(&byte_buf) {
                if !s.is_empty() && write_chunk(&mut *writer, &format!("data: {}\n\n", json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": cfg.model_id, "choices": [{"index": 0, "delta": {"content": s}, "finish_reason": Value::Null}]}))).is_err()
                {
                    io_failed = true;
                }
                byte_buf.clear();
            }
            // else: incomplete multi-byte sequence so far, keep buffering for the next token.
        }
    });

    // Save regardless of whether the client disconnected mid-stream or generation itself
    // errored below -- the router already ran and the counters already bumped either way. Same
    // "not conversation state" rationale as the non-streaming path above.
    if session.usage_cache_enabled
        && let Err(e) = session.caches.save_usage(&session.model_dir)
    {
        eprintln!("usage cache: failed to save: {e}");
    }

    if io_failed {
        eprintln!("stream: client disconnected mid-response");
        return;
    }
    if let Err(e) = gen_result {
        eprintln!("stream: generation error: {e}");
        let _ = write_chunk(&mut *writer, &format!("data: {}\n\n", json!({"error": {"message": e.to_string(), "type": "server_error"}})));
        return;
    }

    // Any bytes still sitting in `byte_buf` never completed a valid UTF-8 character while
    // tokens were still coming — generation ended (stop token or max_tokens) with a dangling
    // partial sequence. Flush it now (lossy, same fallback the non-streaming path already uses
    // for a whole reply) rather than silently dropping the tail of the response.
    if !byte_buf.is_empty() {
        let s = String::from_utf8_lossy(&byte_buf).into_owned();
        let _ = write_chunk(
            &mut *writer,
            &format!(
                "data: {}\n\n",
                json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": cfg.model_id,
                    "choices": [{"index": 0, "delta": {"content": s}, "finish_reason": Value::Null}]})
            ),
        );
    }

    let finish_reason = if hit_stop { "stop" } else { "length" };
    let final_chunk = json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": cfg.model_id,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]});
    let _ = write_chunk(&mut *writer, &format!("data: {final_chunk}\n\n"));
    let _ = write_chunk(&mut *writer, "[DONE]\n\n"); // best-effort; client may already be gone
    let _ = writer.write_all(b"0\r\n\r\n");
    let _ = writer.flush();
}

fn request_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    unix_time().wrapping_mul(1_000_003).wrapping_add(n)
}

fn unix_time() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_time_eq_matches_only_identical_bytes() {
        assert!(constant_time_eq(b"Bearer sk-abc123", b"Bearer sk-abc123"));
        assert!(!constant_time_eq(b"Bearer sk-abc123", b"Bearer sk-abc124"));
        assert!(!constant_time_eq(b"Bearer sk-abc123", b"Bearer sk-abc12"));
        assert!(!constant_time_eq(b"", b"nonempty"));
        assert!(constant_time_eq(b"", b""));
    }

    #[test]
    fn clamp_max_tokens_rejects_zero() {
        let err = clamp_max_tokens(Some(0), 200).unwrap_err();
        assert_eq!(err.status, 400);
    }

    #[test]
    fn clamp_max_tokens_caps_requests_above_the_server_limit_instead_of_rejecting() {
        assert_eq!(clamp_max_tokens(Some(1_000_000), 200).unwrap(), Some(200));
    }

    #[test]
    fn clamp_max_tokens_passes_through_requests_within_the_limit() {
        assert_eq!(clamp_max_tokens(Some(50), 200).unwrap(), Some(50));
    }

    #[test]
    fn clamp_max_tokens_leaves_the_session_default_untouched_when_omitted() {
        assert_eq!(clamp_max_tokens(None, 200).unwrap(), None);
    }
}
