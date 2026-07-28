//! Port of `convert_fp8_to_int4.py`'s `download_retry`/`_download_single`/`_download_multi`-
//! shaped worker logic: a resumable HTTP downloader for pulling FP8 shards from a Hugging Face
//! repo one at a time (the disk-safe conversion workflow's whole point — never more than one
//! multi-GB source shard resident alongside the growing int4 output).
//!
//! Ported in spirit, not line-for-line: the Python source hand-rolls `urllib.request` +
//! `threading.Thread` with its own retry/backoff loop; this uses `ureq` (a blocking HTTP client
//! matching this crate's synchronous style — no async runtime anywhere else in rabbit) and
//! `std::thread` for the multi-stream case. Same externally-observable behavior: Range-header
//! resume, a `.part` file that grows in place, a `.part.seg` JSON sidecar recording each
//! stream's own progress so a killed-and-restarted download picks up mid-segment instead of
//! from zero, and indefinite retry with capped backoff (this is a background bulk-download
//! tool with no caller-facing deadline, matching the source's own "keep trying" policy).
//!
//! NOT tested against a real Hugging Face repo in this session (would need real network access
//! and multi-GB transfers, impractical to run automatically) — the resume/retry LOGIC is
//! covered by unit tests against a local `TestServer` (a bare HTTP/1.1 Range-aware responder,
//! this module's own test-only code), which is enough to verify segment math, state
//! persistence, and actual resume-after-interruption behavior without needing the internet.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

const USER_AGENT: &str = "rabbit-convert";

#[derive(Debug)]
pub enum DownloadError {
    Io(io::Error),
    Http(String),
    /// The server didn't honor a `Range` request (returned `200` instead of `206`) — multi-
    /// stream segmented download is impossible against this server; the caller should fall back
    /// to `download_single`, exactly like the Python source's own `stopfail` path does.
    RangeNotHonored,
}

impl fmt::Display for DownloadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DownloadError::Io(e) => write!(f, "download: {e}"),
            DownloadError::Http(e) => write!(f, "download: {e}"),
            DownloadError::RangeNotHonored => write!(f, "download: server did not honor a Range request"),
        }
    }
}

impl std::error::Error for DownloadError {}

impl From<io::Error> for DownloadError {
    fn from(e: io::Error) -> Self {
        DownloadError::Io(e)
    }
}

fn agent(timeout: Duration) -> ureq::Agent {
    ureq::Agent::config_builder().timeout_recv_response(Some(timeout)).timeout_connect(Some(timeout)).build().into()
}

fn part_path(out: &Path) -> PathBuf {
    let mut s = out.as_os_str().to_owned();
    s.push(".part");
    PathBuf::from(s)
}

fn seg_path(part: &Path) -> PathBuf {
    let mut s = part.as_os_str().to_owned();
    s.push(".seg");
    PathBuf::from(s)
}

/// Segment boundaries for `ns` concurrent streams over `[0, expected)` — port of
/// `[(expected*t//NS, expected*(t+1)//NS) for t in range(NS)]`. The last segment absorbs any
/// remainder from integer division (its end is always `expected`, matching Python's own
/// floor-division split).
pub fn segment_bounds(expected: u64, ns: usize) -> Vec<(u64, u64)> {
    (0..ns as u64).map(|t| (expected * t / ns as u64, expected * (t + 1) / ns as u64)).collect()
}

/// The `.seg` sidecar's on-disk shape — port of `{"n": NS, "size": expected, "done": [...]}`.
#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
struct SegState {
    n: usize,
    size: u64,
    done: Vec<u64>,
}

/// Reads a resumable `.seg` sidecar if it matches the CURRENT run's stream count and expected
/// size — a mismatch (different `--streams` between runs, or a different expected file size)
/// means the sidecar is stale and this returns `None`, exactly like the Python source's
/// `if st.get("n")==NS and st.get("size")==expected: done=st["done"]` guard (silently starting
/// fresh instead of trusting incompatible resume state).
fn read_seg_state(path: &Path, ns: usize, expected: u64) -> Option<Vec<u64>> {
    let text = fs::read_to_string(path).ok()?;
    let st: SegState = serde_json::from_str(&text).ok()?;
    if st.n == ns && st.size == expected { Some(st.done) } else { None }
}

/// Atomically overwrites the `.seg` sidecar (write to a temp file, then rename) — matches the
/// Python source's own `tmpside = side + ".tmp"; ...; os.replace(tmpside, side)` pattern, so a
/// crash mid-write never corrupts the resume state a later run would trust.
fn write_seg_state(path: &Path, ns: usize, expected: u64, done: &[u64]) -> io::Result<()> {
    let tmp = {
        let mut s = path.as_os_str().to_owned();
        s.push(".tmp");
        PathBuf::from(s)
    };
    let st = SegState { n: ns, size: expected, done: done.to_vec() };
    fs::write(&tmp, serde_json::to_vec(&st)?)?;
    fs::rename(&tmp, path)
}

/// Single-stream download with `Range`-header resume — port of `_download_single`. `part` grows
/// in place; on success it's renamed to `out`. Retries transient failures (network errors, a
/// clean-but-short EOF) with capped backoff, matching the Python source's "keep trying, this is
/// a background bulk transfer" policy — the ONE thing that stops the loop is `expected` being
/// reached or, when `expected` is `None`, a single pass completing (matches the source's own
/// "lunghezza ignota: passata singola" fallback for an unknown content length).
pub fn download_single(ag: &ureq::Agent, url: &str, out: &Path, mut expected: Option<u64>) -> Result<PathBuf, DownloadError> {
    let part = part_path(out);
    let mut have = fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    if let Some(exp) = expected
        && have >= exp
    {
        fs::rename(&part, out)?;
        return Ok(out.to_path_buf());
    }

    let mut backoff_n: u32 = 0;
    loop {
        let mut req = ag.get(url).header("User-Agent", USER_AGENT);
        if have > 0 {
            req = req.header("Range", format!("bytes={have}-"));
        }
        let resp = match req.call() {
            Ok(r) => r,
            Err(e) => {
                backoff_n += 1;
                std::thread::sleep(Duration::from_secs((1 + backoff_n).min(15) as u64));
                if backoff_n > 200 {
                    return Err(DownloadError::Http(e.to_string()));
                }
                continue;
            }
        };
        let status = resp.status().as_u16();
        if have > 0 && status == 200 {
            have = 0; // server ignored the Range request -- restart clean, matching the source.
        }
        if expected.is_none()
            && let Some(cl) = resp.headers().get("content-length")
            && let Ok(s) = cl.to_str()
            && let Ok(n) = s.parse::<u64>()
        {
            expected = Some(have + n);
        }

        let mut body = resp.into_body();
        let mut reader = body.as_reader();
        let f = OpenOptions::new().create(true).write(true).truncate(false).open(&part)?;
        f.set_len(have)?; // truncate any stale tail from a previous failed attempt past `have`
        let mut buf = [0u8; 1 << 20];
        loop {
            let n = std::io::Read::read(&mut reader, &mut buf)?;
            if n == 0 {
                break;
            }
            f.write_at(&buf[..n], have)?;
            have += n as u64;
        }

        match expected {
            None => break, // unknown length: one pass is the whole transfer, matches the source.
            Some(exp) if have >= exp => break,
            Some(_) => {
                backoff_n += 1;
                std::thread::sleep(Duration::from_secs((1 + backoff_n).min(15) as u64));
            }
        }
    }

    fs::rename(&part, out)?;
    Ok(out.to_path_buf())
}

/// Multi-stream segmented download with per-segment `Range` resume — port of the multi-stream
/// half of `download_retry`. Opens `ns` concurrent threads, each pulling its own `[s0,s1)` byte
/// range via positioned writes (`pwrite`, no cross-thread contention since ranges are disjoint),
/// checkpointing combined progress to the `.seg` sidecar every few seconds. Falls back to
/// `download_single` if the server doesn't honor `Range` (`DownloadError::RangeNotHonored`) —
/// the caller decides whether to catch that and retry single-stream, matching the Python
/// source's own `stopfail` -> `_download_single` fallback at the call site, not buried in here.
pub fn download_multi(ag: &ureq::Agent, url: &str, out: &Path, expected: u64, ns: usize) -> Result<PathBuf, DownloadError> {
    let part = part_path(out);
    let side = seg_path(&part);
    let mut done = read_seg_state(&side, ns, expected).unwrap_or_else(|| vec![0u64; ns]);
    let segs = segment_bounds(expected, ns);

    if !part.exists() {
        let f = File::create(&part)?;
        f.set_len(expected)?;
    }

    let stopfail = Arc::new(AtomicBool::new(false));
    let done_atomics: Vec<Arc<AtomicU64>> = done.iter().map(|&d| Arc::new(AtomicU64::new(d))).collect();
    let part = Arc::new(part.clone());
    let url = Arc::new(url.to_string());
    let ag = ag.clone();

    std::thread::scope(|scope| {
        for (t, &(s0, s1)) in segs.iter().enumerate() {
            let done_t = done_atomics[t].clone();
            let stopfail = stopfail.clone();
            let part = part.clone();
            let url = url.clone();
            let ag = ag.clone();
            scope.spawn(move || {
                let mut backoff_n: u32 = 0;
                while done_t.load(Ordering::Relaxed) < s1 - s0 && !stopfail.load(Ordering::Relaxed) {
                    let pos = s0 + done_t.load(Ordering::Relaxed);
                    let resp = ag.get(url.as_str()).header("User-Agent", USER_AGENT).header("Range", format!("bytes={pos}-{}", s1 - 1)).call();
                    let resp = match resp {
                        Ok(r) => r,
                        Err(_) => {
                            backoff_n += 1;
                            std::thread::sleep(Duration::from_secs((1 + backoff_n / ns.max(1) as u32).min(15) as u64));
                            continue;
                        }
                    };
                    if resp.status().as_u16() != 206 {
                        stopfail.store(true, Ordering::Relaxed);
                        return;
                    }
                    let mut body = resp.into_body();
                    let mut reader = body.as_reader();
                    let f = match OpenOptions::new().write(true).open(part.as_path()) {
                        Ok(f) => f,
                        Err(_) => return,
                    };
                    let mut buf = [0u8; 1 << 20];
                    loop {
                        if done_t.load(Ordering::Relaxed) >= s1 - s0 {
                            break;
                        }
                        let n = match std::io::Read::read(&mut reader, &mut buf) {
                            Ok(0) => break,
                            Ok(n) => n,
                            Err(_) => break,
                        };
                        let cur = done_t.load(Ordering::Relaxed);
                        let rem = (s1 - s0) - cur;
                        let take = (n as u64).min(rem) as usize;
                        if f.write_at(&buf[..take], s0 + cur).is_err() {
                            break;
                        }
                        done_t.fetch_add(take as u64, Ordering::Relaxed);
                    }
                }
            });
        }

        // Checkpoint the sidecar periodically while workers run.
        loop {
            std::thread::sleep(Duration::from_millis(200));
            done = done_atomics.iter().map(|d| d.load(Ordering::Relaxed)).collect();
            let _ = write_seg_state(&side, ns, expected, &done);
            let all_done = done.iter().zip(&segs).all(|(&d, &(s0, s1))| d >= s1 - s0);
            if all_done || stopfail.load(Ordering::Relaxed) {
                break;
            }
        }
    });

    if stopfail.load(Ordering::Relaxed) {
        return Err(DownloadError::RangeNotHonored);
    }

    let total: u64 = done.iter().sum();
    if total != expected {
        return Err(DownloadError::Http(format!("multi-stream download incomplete: {total} of {expected} bytes")));
    }
    fs::remove_file(&side).ok();
    fs::rename(part.as_path(), out)?;
    Ok(out.to_path_buf())
}

/// The top-level entry — port of `download_retry`: resolves the single-vs-multi-stream decision
/// (small file, `ns<=1`, or a legacy `.part` with no `.seg` sidecar all force single-stream,
/// matching the Python source's own heuristic) and falls back to single-stream if multi-stream
/// reports the server doesn't honor `Range`.
pub fn download_retry(ag: &ureq::Agent, url: &str, out: &Path, expected: Option<u64>, streams: usize) -> Result<PathBuf, DownloadError> {
    if fs::metadata(out).map(|m| expected.is_none_or(|e| m.len() == e)).unwrap_or(false) {
        return Ok(out.to_path_buf());
    }
    let ns = streams.clamp(1, 8);
    let part = part_path(out);
    let side = seg_path(&part);
    let legacy = part.exists() && !side.exists();

    match expected {
        Some(exp) if exp >= (256 << 20) && ns > 1 && !legacy => match download_multi(ag, url, out, exp, ns) {
            Err(DownloadError::RangeNotHonored) => {
                fs::remove_file(&part).ok();
                fs::remove_file(&side).ok();
                download_single(ag, url, out, expected)
            }
            other => other,
        },
        _ => download_single(ag, url, out, expected),
    }
}

/// Builds an `ureq::Agent` with rabbit-convert's standard short read timeout (8s — a hung
/// transfer fails fast instead of hanging forever; a live one delivers chunks continuously, so
/// 8s never trips on real progress, matching the Python source's own
/// `HF_HUB_DOWNLOAD_TIMEOUT=8` reasoning).
pub fn default_agent() -> ureq::Agent {
    agent(Duration::from_secs(8))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Write};
    use std::net::{TcpListener, TcpStream};

    #[test]
    fn segment_bounds_covers_the_whole_range_with_no_gaps_or_overlaps() {
        let segs = segment_bounds(100, 3);
        assert_eq!(segs, vec![(0, 33), (33, 66), (66, 100)]);
        for w in segs.windows(2) {
            assert_eq!(w[0].1, w[1].0, "no gap/overlap between consecutive segments");
        }
        assert_eq!(segs.last().unwrap().1, 100, "last segment ends exactly at `expected`");
    }

    #[test]
    fn seg_state_round_trips_and_rejects_a_mismatched_run() {
        let dir = std::env::temp_dir().join("rabbit_test_convert_download_segstate");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("x.part.seg");

        write_seg_state(&path, 4, 1000, &[10, 20, 30, 40]).unwrap();
        assert_eq!(read_seg_state(&path, 4, 1000), Some(vec![10, 20, 30, 40]));
        assert_eq!(read_seg_state(&path, 3, 1000), None, "stream-count mismatch must be rejected, not silently reused");
        assert_eq!(read_seg_state(&path, 4, 999), None, "size mismatch must be rejected, not silently reused");

        fs::remove_dir_all(&dir).ok();
    }

    /// A minimal HTTP/1.1 responder serving one fixed byte blob, Range-aware — just enough to
    /// exercise `download_single`'s real resume behavior without any internet access. Not a
    /// general-purpose test server: single connection at a time, ignores everything but `Range`.
    struct TestServer {
        addr: std::net::SocketAddr,
        handle: Option<std::thread::JoinHandle<()>>,
        stop: Arc<AtomicBool>,
    }

    impl TestServer {
        fn start(body: Vec<u8>) -> TestServer {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            listener.set_nonblocking(true).unwrap();
            let addr = listener.local_addr().unwrap();
            let stop = Arc::new(AtomicBool::new(false));
            let stop2 = stop.clone();
            let handle = std::thread::spawn(move || {
                while !stop2.load(Ordering::Relaxed) {
                    match listener.accept() {
                        Ok((stream, _)) => Self::handle_conn(stream, &body),
                        Err(_) => std::thread::sleep(Duration::from_millis(5)),
                    }
                }
            });
            TestServer { addr, handle: Some(handle), stop }
        }

        fn handle_conn(stream: TcpStream, body: &[u8]) {
            stream.set_nodelay(true).ok();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut request_line = String::new();
            if reader.read_line(&mut request_line).unwrap_or(0) == 0 {
                return;
            }
            let mut range: Option<(usize, usize)> = None;
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                    break;
                }
                if let Some(v) = line.strip_prefix("Range: bytes=").or_else(|| line.strip_prefix("range: bytes=")) {
                    let v = v.trim();
                    if let Some((a, b)) = v.split_once('-') {
                        let start: usize = a.parse().unwrap_or(0);
                        let end = if b.is_empty() { body.len() - 1 } else { b.trim().parse().unwrap_or(body.len() - 1) };
                        range = Some((start, end.min(body.len() - 1)));
                    }
                }
            }
            let (status, slice, content_range) = match range {
                Some((s, e)) if s <= e && e < body.len() => (
                    "206 Partial Content",
                    &body[s..=e],
                    Some(format!("Content-Range: bytes {s}-{e}/{}\r\n", body.len())),
                ),
                _ => ("200 OK", body, None),
            };
            let mut resp = format!("HTTP/1.1 {status}\r\nContent-Length: {}\r\n", slice.len());
            if let Some(cr) = content_range {
                resp.push_str(&cr);
            }
            resp.push_str("Connection: close\r\n\r\n");
            let mut out = reader.into_inner();
            out.write_all(resp.as_bytes()).ok();
            out.write_all(slice).ok();
        }
    }

    impl Drop for TestServer {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Relaxed);
            // unblock the accept() poll loop by connecting once
            std::net::TcpStream::connect(self.addr).ok();
            if let Some(h) = self.handle.take() {
                h.join().ok();
            }
        }
    }

    #[test]
    fn download_single_resumes_a_partial_file_via_range() {
        let full: Vec<u8> = (0..5000u32).map(|i| (i % 251) as u8).collect();
        let server = TestServer::start(full.clone());
        let url = format!("http://{}/blob", server.addr);

        let dir = std::env::temp_dir().join("rabbit_test_convert_download_resume");
        fs::create_dir_all(&dir).unwrap();
        let out = dir.join("blob.bin");
        fs::remove_file(&out).ok();
        let part = part_path(&out);
        fs::write(&part, &full[..2000]).unwrap(); // simulate a prior interrupted download

        let ag = default_agent();
        let got = download_single(&ag, &url, &out, Some(full.len() as u64)).unwrap();
        assert_eq!(got, out);
        assert_eq!(fs::read(&out).unwrap(), full, "resumed download must reconstruct the exact original bytes");

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn download_single_fetches_a_fresh_file_with_no_prior_part() {
        let full: Vec<u8> = (0..777u32).map(|i| (i % 97) as u8).collect();
        let server = TestServer::start(full.clone());
        let url = format!("http://{}/blob", server.addr);

        let dir = std::env::temp_dir().join("rabbit_test_convert_download_fresh");
        fs::create_dir_all(&dir).unwrap();
        let out = dir.join("blob.bin");
        fs::remove_file(&out).ok();

        let ag = default_agent();
        download_single(&ag, &url, &out, Some(full.len() as u64)).unwrap();
        assert_eq!(fs::read(&out).unwrap(), full);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn download_multi_reassembles_every_segment_in_order() {
        let full: Vec<u8> = (0..20_000u32).map(|i| (i % 241) as u8).collect();
        let server = TestServer::start(full.clone());
        let url = format!("http://{}/blob", server.addr);

        let dir = std::env::temp_dir().join("rabbit_test_convert_download_multi");
        fs::create_dir_all(&dir).unwrap();
        let out = dir.join("blob.bin");
        fs::remove_file(&out).ok();

        let ag = default_agent();
        let got = download_multi(&ag, &url, &out, full.len() as u64, 4).unwrap();
        assert_eq!(got, out);
        assert_eq!(fs::read(&out).unwrap(), full, "every segment must land at its correct offset, byte-exact");
        assert!(!part_path(&out).exists(), "the .part file must be renamed away on success");
        assert!(!seg_path(&part_path(&out)).exists(), "the .seg sidecar must be cleaned up on success");

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn download_multi_resumes_from_a_partial_seg_sidecar() {
        let full: Vec<u8> = (0..8000u32).map(|i| (i % 199) as u8).collect();
        let server = TestServer::start(full.clone());
        let url = format!("http://{}/blob", server.addr);

        let dir = std::env::temp_dir().join("rabbit_test_convert_download_multi_resume");
        fs::create_dir_all(&dir).unwrap();
        let out = dir.join("blob.bin");
        fs::remove_file(&out).ok();
        let part = part_path(&out);
        let side = seg_path(&part);

        // Pre-seed a .part file already fully written (as if 3 of 4 segments had finished) and
        // a matching .seg sidecar recording that -- download_multi must trust it and only fetch
        // the remaining bytes of the last segment, not re-fetch everything.
        let ns = 4;
        let segs = segment_bounds(full.len() as u64, ns);
        {
            let f = File::create(&part).unwrap();
            f.set_len(full.len() as u64).unwrap();
            for (t, &(s0, s1)) in segs.iter().enumerate() {
                let done_len = if t == ns - 1 { (s1 - s0) / 2 } else { s1 - s0 };
                f.write_at(&full[s0 as usize..(s0 + done_len) as usize], s0).unwrap();
            }
        }
        let done: Vec<u64> = segs.iter().enumerate().map(|(t, &(s0, s1))| if t == ns - 1 { (s1 - s0) / 2 } else { s1 - s0 }).collect();
        write_seg_state(&side, ns, full.len() as u64, &done).unwrap();

        let ag = default_agent();
        download_multi(&ag, &url, &out, full.len() as u64, ns).unwrap();
        assert_eq!(fs::read(&out).unwrap(), full);

        fs::remove_dir_all(&dir).ok();
    }
}
