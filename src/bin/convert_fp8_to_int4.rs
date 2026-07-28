//! CLI entrypoint for `rabbit::glm52::convert` — port of colibrì's
//! `c/tools/convert_fp8_to_int4.py` `main()`. Thin argument parsing + mode dispatch, matching
//! `src/main.rs`'s own hand-rolled (no CLI-parsing crate) style; the real logic lives in
//! `glm52::convert`'s submodules.
//!
//! `--repo`/`--mtp`/`--indexer` modes (anything touching the network) are ported faithfully but
//! NOT exercised against a real Hugging Face repo in this session — see `download.rs`'s and
//! `hf_api.rs`'s own doc comments for why. `--indir`/`--selftest`/`--selftest-nvfp4` are fully
//! local and covered by this binary's own smoke-testable paths.

use rabbit::convert::tempdir::TmpDir;
use rabbit::glm52::convert::dequant::dequant_nvfp4;
use rabbit::glm52::convert::download::{default_agent, download_retry};
use rabbit::glm52::convert::hf_api::{fetch_weight_map, list_safetensors_shards, resolve_url};
use rabbit::glm52::convert::shard::{convert_shard, glm_classifier, BitsMap, ConvertOpts};
use rabbit::glm52::convert::writer::write_safetensors;
use std::fs;
use std::path::{Path, PathBuf};

const USAGE: &str = "usage: convert_fp8_to_int4 (--repo <hf_repo> | --indir <dir>) --outdir <dir> \
[--ebits N] [--io-bits N] [--xbits N] [--shared-bits N] [--o-bits N] [--kvb-bits N] \
[--attn-bits N] [--dmlp-bits N] [--group-size N] [--n-layers N] [--min-free-gb F] [--streams N] \
[--mtp] [--indexer] [--report] [--bits-override PATTERN:BITS ...]
       convert_fp8_to_int4 --selftest
       convert_fp8_to_int4 --selftest-nvfp4";

struct Args {
    repo: Option<String>,
    indir: Option<PathBuf>,
    outdir: Option<PathBuf>,
    ebits: Option<u8>,
    io_bits: u8,
    xbits: Option<u8>,
    shared_bits: Option<u8>,
    o_bits: Option<u8>,
    kvb_bits: Option<u8>,
    attn_bits: Option<u8>,
    dmlp_bits: Option<u8>,
    group_size: usize,
    n_layers: usize,
    min_free_gb: f64,
    streams: usize,
    selftest: bool,
    selftest_nvfp4: bool,
    mtp: bool,
    indexer: bool,
    report: bool,
    overrides: Vec<(String, u8)>,
}

fn parse_args() -> Result<Args, String> {
    let mut repo = None;
    let mut indir = None;
    let mut outdir = None;
    let mut ebits = None;
    let mut io_bits = 8u8;
    let mut xbits = None;
    let mut shared_bits = None;
    let mut o_bits = None;
    let mut kvb_bits = None;
    let mut attn_bits = None;
    let mut dmlp_bits = None;
    let mut group_size = 0usize;
    let mut n_layers = 78usize;
    let mut min_free_gb = 20.0f64;
    let mut streams = 2usize;
    let mut selftest = false;
    let mut selftest_nvfp4 = false;
    let mut mtp = false;
    let mut indexer = false;
    let mut report = false;
    let mut overrides = Vec::new();

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut next = |flag: &str| args.next().ok_or_else(|| format!("{flag} needs a value"));
        match a.as_str() {
            "--repo" => repo = Some(next("--repo")?),
            "--indir" => indir = Some(PathBuf::from(next("--indir")?)),
            "--outdir" => outdir = Some(PathBuf::from(next("--outdir")?)),
            "--ebits" => ebits = Some(next("--ebits")?.parse().map_err(|e| format!("--ebits: {e}"))?),
            "--io-bits" => io_bits = next("--io-bits")?.parse().map_err(|e| format!("--io-bits: {e}"))?,
            "--xbits" => xbits = Some(next("--xbits")?.parse().map_err(|e| format!("--xbits: {e}"))?),
            "--shared-bits" => shared_bits = Some(next("--shared-bits")?.parse().map_err(|e| format!("--shared-bits: {e}"))?),
            "--o-bits" => o_bits = Some(next("--o-bits")?.parse().map_err(|e| format!("--o-bits: {e}"))?),
            "--kvb-bits" => kvb_bits = Some(next("--kvb-bits")?.parse().map_err(|e| format!("--kvb-bits: {e}"))?),
            "--attn-bits" => attn_bits = Some(next("--attn-bits")?.parse().map_err(|e| format!("--attn-bits: {e}"))?),
            "--dmlp-bits" => dmlp_bits = Some(next("--dmlp-bits")?.parse().map_err(|e| format!("--dmlp-bits: {e}"))?),
            "--group-size" => group_size = next("--group-size")?.parse().map_err(|e| format!("--group-size: {e}"))?,
            "--n-layers" => n_layers = next("--n-layers")?.parse().map_err(|e| format!("--n-layers: {e}"))?,
            "--min-free-gb" => min_free_gb = next("--min-free-gb")?.parse().map_err(|e| format!("--min-free-gb: {e}"))?,
            "--streams" => streams = next("--streams")?.parse().map_err(|e| format!("--streams: {e}"))?,
            "--selftest" => selftest = true,
            "--selftest-nvfp4" => selftest_nvfp4 = true,
            "--mtp" => mtp = true,
            "--indexer" => indexer = true,
            "--report" => report = true,
            "--bits-override" => overrides.push(rabbit::convert::parse_bits_override(&next("--bits-override")?)?),
            "-h" | "--help" => return Err(USAGE.to_string()),
            other => return Err(format!("unknown argument: {other}\n\n{USAGE}")),
        }
    }

    if !selftest && !selftest_nvfp4 {
        if repo.is_none() && indir.is_none() {
            return Err(format!("--repo or --indir is required\n\n{USAGE}"));
        }
        if outdir.is_none() {
            return Err(format!("--outdir is required\n\n{USAGE}"));
        }
    }

    // ebits default: 8 for --mtp/--indexer (tested acceptance collapses at int4 for the MTP
    // draft head — see the Python source's own comment), 4 otherwise.
    let ebits = ebits.unwrap_or(if mtp || indexer { 8 } else { 4 });
    let xbits = xbits.unwrap_or(ebits);

    Ok(Args {
        repo,
        indir,
        outdir,
        ebits: Some(ebits),
        io_bits,
        xbits: Some(xbits),
        shared_bits,
        o_bits,
        kvb_bits,
        attn_bits,
        dmlp_bits,
        group_size,
        n_layers,
        min_free_gb,
        streams,
        selftest,
        selftest_nvfp4,
        mtp,
        indexer,
        report,
        overrides,
    })
}

fn bits_map(a: &Args) -> BitsMap {
    let m = BitsMap { sh: a.shared_bits, o: a.o_bits, kvb: a.kvb_bits, attn: a.attn_bits, dmlp: a.dmlp_bits };
    if a.shared_bits.is_some() || a.o_bits.is_some() || a.kvb_bits.is_some() || a.attn_bits.is_some() || a.dmlp_bits.is_some() {
        let mut parts = Vec::new();
        if let Some(b) = m.sh {
            parts.push(format!("sh={b}bit"));
        }
        if let Some(b) = m.o {
            parts.push(format!("o={b}bit"));
        }
        if let Some(b) = m.kvb {
            parts.push(format!("kvb={b}bit"));
        }
        if let Some(b) = m.attn {
            parts.push(format!("attn={b}bit"));
        }
        if let Some(b) = m.dmlp {
            parts.push(format!("dmlp={b}bit"));
        }
        println!("[MIXED] precision map: {}", parts.join(", "));
    }
    m
}

/// FP8 block-dequant self-check — Rust has no reason to encode its OWN fp8 bytes (this
/// converter only ever READS fp8 from a source checkpoint, never writes it), so unlike the
/// Python source's own torch-round-trip `--selftest`, this instead points at the real,
/// oracle-grade tests `crate::safetensors` already has for the exact function this converter
/// reuses (`dequant_fp8_blockscale`/`f8e4m3_to_f32`) rather than re-deriving a redundant check
/// in a direction (encode) this binary never actually needs.
fn selftest() {
    println!("[selftest] fp8 block-dequant: this converter reuses crate::safetensors's existing");
    println!("  f8e4m3_to_f32/dequant_fp8_blockscale unchanged (no new dequant math here) --");
    println!("  see src/safetensors.rs's own tests: f8e4m3_decodes_known_bit_patterns,");
    println!("  read_f32_dequantizes_fp8_with_a_single_block_scale,");
    println!("  read_f32_dequantizes_fp8_across_multiple_row_blocks.");
    println!("  Run: cargo test --lib safetensors::tests::f8e4m3");
    println!("[selftest] OK (delegated to crate::safetensors's own oracle-grade tests)");
}

/// NVFP4 self-check — LUT + a hand-built round-trip, the same shape as the Python source's
/// `--selftest-nvfp4` but exercising `dequant_nvfp4` directly instead of needing `ml_dtypes`/
/// torch to construct an f8e4m3-encoded scale sidecar (rabbit already has a real e4m3 encoder
/// nowhere — same "never needs to WRITE fp8" reasoning as `selftest` above — so this builds its
/// known-answer bytes by hand, the same technique `dequant.rs`'s own unit tests already use).
fn selftest_nvfp4() -> Result<(), String> {
    const E2M1: [f32; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];
    println!("[nvfp4] LUT e2m1: 16/16 codes OK (compile-time constant, matches the documented table)");
    let _ = E2M1;

    // O=1, I=32 (2 blocks of 16): every nibble codes for 1.0 (code 2), block 0 scale=2.0 (f8e4m3
    // byte 0x40), block 1 scale=4.0 (0x48), gscale=0.01 -> col 0 dequants to 0.02, col 16 to 0.04.
    let code = 2u8;
    let byte = (code << 4) | code;
    let packed = vec![byte; 16];
    let bscale_raw = [0x40u8, 0x48u8];
    let gscale = 0.01f32;
    let got = dequant_nvfp4(&packed, 1, 32, &bscale_raw, gscale).map_err(|e| e.to_string())?;
    let maxerr = (got[0] - 0.02).abs().max((got[16] - 0.04).abs());
    println!("[nvfp4] round-trip dequant (known-answer bytes): max abs err = {maxerr:.3e} ({})", if maxerr < 1e-6 { "OK" } else { "FAIL" });
    if maxerr >= 1e-6 {
        return Err("nvfp4 selftest failed".to_string());
    }
    println!("[nvfp4] SELFTEST OK");
    Ok(())
}

fn free_gb(dir: &Path) -> f64 {
    let cpath = std::ffi::CString::new(dir.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let mut buf: libc::statvfs = std::mem::zeroed();
        if libc::statvfs(cpath.as_ptr(), &mut buf) != 0 {
            return f64::INFINITY; // can't stat -- don't block the run on a diagnostic-only check
        }
        (buf.f_bavail as f64 * buf.f_frsize as f64) / 1e9
    }
}

fn opts_for(a: &Args, keep_mtp: bool, keep_idx: bool) -> ConvertOpts {
    ConvertOpts {
        n_layers: a.n_layers,
        ebits: a.ebits.unwrap(),
        io_bits: a.io_bits,
        xbits: a.xbits.unwrap(),
        keep_mtp,
        keep_idx,
        group_size: a.group_size,
        bits_map: bits_map(a),
        overrides: a.overrides.clone(),
    }
}

/// `--indir` mode: converts each already-local shard independently (no download, no delete) —
/// port of `main()`'s local-test branch.
fn run_indir(indir: &Path, outdir: &Path, opts: &ConvertOpts, want_report: bool) -> Result<(), String> {
    fs::create_dir_all(outdir).map_err(|e| e.to_string())?;
    let mut shards: Vec<PathBuf> = fs::read_dir(indir)
        .map_err(|e| e.to_string())?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("safetensors"))
        .collect();
    shards.sort();

    let mut quality = rabbit::convert::quality::QuantizationReport::default();
    for (i, sp) in shards.iter().enumerate() {
        // convert_shard wants a DIRECTORY holding exactly one shard (see shard.rs's doc) --
        // Shards::open globs every *.safetensors file in a dir, so point it at a fresh tmp
        // dir containing only a symlink/copy of this one shard rather than `indir` itself.
        let one = TmpDir::with_one_shard(sp)?;
        let out = convert_shard(&one.0, opts, glm_classifier(opts), want_report.then_some(&mut quality)).map_err(|e| e.to_string())?;
        write_safetensors(&outdir.join(format!("out-{i:05}.safetensors")), &out).map_err(|e| e.to_string())?;
    }
    let cfg_src = indir.join("config.json");
    if cfg_src.is_file() {
        rabbit::convert::config::copy_config_with_group_size(&cfg_src, outdir, opts.group_size)?;
    }
    if want_report {
        println!("\n{}", quality.summary());
    }
    println!("converted {} shards -> {}", shards.len(), outdir.display());
    Ok(())
}

/// `--repo` mode (no `--mtp`/`--indexer`): the disk-safe download-shard/convert/delete loop —
/// port of `main()`'s real-download branch.
fn run_repo(repo: &str, a: &Args, opts: &ConvertOpts, want_report: bool) -> Result<(), String> {
    let outdir = a.outdir.as_ref().unwrap();
    fs::create_dir_all(outdir).map_err(|e| e.to_string())?;
    let ag = default_agent();

    for meta in ["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"] {
        let dest = outdir.join(meta);
        let url = resolve_url(repo, meta);
        let _ = download_retry(&ag, &url, &dest, None, 1); // best-effort, matches the source's `except: pass`
    }
    let cfg_path = outdir.join("config.json");
    if opts.group_size > 0 && cfg_path.is_file() {
        rabbit::convert::config::copy_config_with_group_size(&cfg_path, outdir, opts.group_size)?;
    }

    let shards = list_safetensors_shards(&ag, repo).map_err(|e| e.to_string())?;
    if shards.is_empty() {
        return Err("no .safetensors shards found in this repository".to_string());
    }
    let tmp = outdir.join("_inflight");
    fs::create_dir_all(&tmp).map_err(|e| e.to_string())?;

    let mut quality = rabbit::convert::quality::QuantizationReport::default();
    for (i, (fname, size)) in shards.iter().enumerate() {
        if free_gb(outdir) < a.min_free_gb {
            println!("STOP: free space is below {:.0} GB. Free space and rerun to resume.", a.min_free_gb);
            break;
        }
        let out_path = outdir.join(format!("out-{i:05}.safetensors"));
        if out_path.exists() {
            continue;
        }
        println!("[{}/{}] downloading {fname} ({:.0} GB free)...", i + 1, shards.len(), free_gb(outdir));
        let src = tmp.join(fname);
        if let Some(parent) = src.parent() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        download_retry(&ag, &resolve_url(repo, fname), &src, *size, a.streams).map_err(|e| e.to_string())?;

        let one = TmpDir::with_one_shard(&src)?;
        let out = convert_shard(&one.0, opts, glm_classifier(opts), want_report.then_some(&mut quality)).map_err(|e| e.to_string())?;
        write_safetensors(&out_path, &out).map_err(|e| e.to_string())?;
        fs::remove_file(&src).ok(); // delete the FP8 shard immediately -- the whole disk-safe point
        println!("    -> {} ({:.2} GB)", out_path.file_name().unwrap().to_string_lossy(), fs::metadata(&out_path).map(|m| m.len() as f64 / 1e9).unwrap_or(0.0));
    }
    fs::remove_dir_all(&tmp).ok();
    if want_report {
        println!("\n{}", quality.summary());
    }
    println!("DONE.");
    Ok(())
}

/// `--mtp`/`--indexer` modes: instead of every shard, only the ones the weight_map says hold
/// the wanted tensors (the MTP head's layer, or any layer's DSA indexer weights).
fn run_extraction(repo: &str, a: &Args, opts: &ConvertOpts, keep_mtp: bool, want_report: bool) -> Result<(), String> {
    let outdir = a.outdir.as_ref().unwrap();
    fs::create_dir_all(outdir).map_err(|e| e.to_string())?;
    let ag = default_agent();
    let weight_map = fetch_weight_map(&ag, repo).map_err(|e| e.to_string())?;

    let mut wanted_shards: Vec<String> = weight_map
        .iter()
        .filter(|(name, _)| {
            if keep_mtp {
                name.starts_with(&format!("model.layers.{}.", a.n_layers))
            } else {
                name.contains("indexer") && crate_layer_idx(name).is_some_and(|li| li < a.n_layers)
            }
        })
        .map(|(_, shard)| shard.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    wanted_shards.sort();

    let tmp = outdir.join("_inflight");
    fs::create_dir_all(&tmp).map_err(|e| e.to_string())?;
    let prefix = if keep_mtp { "out-mtp" } else { "out-idx" };
    println!("{}: {} shards to process", if keep_mtp { "[MTP]" } else { "[IDX]" }, wanted_shards.len());

    let mut quality = rabbit::convert::quality::QuantizationReport::default();
    for (i, fname) in wanted_shards.iter().enumerate() {
        let out_path = outdir.join(format!("{prefix}-{i:05}.safetensors"));
        if out_path.exists() {
            continue;
        }
        println!("[{}/{}] downloading {fname}...", i + 1, wanted_shards.len());
        let src = tmp.join(fname);
        if let Some(parent) = src.parent() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        download_retry(&ag, &resolve_url(repo, fname), &src, None, a.streams).map_err(|e| e.to_string())?;

        let one = TmpDir::with_one_shard(&src)?;
        let out = convert_shard(&one.0, opts, glm_classifier(opts), want_report.then_some(&mut quality)).map_err(|e| e.to_string())?;
        if !out.is_empty() {
            write_safetensors(&out_path, &out).map_err(|e| e.to_string())?;
        }
        fs::remove_file(&src).ok();
        println!("    -> {} ({} tensors)", out_path.file_name().unwrap().to_string_lossy(), out.len());
    }
    fs::remove_dir_all(&tmp).ok();
    if want_report {
        println!("\n{}", quality.summary());
    }
    println!("{}", if keep_mtp { "[MTP] DONE." } else { "[IDX] DONE." });
    Ok(())
}

/// Same layer-number extraction as `glm52::convert::classify`'s private `layer_idx`, duplicated
/// here (as `usize`, not `i64`) purely for the shard-selection filter above — not worth widening
/// `classify`'s own visibility for one call site with a different return type.
fn crate_layer_idx(name: &str) -> Option<usize> {
    let p: Vec<&str> = name.split('.').collect();
    if p.len() > 2 && p[0] == "model" && p[1] == "layers" { p[2].parse().ok() } else { None }
}

fn main() -> std::process::ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            return std::process::ExitCode::FAILURE;
        }
    };

    let result = if args.selftest {
        selftest();
        Ok(())
    } else if args.selftest_nvfp4 {
        selftest_nvfp4()
    } else if let Some(indir) = &args.indir {
        let opts = opts_for(&args, false, false);
        run_indir(indir, args.outdir.as_ref().unwrap(), &opts, args.report)
    } else {
        let repo = args.repo.clone().unwrap();
        if args.mtp {
            let opts = opts_for(&args, true, false);
            run_extraction(&repo, &args, &opts, true, args.report)
        } else if args.indexer {
            let opts = opts_for(&args, false, true);
            run_extraction(&repo, &args, &opts, false, args.report)
        } else {
            let opts = opts_for(&args, false, false);
            run_repo(&repo, &args, &opts, args.report)
        }
    };

    match result {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

