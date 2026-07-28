//! Rabbit's own architecture-agnostic checkpoint converter — the V1 from
//! `resumen-conversor-rabbit.md`'s design discussion. Unlike `convert_fp8_to_int4` (a faithful
//! port of colibrì's GLM-5.2-specific script), this uses `classify_generic`'s name+dimensionality
//! heuristic, so it can run against ANY architecture's checkpoint without per-model code changes
//! — including ones rabbit's own `model.rs` doesn't support for inference yet. Local-only for
//! now (`--indir`, no `--repo`/download) — the doc's own "no diseñar para casos hipotéticos"
//! discipline: HF-repo support gets added when a real case needs it, reusing
//! `glm52::convert::{download,hf_api}` (already architecture-agnostic themselves) rather than
//! duplicating them.
//!
//! `--classify-only` prints the classification report (`crate::convert::report`) and exits
//! without writing anything — "el log de clasificación... visible antes de escribir nada" from
//! the design doc, a sanity check before trusting a new architecture's tensors to the heuristic.

use rabbit::convert::classify::classify_generic;
use rabbit::convert::report::classify_report;
use rabbit::convert::tempdir::TmpDir;
use rabbit::glm52::convert::shard::{convert_shard, BitsMap, ConvertOpts};
use rabbit::glm52::convert::writer::write_safetensors;
use rabbit::safetensors::Shards;
use std::fs;
use std::path::{Path, PathBuf};

const USAGE: &str = "usage: convert --indir <dir> --outdir <dir> [--dense-bits N] [--io-bits N] \
[--expert-bits N] [--group-size N] [--classify-only] [--report] [--bits-override PATTERN:BITS ...]";

struct Args {
    indir: PathBuf,
    outdir: Option<PathBuf>,
    dense_bits: u8,
    io_bits: u8,
    expert_bits: u8,
    group_size: usize,
    classify_only: bool,
    report: bool,
    overrides: Vec<(String, u8)>,
}

fn parse_args() -> Result<Args, String> {
    let mut indir = None;
    let mut outdir = None;
    let mut dense_bits = 4u8;
    let mut io_bits = 8u8;
    let mut expert_bits = 4u8;
    let mut group_size = 0usize;
    let mut classify_only = false;
    let mut report = false;
    let mut overrides = Vec::new();

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut next = |flag: &str| args.next().ok_or_else(|| format!("{flag} needs a value"));
        match a.as_str() {
            "--indir" => indir = Some(PathBuf::from(next("--indir")?)),
            "--outdir" => outdir = Some(PathBuf::from(next("--outdir")?)),
            "--dense-bits" => dense_bits = next("--dense-bits")?.parse().map_err(|e| format!("--dense-bits: {e}"))?,
            "--io-bits" => io_bits = next("--io-bits")?.parse().map_err(|e| format!("--io-bits: {e}"))?,
            "--expert-bits" => expert_bits = next("--expert-bits")?.parse().map_err(|e| format!("--expert-bits: {e}"))?,
            "--group-size" => group_size = next("--group-size")?.parse().map_err(|e| format!("--group-size: {e}"))?,
            "--classify-only" => classify_only = true,
            "--report" => report = true,
            "--bits-override" => overrides.push(rabbit::convert::parse_bits_override(&next("--bits-override")?)?),
            "-h" | "--help" => return Err(USAGE.to_string()),
            other => return Err(format!("unknown argument: {other}\n\n{USAGE}")),
        }
    }

    let indir = indir.ok_or_else(|| format!("--indir is required\n\n{USAGE}"))?;
    if !classify_only && outdir.is_none() {
        return Err(format!("--outdir is required unless --classify-only\n\n{USAGE}"));
    }

    Ok(Args { indir, outdir, dense_bits, io_bits, expert_bits, group_size, classify_only, report, overrides })
}

fn opts_from(a: &Args) -> ConvertOpts {
    // n_layers/keep_mtp/keep_idx are GLM-5.2-specific concepts `classify_generic` never looks
    // at; ConvertOpts still carries the fields (shared with glm52::convert), just unused here.
    ConvertOpts {
        n_layers: 0,
        ebits: a.dense_bits,
        io_bits: a.io_bits,
        xbits: a.expert_bits,
        keep_mtp: false,
        keep_idx: false,
        group_size: a.group_size,
        bits_map: BitsMap::default(),
        overrides: a.overrides.clone(),
    }
}

fn list_shards(indir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut shards: Vec<PathBuf> = fs::read_dir(indir)
        .map_err(|e| e.to_string())?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("safetensors"))
        .collect();
    shards.sort();
    Ok(shards)
}

/// `--classify-only`: runs the heuristic over every shard and prints one combined report,
/// writing nothing — the "visible before writing anything" preview the design doc asked for.
fn run_classify_only(indir: &Path) -> Result<(), String> {
    let shards = list_shards(indir)?;
    if shards.is_empty() {
        return Err(format!("no .safetensors shards found in {}", indir.display()));
    }
    for sp in &shards {
        let one = TmpDir::with_one_shard(sp)?;
        let opened = Shards::open(&one.0).map_err(|e| e.to_string())?;
        let report = classify_report(&opened, classify_generic);
        println!("{}:", sp.file_name().unwrap().to_string_lossy());
        print!("{}", report.summary());
    }
    Ok(())
}

fn run_convert(indir: &Path, outdir: &Path, opts: &ConvertOpts, want_report: bool) -> Result<(), String> {
    fs::create_dir_all(outdir).map_err(|e| e.to_string())?;
    let shards = list_shards(indir)?;
    if shards.is_empty() {
        return Err(format!("no .safetensors shards found in {}", indir.display()));
    }

    let mut totals = rabbit::convert::report::ClassificationReport::default();
    let mut quality = rabbit::convert::quality::QuantizationReport::default();
    for (i, sp) in shards.iter().enumerate() {
        let one = TmpDir::with_one_shard(sp)?;
        let opened = Shards::open(&one.0).map_err(|e| e.to_string())?;
        let report = classify_report(&opened, classify_generic);
        for (kind, stats) in &report.buckets {
            let bucket = totals.buckets.entry(*kind).or_default();
            bucket.count += stats.count;
        }
        totals.total += report.total;

        let out = convert_shard(&one.0, opts, classify_generic, want_report.then_some(&mut quality)).map_err(|e| e.to_string())?;
        write_safetensors(&outdir.join(format!("out-{i:05}.safetensors")), &out).map_err(|e| e.to_string())?;
        println!("[{}/{}] converted {} ({} tensors written)", i + 1, shards.len(), sp.file_name().unwrap().to_string_lossy(), out.len());
    }

    println!("\n{}", totals.summary());
    if want_report {
        println!("\n{}", quality.summary());
    }
    let cfg_src = indir.join("config.json");
    if cfg_src.is_file() {
        rabbit::convert::config::copy_config_with_group_size(&cfg_src, outdir, opts.group_size)?;
    }
    println!("DONE: {} shards -> {}", shards.len(), outdir.display());
    Ok(())
}

fn main() -> std::process::ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            return std::process::ExitCode::FAILURE;
        }
    };

    let result = if args.classify_only {
        run_classify_only(&args.indir)
    } else {
        run_convert(&args.indir, args.outdir.as_ref().unwrap(), &opts_from(&args), args.report)
    };

    match result {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}
