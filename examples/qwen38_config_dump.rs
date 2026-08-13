//! Loads a real Qwen 3.8 checkpoint directory's `config.json` (+ `generation_config.json`) through
//! `qwen38::config::Cfg` and prints what it parsed — the same "run it against the real file before
//! trusting the parser" check `examples/k3_smoke.rs` does for Kimi K3, minus any weight loading, so
//! it costs milliseconds and needs only the small metadata files (which land long before the 213
//! MXFP4 shards finish downloading).
//!
//! ```text
//! cargo run --release --example qwen38_config_dump -- /mnt/data/qwen38-max-mxfp4
//! ```

use rabbit::qwen38::config::Cfg;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args().nth(1).ok_or("usage: qwen38_config_dump <checkpoint-dir>")?;
    let cfg = Cfg::load(std::path::Path::new(&dir))?;

    println!("hidden {}, layers {}, vocab {}", cfg.hidden, cfg.n_layers, cfg.vocab);
    println!(
        "full attention: {} heads / {} kv heads, head_dim {}, rope on {} of {} dims, output gate {}",
        cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.rope_dim, cfg.head_dim, cfg.attn_output_gate
    );
    println!(
        "linear attention (GDN): {} key heads x {} dims, {} value heads x {} dims ({} value heads per key head), conv kernel {}",
        cfg.lin_key_heads,
        cfg.lin_key_head_dim,
        cfg.lin_value_heads,
        cfg.lin_value_head_dim,
        cfg.lin_heads_per_key(),
        cfg.conv_kernel
    );
    println!(
        "layers: {} linear / {} full attention; pattern starts {:?}",
        cfg.is_linear.iter().filter(|&&l| l).count(),
        cfg.n_full_attn_layers(),
        &cfg.is_linear[..8.min(cfg.is_linear.len())]
    );
    println!(
        "MoE: {} experts, top-{} (norm_topk {}), moe_inter {}, shared expert inter {}",
        cfg.n_experts, cfg.topk, cfg.norm_topk, cfg.moe_inter, cfg.shared_inter
    );
    println!("eps {}, theta {}, attn_scale {}", cfg.eps, cfg.theta, cfg.attn_scale);
    println!("stop ids {:?}, mtp layers {} (skipped)", cfg.stop_ids, cfg.mtp_layers);
    println!("experts natively MXFP4: {}, rabbit group_size {}", cfg.mxfp4_experts, cfg.group_size);

    // Per-token routed-expert traffic: the number that sets decode speed on a disk-streaming
    // engine. MXFP4 is 4 bits + one E8M0 byte per 32 values = 0.53125 bytes/param.
    let per_expert = 3.0 * cfg.hidden as f64 * cfg.moe_inter as f64;
    let per_token = per_expert * cfg.topk as f64 * cfg.n_layers as f64;
    let bytes = if cfg.mxfp4_experts { 0.53125 } else { 2.0 };
    println!(
        "\nrouted experts: {:.1} B params/token ({:.1} GB/token at {} bytes/param), {:.2} TB total on disk",
        per_token / 1e9,
        per_token * bytes / 1e9,
        bytes,
        per_expert * cfg.n_experts as f64 * cfg.n_layers as f64 * bytes / 1e12
    );
    Ok(())
}
