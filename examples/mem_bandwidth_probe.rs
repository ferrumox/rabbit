//! Standalone raw memory-bandwidth probe (1 thread vs. rayon's full pool), independent of any
//! matmul kernel — establishes ground truth for `PERFORMANCE_KIMI_K3.md`'s "Where does compute
//! actually go?" question: is a single CPU core already close to this machine's achievable
//! memory bandwidth (which would explain why more rayon threads barely helped
//! `matmul_mxfp4_k3_dims_cold`), or is total system bandwidth much higher than anything measured
//! there (meaning something other than raw DRAM bandwidth is the real limit)? Throwaway
//! diagnostic, not tied to any rabbit kernel — kept only if it turns out to answer the question
//! cleanly.

use rayon::prelude::*;
use std::time::Instant;

fn sum_bytes(buf: &[u8]) -> u64 {
    // One read per cache line (64B), not every byte — measures line-fetch bandwidth, not
    // per-byte ALU throughput, matching what a real matmul's memory traffic looks like.
    buf.iter().step_by(64).map(|&b| b as u64).sum()
}

fn main() {
    const SIZE: usize = 1024 * 1024 * 1024; // 1GiB — far bigger than this machine's 24MiB L3
    let buf = vec![7u8; SIZE];

    for round in 1..=3 {
        let t = Instant::now();
        let s1 = sum_bytes(&buf);
        let e1 = t.elapsed();
        let gbps1 = (SIZE as f64 / e1.as_secs_f64()) / 1e9;
        println!("round {round}: 1 thread:   {gbps1:>7.2} GB/s ({:>6.1} ms, checksum {s1})", e1.as_secs_f64() * 1000.0);

        let nthreads = rayon::current_num_threads();
        let chunk = SIZE / nthreads;
        let t = Instant::now();
        let s2: u64 = buf.par_chunks(chunk).map(sum_bytes).sum();
        let e2 = t.elapsed();
        let gbps2 = (SIZE as f64 / e2.as_secs_f64()) / 1e9;
        println!("round {round}: {nthreads} threads: {gbps2:>7.2} GB/s ({:>6.1} ms, checksum {s2})", e2.as_secs_f64() * 1000.0);
    }
}
