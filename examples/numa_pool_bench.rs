//! Phase N2 gate bench (`NUMA_AMX_BRIEF.md` §6): a NUMA-sharded matvec — the prototype of
//! N4a's `matmul_qt_sharded` and the empirical input to decision D1 (per-node vs per-socket
//! pools) — timed against the exact same matvec on today's single global pool.
//!
//! Three configurations over one `[rows, 7168]` int4 weight at s=1 (batch-1 decode, the shape
//! every dense matmul in the token loop has):
//!
//!  - `global`: one `matmul_qt` on the global rayon pool, weight allocated wherever first touch
//!    put it — exactly today's behavior.
//!  - `per-node`: weight rows split into one contiguous block per NUMA node, each block
//!    allocated AND filled inside that node's pinned pool (first touch = placement), then each
//!    step computed as one `run_all` fan-out with every pool running `matmul_qt_rows` over its
//!    own block's rows in parallel within the pool.
//!  - `per-socket`: same, but with nodes merged into per-socket domains (built from node
//!    distances: nodes closer than 20 share a socket) — D1's alternative shape.
//!
//! Not criterion: the pinned pools and placement are process-global state that criterion's
//! multi-process harness would rebuild per benchmark; a plain loop with a discarded warmup and
//! a reported min/median over 30 reps is honest enough for a scheduling-floor comparison.
//!
//! Run on the target box (RAYON_NUM_THREADS sets the global pool AND the per-config total):
//! `RAYON_NUM_THREADS=48 cargo run --release --example numa_pool_bench`

use rabbit::kernels::{matmul_qt, matmul_qt_rows, RowActs};
use rabbit::numa::{self, NumaNode};
use rabbit::quant::QT;
use rayon::prelude::*;
use std::time::Instant;

const COLS: usize = 7168;
const ROWS: usize = 32768; // lm_head-class row count, scaled down 5x to keep fill time sane
const REPS: usize = 30;

fn random_vec(n: usize, seed: &mut u32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 17;
            *seed ^= *seed << 5;
            (*seed as f32 / u32::MAX as f32) - 0.5
        })
        .collect()
}

fn bench<F: FnMut()>(name: &str, mut f: F) {
    f(); // warmup, discarded
    let mut times: Vec<f64> = (0..REPS)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    times.sort_by(f64::total_cmp);
    println!("{name}: min {:.3} ms, median {:.3} ms", times[0], times[REPS / 2]);
}

/// Row-blocks a weight across `domains`, allocating+filling each block inside its domain's
/// pinned pool (first touch), then benches one fan-out per step.
fn bench_sharded(name: &str, domains: &[NumaNode], threads_per: usize, w_src: &[f32], x: &[f32]) {
    let pools: Vec<rayon::ThreadPool> = domains
        .iter()
        .map(|d| {
            let pin = d.clone();
            rayon::ThreadPoolBuilder::new().num_threads(threads_per).start_handler(move |_| numa::pin_current_thread(&pin)).build().unwrap()
        })
        .collect();
    let n = domains.len();
    let per = ROWS.div_ceil(n);
    // Placement: each shard filled inside its own pinned pool -> its pages live on that domain.
    let shards: Vec<QT> = pools
        .iter()
        .enumerate()
        .map(|(i, pool)| {
            pool.install(|| {
                let r0 = i * per;
                let nr = per.min(ROWS - r0);
                let mut t = QT::alloc(nr, COLS, 4, false);
                t.fill(&w_src[r0 * COLS..(r0 + nr) * COLS]);
                t
            })
        })
        .collect();
    let mut outs: Vec<Vec<f32>> = shards.iter().map(|t| vec![0f32; t.rows]).collect();
    bench(name, || {
        let out_refs: Vec<std::sync::Mutex<&mut Vec<f32>>> = outs.iter_mut().map(std::sync::Mutex::new).collect();
        std::thread::scope(|s| {
            for (i, pool) in pools.iter().enumerate() {
                let (shard, out, x) = (&shards[i], &out_refs[i], &x);
                s.spawn(move || {
                    pool.install(|| {
                        let mut out = out.lock().unwrap();
                        let acts = RowActs::prepare(x, shard, 1);
                        let blk = shard.rows.div_ceil(pool.current_num_threads() * 4).max(1);
                        out.par_chunks_mut(blk).enumerate().for_each(|(b, seg)| {
                            matmul_qt_rows(seg, &acts, shard, b * blk, seg.len());
                        });
                    });
                });
            }
        });
    });
}

fn main() {
    let threads: usize = std::env::var("RAYON_NUM_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or_else(|| num_cpus::get_physical());
    rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().unwrap();
    let Some(topo) = numa::topology() else {
        eprintln!("no NUMA topology — nothing to compare");
        return;
    };
    println!("{} nodes, {} total threads, weight [{}x{}] int4", topo.n_nodes(), threads, ROWS, COLS);

    let mut seed = 0x2545F491u32;
    let w_src = random_vec(ROWS * COLS, &mut seed);
    let x = random_vec(COLS, &mut seed);

    // Today's behavior: one global-pool matmul_qt, weight placed by whatever thread fills it.
    let mut w = QT::alloc(ROWS, COLS, 4, false);
    w.fill(&w_src);
    let mut y = vec![0f32; ROWS];
    bench("global-pool matmul_qt   ", || matmul_qt(&mut y, &x, &w, 1));

    bench_sharded("per-node sharded matvec ", &topo.nodes, (threads / topo.n_nodes()).max(1), &w_src, &x);

    // The cross-pool fan-out floor: what does ONE NodePools::run_all cost when the pools have
    // (a) nothing and (b) a token's worth of trivial work to do? K3 decode pays this once per
    // MoE layer (92/token on the real checkpoint) — measured after the serve gate showed ~0.6
    // s/token of expert-phase wall NOT accounted for by node busy time.
    {
        use rabbit::numa::NodePools;
        for per_pool_total in [48usize, 192, 384] {
            // Per-socket domains — the shape NodePools::init builds since the D1 flip; the
            // run_all costs measured here are the ones decode actually pays.
            let topo = rabbit::numa::topology().unwrap();
            let Some(pools) = NodePools::build(topo.socket_domains(rabbit::numa::sys_distance_row), per_pool_total) else { break };
            // Wake the pools once so the first bench iteration isn't a cold outlier beyond
            // what steady decode sees (pools sleep between layers there too).
            pools.run_all(|_| {});
            bench(&format!("run_all noop        (total {per_pool_total})"), || pools.run_all(|_| {}));
            bench(&format!("run_all 1ms-ish work (total {per_pool_total})"), || {
                pools.run_all(|_| {
                    (0..pools.threads_per_pool() * 8).into_par_iter().for_each(|_| {
                        std::hint::black_box((0..20_000u64).fold(0u64, |a, x| a.wrapping_add(x * x)));
                    });
                })
            });
            // Same work, but after a 5 ms idle gap — the shape decode actually has (attention
            // runs between MoE layers, so the pinned pools go to sleep every layer). The delta
            // vs the back-to-back row above separates SLEEP-WAKE cost from task-distribution
            // cost: that split is what the timeboxed "pools stay hot" investigation keys on.
            // (Hand-rolled timing loop rather than `bench` so the sleep itself stays untimed.)
            {
                let mut times: Vec<f64> = (0..REPS)
                    .map(|_| {
                        std::thread::sleep(std::time::Duration::from_millis(5));
                        let t = Instant::now();
                        pools.run_all(|_| {
                            (0..pools.threads_per_pool() * 8).into_par_iter().for_each(|_| {
                                std::hint::black_box((0..20_000u64).fold(0u64, |a, x| a.wrapping_add(x * x)));
                            });
                        });
                        t.elapsed().as_secs_f64() * 1e3
                    })
                    .collect();
                times.sort_by(f64::total_cmp);
                println!("run_all same, 5ms gap (total {per_pool_total}): min {:.3} ms, median {:.3} ms", times[0], times[REPS / 2]);
            }
        }
    }

    // Per-socket domains (the D1 production shape since the flip — numa::socket_domains).
    let sockets = topo.socket_domains(rabbit::numa::sys_distance_row);
    if sockets.len() < topo.n_nodes() {
        bench_sharded("per-socket sharded matvec", &sockets, (threads / sockets.len()).max(1), &w_src, &x);
    }
}
