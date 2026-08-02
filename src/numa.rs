//! NUMA topology discovery + placement primitives (Phase N1 of `NUMA_AMX_BRIEF.md`).
//!
//! The design principle this module serves (brief §1): all tensors of one computation group live
//! on one NUMA node, and the compute for that group runs on that node's cores. This is
//! *bandwidth* locality, not cache locality — in decode every expert-weight byte is read exactly
//! once per token, so what placement buys is concurrent use of all nodes' DRAM channels instead
//! of hammering the one or two nodes first-touch happened to fill (measured on the target box:
//! 183 GB/s node-local reads vs 87 GB/s cross-socket — a 2.1× split, `PERFORMANCE.md` Phase N0).
//!
//! Everything here is `libc`-only (house rule: no `hwloc`, no `numa`/`numactl` crates —
//! `mbind`/`get_mempolicy` have no glibc wrapper at all, they only exist as wrappers in the
//! banned libnuma, so they are raw `libc::syscall` calls). The whole module is compiled twice:
//! the real implementation for Linux, and a no-op twin for everything else, so the owner's Mac
//! and the single-node dev laptop build and behave exactly as before. Nothing here may hardcode
//! node or core counts — the brief's provenance caveat (§1) is explicit that topology comes from
//! [`topology()`] at runtime, never from documentation.
//!
//! Placement mechanism of choice is **first touch on a pinned thread** (an allocation's pages
//! land on the node of the CPU that first writes them): wherever a load/init loop already runs
//! on a thread we control — the expert preload path — pinning that thread is the entire
//! placement story, zero syscalls per buffer. [`bind_region`]/[`interleave_region`] exist for
//! buffers whose faulting thread can't be controlled (e.g. a reserve-then-grow KV cache, Phase
//! N4c): `mbind` sets a **VMA policy**, so pages faulted *later* in the range follow it — but it
//! does not migrate pages already touched (first-touch discipline instead of `MPOL_MF_MOVE`;
//! moving pages is a rescue, not a design).

#[cfg(target_os = "linux")]
pub use linux::*;

#[cfg(not(target_os = "linux"))]
pub use stub::*;

use std::sync::OnceLock;

/// The home node of routed expert `eid` in layer `layer` (Phase N3a): static, deterministic,
/// stateless — `(layer * n_experts + eid) % n_nodes`. Placement (which node's pool loads the
/// expert, so first touch lands its pages there) and dispatch (which pool computes it) MUST both
/// come through here: the whole point is that the two always agree, and a pure function of
/// `(layer, eid)` can't drift the way persisted state could. The `layer *` term rotates the
/// assignment so an expert id hot in many layers doesn't pile onto one node.
pub fn home_node(layer: usize, n_experts: usize, eid: usize, n_nodes: usize) -> usize {
    (layer * n_experts + eid) % n_nodes.max(1)
}

/// One pinned worker pool per NUMA **domain** (Phase N2, domain choice = decision D1). Built at
/// most once, only when `--numa` is active AND the machine really has multiple nodes; everything
/// else leaves the singleton `None` and every consumer takes its existing global-pool path
/// unchanged.
///
/// **D1: domains are per-SOCKET** (2026-08-02, owner-approved flip from the initial per-node
/// build) — see [`Topology::socket_domains`] for the grouping rule and the measured skew/locality
/// argument. On a non-SNC machine where every node IS a socket, the grouping is the identity and
/// nothing changes.
///
/// **Pool size derives from the effective `--threads` total, not the core count**
/// (`total / n_domains`, min 1 per domain): the owner's measured 48-beats-192 result means the
/// thread sweep must stay meaningful with `--numa` on — hardcoding cores-per-domain would
/// silently override it (brief §6).
///
/// The hazard this design is shaped around (brief §6, trap #2): **never `pool.install(..)` from
/// inside another pool's worker** — it blocks a worker thread waiting on a different pool, which
/// deadlocks under load. [`NodePools::run_all`] is the only cross-pool fan-out and it always
/// orchestrates from a NON-pool thread (debug-asserted).
pub struct NodePools {
    pools: Vec<(NumaNode, rayon::ThreadPool)>,
}

static NODE_POOLS: OnceLock<Option<NodePools>> = OnceLock::new();

impl NodePools {
    /// Builds the singleton from [`topology()`] with `total_threads` workers spread evenly
    /// across nodes, pinning each pool's workers to its node's CPUs. Returns the reason as a
    /// human-readable `Err` when NUMA execution isn't possible (single node, no topology,
    /// non-Linux, or a pool failing to build) — the caller logs it once; all consumers then see
    /// [`node_pools()`] `== None` and degrade to the global pool.
    pub fn init(total_threads: usize) -> Result<&'static NodePools, String> {
        let built = NODE_POOLS.get_or_init(|| {
            let topo = topology()?;
            if topo.n_nodes() < 2 {
                return None;
            }
            Self::build(topo.socket_domains(sys_distance_row), total_threads)
        });
        built.as_ref().ok_or_else(|| "no multi-node NUMA topology (or pool build failed) — running without node pools".to_string())
    }

    /// The non-singleton constructor behind [`NodePools::init`] — public so tests and benches can
    /// build throwaway pools against explicit domains (per-socket via
    /// [`Topology::socket_domains`], per-node via `topo.nodes`, or synthetic ones) WITHOUT
    /// flipping the process-global singleton, which would silently switch every concurrently
    /// running test onto the NUMA code paths.
    pub fn build(domains: Vec<NumaNode>, total_threads: usize) -> Option<NodePools> {
        if domains.len() < 2 {
            return None;
        }
        let per = (total_threads / domains.len()).max(1);
        let mut pools = Vec::with_capacity(domains.len());
        for domain in domains {
            let pin_to = domain.clone();
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(per)
                .start_handler(move |_| pin_current_thread(&pin_to))
                .build()
                .ok()?;
            pools.push((domain, pool));
        }
        Some(NodePools { pools })
    }

    /// The pools if `--numa` was requested and [`NodePools::init`] succeeded, else `None`. This
    /// is the one switch every NUMA-aware code path keys off.
    pub fn get() -> Option<&'static NodePools> {
        NODE_POOLS.get().and_then(|p| p.as_ref())
    }

    pub fn n(&self) -> usize {
        self.pools.len()
    }

    pub fn threads_per_pool(&self) -> usize {
        self.pools[0].1.current_num_threads()
    }

    /// Runs `f(i)` INSIDE pool `i`, for all pools concurrently, returning when every call is
    /// done — the one fork/join a NUMA-dispatched MoE layer pays. `f(i)` may freely use rayon
    /// parallelism; it executes on pool `i`'s pinned workers, so nested `par_iter`s stay on that
    /// domain. A pool with nothing to do should just return immediately from its `f(i)`.
    ///
    /// Orchestration is NESTED `in_place_scope`s, one per pool, recursed on the calling thread:
    /// each scope's `spawn` injects `f(i)` into ITS pool, the innermost recursion step then
    /// unwinds, and each scope exit blocks (on the orchestrator thread only — never a pool
    /// worker, so the install-from-another-pool deadlock stays structurally impossible) until
    /// its pool's task is done. No OS threads are spawned and no `'static` bound is needed —
    /// this replaced the original one-scoped-thread-per-pool shape after the N3 serve gate
    /// measured the per-call cost: ~0.4 ms/call × 92 MoE layers ≈ 35 ms/token of pure spawn
    /// overhead. (The larger cost of waking a SLEEPING pool's workers remains either way; see
    /// `PERFORMANCE.md`'s N3 serve-gate decomposition.)
    ///
    /// Must not be called from inside any rayon pool (that would re-create the blocked-worker
    /// hazard one level up); debug-asserted.
    pub fn run_all<F: Fn(usize) + Sync>(&self, f: F) {
        debug_assert!(rayon::current_thread_index().is_none(), "NodePools::run_all must be orchestrated from a non-pool thread");
        fn nest<F: Fn(usize) + Sync>(pools: &[(NumaNode, rayon::ThreadPool)], i: usize, f: &F) {
            let Some((_, pool)) = pools.get(i) else { return };
            pool.in_place_scope(|s| {
                s.spawn(move |_| f(i));
                nest(pools, i + 1, f);
            });
        }
        nest(&self.pools, 0, &f);
    }
}

/// One NUMA node: its kernel id and the logical CPUs it owns, as parsed from
/// `/sys/devices/system/node/node<id>/cpulist`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NumaNode {
    pub id: usize,
    pub cpus: Vec<usize>,
}

/// The machine's NUMA layout, nodes sorted by id. Obtained via [`topology()`]; a machine (or OS)
/// where that returns `None` — or a topology with fewer than 2 nodes — means every downstream
/// consumer degrades to exactly the current single-pool behavior.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Topology {
    pub nodes: Vec<NumaNode>,
}

impl Topology {
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Groups the nodes into per-socket domains — one synthetic [`NumaNode`] per socket holding
    /// the union of its nodes' CPUs (ids renumbered 0..n_sockets) — using the kernel's node
    /// distance matrix: two nodes share a socket iff their mutual distance is below 20 (SNC
    /// sub-NUMA domains report 10–17 to each other, 21+ across a socket boundary; the same
    /// threshold `numactl --hardware` output makes obvious on this box). Falls back to the
    /// per-node identity when a distance row is unreadable or the grouping merges nothing.
    ///
    /// This is decision **D1 resolved to per-socket** (2026-08-02, owner-approved), on N3's own
    /// measured data: per-token expert wall was dominated by per-LAYER routing skew — with 6
    /// per-node pools the busiest node draws ~4.5–5 of 16 experts against a 2.7 mean (~1.7×),
    /// while 2 per-socket domains draw ~9.5 of 16 against an 8 mean (~1.19×) — and the N0a probe
    /// measured intra-socket-remote reads at only −4% vs node-local, so widening the domain
    /// buys back most of the skew cost for almost no locality cost. (First touch inside a
    /// socket-pinned pool lands pages on whichever of the socket's nodes the faulting CPU
    /// belongs to — still socket-local, which is what the −4% number says is enough.)
    pub fn socket_domains(&self, distance_row: impl Fn(usize) -> Option<String>) -> Vec<NumaNode> {
        let mut groups: Vec<Vec<&NumaNode>> = Vec::new();
        'outer: for node in &self.nodes {
            let Some(row) = distance_row(node.id) else { return self.nodes.clone() };
            let dists: Vec<usize> = row.split_whitespace().filter_map(|t| t.parse().ok()).collect();
            for g in &mut groups {
                if dists.get(g[0].id).is_some_and(|&d| d < 20) {
                    g.push(node);
                    continue 'outer;
                }
            }
            groups.push(vec![node]);
        }
        groups
            .into_iter()
            .enumerate()
            .map(|(i, g)| NumaNode { id: i, cpus: g.iter().flat_map(|n| n.cpus.iter().copied()).collect() })
            .collect()
    }
}

/// Parses one `cpulist` string (`"0-31,192-223"`, `"7"`, `"0-3,8"`) into the CPU ids it names.
/// Kernel format: comma-separated decimal ranges, each either `a` or `a-b` inclusive. Pure
/// function so the parser is unit-testable without a `/sys` to read; `None` on anything
/// malformed rather than a guess.
fn parse_cpulist(s: &str) -> Option<Vec<usize>> {
    let mut cpus = Vec::new();
    for part in s.trim().split(',') {
        if part.is_empty() {
            return None;
        }
        match part.split_once('-') {
            Some((a, b)) => {
                let (a, b) = (a.parse::<usize>().ok()?, b.parse::<usize>().ok()?);
                if b < a {
                    return None;
                }
                cpus.extend(a..=b);
            }
            None => cpus.push(part.parse().ok()?),
        }
    }
    Some(cpus)
}

#[cfg(target_os = "linux")]
mod linux {
    use super::{parse_cpulist, NumaNode, Topology};

    /// Reads the machine's NUMA layout from `/sys/devices/system/node/node*/cpulist`. `None` if
    /// the directory doesn't exist or any node's cpulist fails to parse — the caller treats that
    /// exactly like a single-node machine (no placement, current behavior).
    pub fn topology() -> Option<Topology> {
        let mut nodes = Vec::new();
        for entry in std::fs::read_dir("/sys/devices/system/node").ok()? {
            let entry = entry.ok()?;
            let name = entry.file_name();
            let name = name.to_str()?;
            let Some(id) = name.strip_prefix("node").and_then(|n| n.parse::<usize>().ok()) else {
                continue; // `has_cpu`, `possible`, etc. — not node directories
            };
            let cpulist = std::fs::read_to_string(entry.path().join("cpulist")).ok()?;
            let cpus = parse_cpulist(&cpulist)?;
            if !cpus.is_empty() {
                // CPU-less (memory-only) nodes exist on some machines; they can't host a pinned
                // pool, so they are simply not schedulable entities here.
                nodes.push(NumaNode { id, cpus });
            }
        }
        if nodes.is_empty() {
            return None;
        }
        nodes.sort_by_key(|n| n.id);
        Some(Topology { nodes })
    }

    /// Pins the calling thread to `node`'s CPUs (`sched_setaffinity` on the current thread).
    /// Best-effort: on failure the thread simply stays where the scheduler put it — placement
    /// quality degrades, correctness doesn't.
    ///
    /// Also resets the calling thread's memory policy to the kernel default (first-touch-local)
    /// via `set_mempolicy(MPOL_DEFAULT)`. Memory policy is inherited per-thread from the
    /// process: under the canonical `numactl --interleave=all` launch, every thread starts with
    /// `MPOL_INTERLEAVE`, which would spread a pinned worker's allocations across ALL nodes and
    /// silently defeat the entire first-touch placement scheme this function exists for. The
    /// reset is thread-local — the main thread (dense-weight loads, which N4 hasn't sharded yet)
    /// keeps whatever policy the launch prefix set.
    pub fn pin_current_thread(node: &NumaNode) {
        unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            for &cpu in &node.cpus {
                if cpu < 8 * std::mem::size_of::<libc::cpu_set_t>() {
                    libc::CPU_SET(cpu, &mut set);
                }
            }
            // 0 = the calling thread. Failure is not worth even a log line per thread; the
            // pool builder logs once if the topology looks unusable.
            libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
            libc::syscall(libc::SYS_set_mempolicy, libc::MPOL_DEFAULT, std::ptr::null::<libc::c_ulong>(), 0 as libc::c_ulong);
        }
    }

    /// `maxnode` for the raw `mbind`/`get_mempolicy` syscalls: the kernel's `get_nodes()`
    /// internally does `--maxnode` before `BITS_TO_LONGS(maxnode)`, so passing one u64 mask word
    /// as 65 declared bits makes it read exactly that one word (bits 0–63 — machines with more
    /// than 64 NUMA nodes are out of scope, checked at the call sites).
    const MAXNODE: libc::c_ulong = 65;

    /// The whole-page subrange of `[addr, addr+len)` — `mbind` demands a page-aligned start and
    /// rejects anything else with `EINVAL`, and heap pointers are NOT page-aligned in general
    /// (glibc's mmap'd large chunks carry a 16-byte header, so even a fresh 16 MiB `Vec` starts
    /// 16 bytes past the page boundary). The sliver pages outside the aligned range keep the
    /// default policy and land wherever first touch puts them: at the multi-MB buffer sizes this
    /// module exists for, two stray pages are noise.
    fn page_aligned(addr: *mut u8, len: usize) -> Option<(usize, usize)> {
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let start = (addr as usize).next_multiple_of(page);
        let end = (addr as usize + len) / page * page;
        (end > start).then_some((start, end - start))
    }

    /// Sets an `MPOL_BIND`-to-`node` VMA policy on the whole pages of `[addr, addr+len)`. Pages
    /// already faulted stay where they are (no `MPOL_MF_MOVE` — see the module doc); pages
    /// faulted after this call land on `node`. Returns `false` (having done nothing) on syscall
    /// failure, a node id past the one-word mask, or a range without a single whole page in it.
    ///
    /// # Safety
    /// `addr..addr+len` must be a valid mapping owned by the caller for the policy's lifetime.
    pub unsafe fn bind_region(addr: *mut u8, len: usize, node: usize) -> bool {
        let Some((start, alen)) = page_aligned(addr, len) else {
            return false;
        };
        if node >= 64 {
            return false;
        }
        let mask: libc::c_ulong = 1 << node;
        unsafe { libc::syscall(libc::SYS_mbind, start, alen, libc::MPOL_BIND, &mask, MAXNODE, 0) == 0 }
    }

    /// Sets an `MPOL_INTERLEAVE`-across-all-nodes VMA policy on the whole pages of
    /// `[addr, addr+len)` — the in-process equivalent of launching under
    /// `numactl --interleave=all`, for one buffer. Same fault-time semantics and failure
    /// behavior as [`bind_region`].
    ///
    /// # Safety
    /// Same contract as [`bind_region`].
    pub unsafe fn interleave_region(addr: *mut u8, len: usize, topo: &Topology) -> bool {
        let Some((start, alen)) = page_aligned(addr, len) else {
            return false;
        };
        let mut mask: libc::c_ulong = 0;
        for n in &topo.nodes {
            if n.id >= 64 {
                return false;
            }
            mask |= 1 << n.id;
        }
        unsafe { libc::syscall(libc::SYS_mbind, start, alen, libc::MPOL_INTERLEAVE, &mask, MAXNODE, 0) == 0 }
    }

    /// One node's row of the kernel distance matrix, for [`super::Topology::socket_domains`] —
    /// e.g. `"10 15 17 21 28 26"` for node 0 on the target box.
    pub fn sys_distance_row(node_id: usize) -> Option<String> {
        std::fs::read_to_string(format!("/sys/devices/system/node/node{node_id}/distance")).ok()
    }

    /// Which node the page containing `addr` is resident on, via
    /// `get_mempolicy(MPOL_F_NODE | MPOL_F_ADDR)` — `None` if the syscall fails (e.g. the page
    /// was never faulted). Diagnostic/test instrument, not a hot-path call.
    pub fn node_of_page(addr: *const u8) -> Option<usize> {
        // get_mempolicy query flags; not exported by the libc crate (they have no glibc wrapper
        // either — values from linux/include/uapi/linux/mempolicy.h, stable ABI).
        const MPOL_F_NODE: libc::c_ulong = 1 << 0;
        const MPOL_F_ADDR: libc::c_ulong = 1 << 1;
        let mut node: libc::c_int = -1;
        let rc = unsafe {
            libc::syscall(
                libc::SYS_get_mempolicy,
                &mut node,
                std::ptr::null_mut::<libc::c_ulong>(),
                0 as libc::c_ulong,
                addr,
                MPOL_F_NODE | MPOL_F_ADDR,
            )
        };
        (rc == 0 && node >= 0).then_some(node as usize)
    }
}

#[cfg(not(target_os = "linux"))]
mod stub {
    use super::{NumaNode, Topology};

    /// Non-Linux twin: there is no NUMA API to speak to, so there is no topology — every
    /// consumer takes its single-node degradation path. (macOS has no NUMA placement syscalls;
    /// Apple silicon is UMA anyway.)
    pub fn topology() -> Option<Topology> {
        None
    }

    pub fn pin_current_thread(_node: &NumaNode) {}

    /// # Safety
    /// No-op; same signature as the Linux twin so call sites don't need their own cfg.
    pub unsafe fn bind_region(_addr: *mut u8, _len: usize, _node: usize) -> bool {
        false
    }

    /// # Safety
    /// No-op; same signature as the Linux twin so call sites don't need their own cfg.
    pub unsafe fn interleave_region(_addr: *mut u8, _len: usize, _topo: &Topology) -> bool {
        false
    }

    pub fn node_of_page(_addr: *const u8) -> Option<usize> {
        None
    }

    pub fn sys_distance_row(_node_id: usize) -> Option<String> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpulist_parser_handles_the_kernel_formats() {
        assert_eq!(parse_cpulist("0-31,192-223\n").unwrap().len(), 64);
        assert_eq!(parse_cpulist("0-31,192-223").unwrap()[32], 192);
        assert_eq!(parse_cpulist("7").unwrap(), vec![7]);
        assert_eq!(parse_cpulist("0-3,8").unwrap(), vec![0, 1, 2, 3, 8]);
        assert_eq!(parse_cpulist("0"), Some(vec![0]));
    }

    /// The target box's real distance matrix must group its 6 SNC3 nodes into 2 sockets
    /// {0,1,2} and {3,4,5}; a machine whose "nodes" are already sockets (all cross distances
    /// ≥ 20) must come back unchanged; unreadable distances fall back to per-node identity.
    #[test]
    fn socket_domains_groups_snc_nodes_by_distance_and_falls_back_per_node() {
        let topo = Topology {
            nodes: (0..6).map(|id| NumaNode { id, cpus: vec![id * 10, id * 10 + 1] }).collect(),
        };
        let matrix = ["10 15 17 21 28 26", "15 10 15 23 26 23", "17 15 10 26 23 21", "21 28 26 10 15 17", "23 26 23 15 10 15", "26 23 21 17 15 10"];
        let sockets = topo.socket_domains(|id| Some(matrix[id].to_string()));
        assert_eq!(sockets.len(), 2);
        assert_eq!(sockets[0].cpus, vec![0, 1, 10, 11, 20, 21]);
        assert_eq!(sockets[1].cpus, vec![30, 31, 40, 41, 50, 51]);

        let two = Topology { nodes: (0..2).map(|id| NumaNode { id, cpus: vec![id] }).collect() };
        let same = two.socket_domains(|id| Some(["10 21", "21 10"][id].to_string()));
        assert_eq!(same.len(), 2, "already-per-socket nodes must not merge");

        assert_eq!(topo.socket_domains(|_| None), topo.nodes, "unreadable distances fall back to per-node");
    }

    #[test]
    fn cpulist_parser_rejects_malformed_input_instead_of_guessing() {
        assert_eq!(parse_cpulist(""), None);
        assert_eq!(parse_cpulist("a-b"), None);
        assert_eq!(parse_cpulist("3-1"), None);
        assert_eq!(parse_cpulist("1,,2"), None);
        assert_eq!(parse_cpulist("1-"), None);
    }

    /// `run_all(f)` must execute `f(i)` — and any rayon work `f(i)` fans out — on pool `i`'s
    /// pinned CPUs, concurrently across pools. Checked with `sched_getcpu` from inside a
    /// `par_iter` running in each pool: every observed CPU must belong to that pool's node.
    /// SKIP-not-fail off Linux/multi-node, same as the placement test below.
    #[test]
    fn node_pools_run_work_on_their_own_nodes_cpus() {
        #[cfg(target_os = "linux")]
        {
            let Some(topo) = topology() else {
                eprintln!("SKIP: no NUMA topology on this machine");
                return;
            };
            if topo.n_nodes() < 2 {
                eprintln!("SKIP: single NUMA node — pinning is unobservable");
                return;
            }
            // `build`, NOT `init`: setting the process-global singleton from a test would
            // silently switch every other test in this binary onto the NUMA code paths.
            // Per-node identity domains, so "pool i runs on node i's CPUs" stays assertable.
            let pools = NodePools::build(topo.nodes.clone(), topo.n_nodes() * 2).expect("multi-node box must build pools");
            use rayon::prelude::*;
            let seen: Vec<std::sync::Mutex<Vec<usize>>> = (0..pools.n()).map(|_| std::sync::Mutex::new(Vec::new())).collect();
            pools.run_all(|i| {
                (0..64).into_par_iter().for_each(|_| {
                    let cpu = unsafe { libc::sched_getcpu() };
                    assert!(cpu >= 0);
                    seen[i].lock().unwrap().push(cpu as usize);
                });
            });
            for (i, node) in topo.nodes.iter().enumerate() {
                for &cpu in seen[i].lock().unwrap().iter() {
                    assert!(node.cpus.contains(&cpu), "pool {i} ran on CPU {cpu}, which is not on node {}", node.id);
                }
            }
        }
    }

    /// The `numactl --interleave=all` hazard: memory policy is inherited by new threads from
    /// their creator, so under the canonical interleaved launch every pool worker would START
    /// with `MPOL_INTERLEAVE` — and first touch would spray each expert's pages across all nodes
    /// instead of homing them. `pin_current_thread` must reset the worker's policy to default;
    /// this reproduces the launch condition (interleave set on the spawning thread) and asserts
    /// a pinned child's first-touched pages still land on ITS node.
    #[test]
    fn pin_current_thread_overrides_an_inherited_interleave_policy() {
        #[cfg(target_os = "linux")]
        {
            let Some(topo) = topology() else {
                eprintln!("SKIP: no NUMA topology on this machine");
                return;
            };
            if topo.n_nodes() < 2 {
                eprintln!("SKIP: single NUMA node — placement is unobservable");
                return;
            }
            let target = topo.nodes.last().unwrap().clone();
            let len = 4 << 20;
            let buf = std::thread::scope(|s| {
                // The spawning thread takes the interleave policy `numactl --interleave=all`
                // would give the whole process; the child inherits it at spawn.
                let mut mask: libc::c_ulong = 0;
                for n in &topo.nodes {
                    mask |= 1 << n.id;
                }
                unsafe { libc::syscall(libc::SYS_set_mempolicy, libc::MPOL_INTERLEAVE, &mask, 65 as libc::c_ulong) };
                let buf = s
                    .spawn(|| {
                        pin_current_thread(&target);
                        vec![1u8; len]
                    })
                    .join()
                    .unwrap();
                unsafe { libc::syscall(libc::SYS_set_mempolicy, libc::MPOL_DEFAULT, std::ptr::null::<libc::c_ulong>(), 0 as libc::c_ulong) };
                buf
            });
            let page = 4096;
            for off in (0..len).step_by(len / 8) {
                let addr = unsafe { buf.as_ptr().add((off / page) * page) };
                assert_eq!(node_of_page(addr), Some(target.id), "page at offset {off} escaped the pinned thread's node — the inherited interleave policy leaked through");
            }
        }
    }

    /// On a real multi-node Linux box: pinning a thread to a node and first-touching an
    /// allocation from it must land the pages on that node (the exact mechanism Phase N3's
    /// preload placement relies on), and `bind_region` must steer pages faulted after the call.
    /// Prints a SKIP line (repo convention for absent-fixture tests) on single-node machines and
    /// non-Linux — this is a placement test, not a parser test, and needs real NUMA to mean
    /// anything.
    #[test]
    fn first_touch_on_a_pinned_thread_and_bind_region_place_pages_on_the_requested_node() {
        let Some(topo) = topology() else {
            eprintln!("SKIP: no NUMA topology on this machine");
            return;
        };
        if topo.n_nodes() < 2 {
            eprintln!("SKIP: single NUMA node — placement is unobservable");
            return;
        }
        let target = topo.nodes.last().unwrap().clone();
        let len = 4 << 20; // 4 MiB: large enough that malloc mmaps it fresh (own VMA, no reused pages)

        // First-touch: fault the pages from a thread pinned to `target`.
        let buf = std::thread::scope(|s| {
            s.spawn(|| {
                pin_current_thread(&target);
                vec![1u8; len]
            })
            .join()
            .unwrap()
        });
        let got = node_of_page(buf.as_ptr());
        assert_eq!(got, Some(target.id), "first-touched pages must sit on the pinned thread's node");

        // bind_region: reserve untouched memory, set the policy, THEN fault from an unpinned
        // thread — the VMA policy, not the faulting CPU, must decide placement.
        let first = &topo.nodes[0];
        let mut reserved: Vec<u8> = Vec::with_capacity(len);
        let ok = unsafe { bind_region(reserved.as_mut_ptr(), len, first.id) };
        assert!(ok, "mbind on a fresh allocation should succeed");
        reserved.resize(len, 1);
        // Probe mid-buffer: the first page can straddle the alignment sliver `bind_region`
        // leaves on the default policy (see `page_aligned`), the middle can't.
        let mid = unsafe { reserved.as_ptr().add(len / 2) };
        assert_eq!(node_of_page(mid), Some(first.id), "pages faulted after bind_region must follow the VMA policy");
    }
}
