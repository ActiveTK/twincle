use anyhow::Result;
use clap::Parser;
use cust::context::Context as CudaContext;
use cust::memory::{CopyDestination, DeviceBuffer}; // <-- CopyDestination を追加
use cust::module::Module;
use cust::prelude::{CudaFlags, Device, Stream, StreamFlags};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::mem::size_of;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, mpsc};
use std::sync::{Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};
use time::{OffsetDateTime, format_description};
use serde_json::json;

#[cfg(ptx_only)]
const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernel.ptx"));
#[cfg(not(ptx_only))]
const FATBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kernel.fatbin"));

/// Twin prime constant C2 (OEIS A005597) — enough precision for f64 usage
const TWIN_PRIME_CONSTANT_C2: f64 = 0.6601618158468695739;
const SMALL_PRIME_MAX: u32 = 1 << 15;
const SMALL_RES_TILE: usize = 128;
const UNIT_ROUND: f64 = std::f64::EPSILON / 2.0;

fn accum_error_bound(sum_abs: f64, n_terms: u64) -> f64 {
    // Higham-style bound for Kahan summation of positive terms:
    // |err| <= (2u + O(nu^2)) * sum_abs.
    // Add a conservative cushion for the (small) non-Kahan reduction steps.
    let n = n_terms as f64;
    let kahan = (2.0 * UNIT_ROUND + 2.0 * n * UNIT_ROUND * UNIT_ROUND) * sum_abs;
    let reduction = 64.0 * UNIT_ROUND * sum_abs;
    kahan + reduction
}

fn gamma_n(n: u32) -> f64 {
    // Higham's gamma_n = (n*u)/(1 - n*u), valid for n*u < 1.
    let nu = (n as f64) * UNIT_ROUND;
    nu / (1.0 - nu)
}

static LOG: OnceLock<Mutex<BufWriter<std::fs::File>>> = OnceLock::new();

fn log_system_info(mp: &MultiProgress) {
    let logical = num_cpus::get();
    let physical = num_cpus::get_physical();
    let core_suffix = format!("[{physical}c{logical}t]");

    #[cfg(target_os = "linux")]
    {
        let mut cpu_name: Option<String> = None;
        let mut total_kb: Option<u64> = None;
        let mut avail_kb: Option<u64> = None;
        if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
            for line in cpuinfo.lines() {
                if let Some((k, v)) = line.split_once(':') {
                    if k.trim() == "model name" {
                        cpu_name = Some(v.trim().to_string());
                        break;
                    }
                }
            }
        }
        if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
            for line in meminfo.lines() {
                if let Some((k, v)) = line.split_once(':') {
                    let val = v
                        .trim()
                        .split_whitespace()
                        .next()
                        .and_then(|x| x.parse::<u64>().ok());
                    if k == "MemTotal" {
                        total_kb = val;
                    } else if k == "MemAvailable" {
                        avail_kb = val;
                    }
                }
            }
        }
        if let Some(name) = cpu_name {
            print_mp_and_log(mp, &format!("CPU: {name} {core_suffix}"));
        } else {
            print_mp_and_log(mp, &format!("CPU: {core_suffix}"));
        }
        if let Some(total) = total_kb {
            let total_gb = total as f64 / (1024.0 * 1024.0);
            let avail_gb = avail_kb.map(|kb| kb as f64 / (1024.0 * 1024.0));
            if let Some(av) = avail_gb {
                print_mp_and_log(
                    mp,
                    &format!("RAM: {:.2} GB ({:.2} GB available)", total_gb, av),
                );
            } else {
                print_mp_and_log(mp, &format!("RAM: {:.2} GB", total_gb));
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        let mut cpu_name: Option<String> = None;
        if let Ok(output) = std::process::Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
        {
            if output.status.success() {
                let name = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !name.is_empty() {
                    cpu_name = Some(name);
                }
            }
        }
        if let Some(name) = cpu_name {
            print_mp_and_log(mp, &format!("CPU: {name} {core_suffix}"));
        } else {
            print_mp_and_log(mp, &format!("CPU: {core_suffix}"));
        }
    }

    #[cfg(target_os = "windows")]
    {
        let mut cpu_name: Option<String> = None;
        if let Ok(output) = std::process::Command::new("powershell")
            .args([
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name",
            ])
            .output()
        {
            if output.status.success() {
                let text = String::from_utf8_lossy(&output.stdout);
                if let Some(line) = text.lines().find(|l| !l.trim().is_empty()) {
                    cpu_name = Some(line.trim().to_string());
                }
            }
        }

        if let Some(name) = cpu_name {
            print_mp_and_log(mp, &format!("CPU: {name} {core_suffix}"));
        } else {
            print_mp_and_log(mp, &format!("CPU: {core_suffix}"));
        }

        if let Ok(output) = std::process::Command::new("powershell")
            .args([
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_OperatingSystem | Select-Object TotalVisibleMemorySize,FreePhysicalMemory | Format-List",
            ])
            .output()
        {
            if output.status.success() {
                let text = String::from_utf8_lossy(&output.stdout);
                let mut total_kb: Option<u64> = None;
                let mut free_kb: Option<u64> = None;
                for line in text.lines() {
                    if let Some((k, v)) = line.split_once(':') {
                        let val = v.trim().parse::<u64>().ok();
                        if k.trim() == "TotalVisibleMemorySize" {
                            total_kb = val;
                        } else if k.trim() == "FreePhysicalMemory" {
                            free_kb = val;
                        }
                    }
                }
                if let Some(total) = total_kb {
                    let total_gb = total as f64 / (1024.0 * 1024.0);
                    let avail_gb = free_kb.map(|kb| kb as f64 / (1024.0 * 1024.0));
                    if let Some(av) = avail_gb {
                        print_mp_and_log(mp, &format!("RAM: {:.2} GB ({:.2} GB available)", total_gb, av));
                    } else {
                        print_mp_and_log(mp, &format!("RAM: {:.2} GB", total_gb));
                    }
                }
            }
        }
    }
}

fn timestamp() -> String {
    let format = format_description::parse("[year]-[month]-[day] [hour]:[minute]:[second]")
        .unwrap_or_else(|_| {
            format_description::parse("[year]-[month]-[day] [hour]:[minute]:[second]").unwrap()
        });
    let now = OffsetDateTime::now_local().unwrap_or_else(|_| OffsetDateTime::now_utc());
    now.format(&format)
        .unwrap_or_else(|_| "0000-00-00 00:00:00".to_string())
}

fn log_line(line: &str) {
    if let Some(log) = LOG.get() {
        if let Ok(mut f) = log.lock() {
            let _ = writeln!(f, "[{}] {}", timestamp(), line);
            let _ = f.flush();
        }
    }
}

fn print_and_log(line: &str) {
    let msg = format!("[{}] {}", timestamp(), line);
    println!("{msg}");
    log_line(line);
}

fn print_mp_and_log(mp: &MultiProgress, line: &str) {
    let msg = format!("[{}] {}", timestamp(), line);
    let _ = mp.println(&msg);
    log_line(line);
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about)]
struct Args {
    /// Upper limit (inclusive) for searching twin primes up to N.
    #[arg(long, default_value = "10000000000000")]
    limit: u64,

    /// If set, uses limit = 10^exp (inclusive).
    #[arg(long)]
    exp: Option<u32>,

    /// Inclusive start of the k-range shard (p = M*k + r).
    #[arg(long)]
    k_start: Option<u64>,

    /// Exclusive end of the k-range shard (p = M*k + r).
    #[arg(long)]
    k_end: Option<u64>,

    /// Wheel modulus: 30 / 210 / 30030
    #[arg(long, default_value = "30030")]
    wheel: u32,

    /// If set, tests all wheel sizes (30, 210, 30030) for 30s each to find the best.
    #[arg(long)]
    test_wheels: bool,

    /// Segment size in number of k values (p = M*k + r).
    /// If 0, auto-pick from VRAM.
    #[arg(long, default_value_t = 0)]
    segment_k: u64,

    /// Fraction of total VRAM to allocate for the bitsets when auto-picking.
    #[arg(long, default_value_t = 0.25)]
    segment_mem_frac: f64,

    /// Auto-tune segment_k by testing a few sizes (adds a short startup cost).
    #[arg(long)]
    auto_tune_seg: bool,

    /// Run a timed benchmark instead of a full search.
    #[arg(long)]
    benchmark: bool,

    /// Benchmark duration in seconds.
    #[arg(long, default_value_t = 10)]
    benchmark_seconds: u64,

    /// Target limit for time-to-solution estimate (used in benchmark report).
    #[arg(long, default_value = "100000000000000000")]
    benchmark_target: u64,
}

/// Runs the search for a given wheel configuration.
/// If `max_duration` is Some, it stops after that duration.
/// Returns (total_twins, total_sum, elapsed_secs, candidates_processed, optional segment results).
fn run_search(
    wheel_m: u32,
    limit: u64,
    k_start: u64,
    k_end_excl: u64,
    segment_k: u64,
    residues: Arc<Vec<u32>>,
    small_primes: Arc<Vec<u32>>,
    small_inv_m: Arc<Vec<u32>>,
    large_primes: Arc<Vec<u32>>,
    large_inv_m: Arc<Vec<u32>>,
    num_gpus: u32,
    mp: &MultiProgress,
    max_duration: Option<Duration>,
    mut exp_log: Option<ExpLog>,
) -> Result<(u64, f64, f64, u64, Option<ExpLog>)> {
    let total_k = k_end_excl.saturating_sub(k_start);
    let total_candidates = (total_k as u128) * (residues.len() as u128);
    let pb = mp.add(ProgressBar::new(
        total_candidates.min(u64::MAX as u128) as u64
    ));
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar:40.cyan/dim}] {percent:>3}% {pos}/{len} | {msg}")?
            .progress_chars("#>-"),
    );
    let msg_prefix = if max_duration.is_some() {
        "testing"
    } else {
        "searching"
    };
    let num_segments = if total_k == 0 {
        0
    } else {
        (total_k + segment_k - 1) / segment_k
    };
    pb.set_message(format!(
        "{} M={} | k=[{}, {}) | seg_k={} | {} segments",
        msg_prefix,
        wheel_m,
        format_with_commas(k_start),
        format_with_commas(k_end_excl),
        format_with_commas(segment_k),
        format_with_commas(num_segments)
    ));
    pb.enable_steady_tick(Duration::from_millis(200));

    let pb = Arc::new(pb);
    let next_k_low = Arc::new(AtomicU64::new(k_start));
    let (tx, rx) = mpsc::channel::<SegResult>();
    let run0 = Instant::now();

    let checkpoint_ks = exp_log.as_ref().map(|l| Arc::new(l.k_floors.clone()));
    let mut handles = Vec::with_capacity(num_gpus as usize);
    for gpu_id in 0..num_gpus {
        let residues = Arc::clone(&residues);
        let small_primes = Arc::clone(&small_primes);
        let small_inv_m_mod_p = Arc::clone(&small_inv_m);
        let large_primes = Arc::clone(&large_primes);
        let large_inv_m_mod_p = Arc::clone(&large_inv_m);
        let next_k_low = Arc::clone(&next_k_low);
        let pb = Arc::clone(&pb);
        let checkpoint_ks = checkpoint_ks.clone();
        let tx = tx.clone();
        let handle = thread::Builder::new()
            .name(format!("gpu-{}", gpu_id))
            .spawn(move || -> Result<()> {
                gpu_worker(
                    gpu_id,
                    wheel_m,
                    residues,
                    small_primes,
                    small_inv_m_mod_p,
                    large_primes,
                    large_inv_m_mod_p,
                    segment_k,
                    k_end_excl,
                    limit,
                    next_k_low,
                    pb,
                    tx,
                    checkpoint_ks,
                )
            })?;
        handles.push(handle);
    }
    drop(tx);

    let mut total = Kahan::default();
    let mut total_twins: u64 = 0;
    let mut segments_done: u64 = 0;
    let mut last_msg = Instant::now();
    let mut ema_speed: f64 = 0.0;
    for r in rx.iter() {
        total.add(r.sum);
        total_twins += r.count;
        segments_done += 1;
        if let Some(ref mut log) = exp_log {
            if let Err(e) = log.on_segment(r) {
                print_mp_and_log(mp, &format!("exp log error: {e}"));
            }
        }

        if let Some(d) = max_duration {
            if run0.elapsed() >= d {
                // Signal termination by exhausting the atomic counter
                next_k_low.store(u64::MAX / 2, Ordering::SeqCst);
            }
        }

        if last_msg.elapsed().as_millis() >= 150 {
            let secs = run0.elapsed().as_secs_f64();
            let done = pb.position();
            let inst_speed = if secs > 0.0 { done as f64 / secs } else { 0.0 };
            ema_speed = if ema_speed == 0.0 {
                inst_speed
            } else {
                0.2 * inst_speed + 0.8 * ema_speed
            };
            let remaining = (total_candidates.min(u64::MAX as u128) as u64).saturating_sub(done);
            let eta_secs = if ema_speed > 0.0 {
                remaining as f64 / ema_speed
            } else {
                0.0
            };

            let msg = format!(
                "seg {}/{} | twins={} | shard_sum≈{:.12} | {:.2e} cand/s | ETA {:.1} min",
                format_with_commas(segments_done),
                format_with_commas(num_segments),
                format_with_commas(total_twins),
                total.value(),
                ema_speed,
                eta_secs / 60.0
            );
            pb.set_message(msg.clone());
            last_msg = Instant::now();
            log_line(&format!("PROGRESS {msg}"));
        }
    }

    for handle in handles {
        handle
            .join()
            .map_err(|e| anyhow::anyhow!("GPU thread panicked: {:?}", e))??;
    }

    let elapsed = run0.elapsed().as_secs_f64();
    let cand_processed = pb.position();

    if max_duration.is_some() {
        pb.finish_and_clear();
    } else {
        pb.finish_with_message(format!("M={} search complete.", wheel_m));
    }

    Ok((total_twins, total.value(), elapsed, cand_processed, exp_log))
}

/// Per-segment result sent back through the channel.
#[derive(Clone)]
struct SegResult {
    k_low: u64,
    k_count: u64,
    sum: f64,
    count: u64,
}

#[derive(Default, Clone, Copy)]
struct Kahan {
    sum: f64,
    c: f64,
}
impl Kahan {
    fn add(&mut self, x: f64) {
        let y = x - self.c;
        let t = self.sum + y;
        self.c = (t - self.sum) - y;
        self.sum = t;
    }
    fn value(self) -> f64 {
        self.sum
    }
}

fn pow10_u64(e: u32) -> Result<u64> {
    if e > 18 {
        anyhow::bail!("--exp too large for u64 (max 18)");
    }
    let mut v: u64 = 1;
    for _ in 0..e {
        v = v.saturating_mul(10);
    }
    Ok(v)
}

fn ceil_sqrt(n: u64) -> u64 {
    let x = (n as f64).sqrt().floor() as u64;
    if x * x >= n { x } else { x + 1 }
}

/// odd-only sieve up to `n` (inclusive), returns odd primes >=3 as u32.
fn sieve_odd_primes_u32(n: u64, pb: Option<&ProgressBar>) -> Vec<u32> {
    if n < 3 {
        return vec![];
    }
    if let Some(p) = pb {
        p.set_length(n);
        p.set_message("Sieving base primes...");
    }
    let max_odd = if n % 2 == 1 { n } else { n - 1 };
    let len = ((max_odd - 3) / 2 + 1) as usize;
    let mut is_comp = vec![false; len];

    let limit = (n as f64).sqrt() as u64;
    let mut p = 3u64;
    while p <= limit {
        let idx = ((p - 3) / 2) as usize;
        if !is_comp[idx] {
            let mut m = p * p;
            let step = 2 * p;
            while m <= max_odd {
                let j = ((m - 3) / 2) as usize;
                is_comp[j] = true;
                m += step;
            }
        }
        if let Some(bar) = pb {
            if p % 1001 == 0 {
                bar.set_position(p);
            }
        }
        p += 2;
    }

    if let Some(p) = pb {
        p.set_position(n);
        p.set_message("Collecting base primes...");
    }
    let mut primes = Vec::new();
    for (i, &c) in is_comp.iter().enumerate() {
        if !c {
            let val = 3u64 + (i as u64) * 2;
            if val <= n {
                primes.push(val as u32);
            }
        }
    }
    primes
}

fn gcd_u32(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// Compute residues r in [0..M) such that:
/// - r is odd
/// - gcd(r, M) == 1
/// - gcd((r+2) mod M, M) == 1
fn wheel_residues(wheel_m: u32, pb: Option<&ProgressBar>) -> Vec<u32> {
    if let Some(p) = pb {
        p.set_length(wheel_m as u64);
        p.set_message("Computing wheel residues...");
    }
    let mut v = Vec::new();
    for r in 0..wheel_m {
        if let Some(p) = pb {
            if r % 100 == 0 {
                p.set_position(r as u64);
            }
        }
        if (r & 1) == 0 {
            continue;
        }
        if gcd_u32(r, wheel_m) != 1 {
            continue;
        }
        let rp2 = (r + 2) % wheel_m;
        if gcd_u32(rp2, wheel_m) != 1 {
            continue;
        }
        v.push(r);
    }
    if let Some(p) = pb {
        p.set_position(wheel_m as u64);
    }
    v
}

/// Extended Euclid to compute modular inverse a^{-1} mod m (a,m coprime).
fn modinv_u32(a: u32, m: u32) -> u32 {
    let (mut t, mut new_t) = (0i64, 1i64);
    let (mut r, mut new_r) = (m as i64, a as i64);
    while new_r != 0 {
        let q = r / new_r;
        (t, new_t) = (new_t, t - q * new_t);
        (r, new_r) = (new_r, r - q * new_r);
    }
    if r != 1 {
        panic!("modinv does not exist");
    }
    if t < 0 {
        t += m as i64;
    }
    t as u32
}

fn is_prime_small(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n < 4 {
        return true;
    }
    if n % 2 == 0 || n % 3 == 0 {
        return false;
    }
    let mut d = 5u64;
    while d * d <= n {
        if n % d == 0 || n % (d + 2) == 0 {
            return false;
        }
        d += 6;
    }
    true
}

fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// Count twin prime pairs missed by the wheel sieve.
/// The wheel only considers candidates p where gcd(p,M)==1 and gcd(p+2,M)==1,
/// so twin pairs involving prime factors of M are missed.
fn count_wheel_missed_twins(wheel_m: u32, limit: u64) -> (u64, f64) {
    // A twin prime (p, p+2) can only be "missed" if p or p+2 shares a factor with M.
    // For true twin primes, that implies p or p+2 equals a prime factor of M, so
    // all missed pairs must satisfy p <= M. We brute-force this small range to avoid
    // logic gaps and keep correctness obvious.
    let mut count = 0u64;
    let mut sum = 0.0f64;
    let max_p = std::cmp::min(limit.saturating_sub(2), wheel_m as u64);
    for p in 3..=max_p {
        if !is_prime_small(p) || !is_prime_small(p + 2) {
            continue;
        }
        let pc = gcd_u64(p, wheel_m as u64) == 1;
        let p2c = gcd_u64(p + 2, wheel_m as u64) == 1;
        if !(pc && p2c) {
            count += 1;
            sum += 1.0 / (p as f64) + 1.0 / ((p + 2) as f64);
        }
    }

    (count, sum)
}

fn git_sha_short() -> Option<String> {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if s.is_empty() { None } else { Some(s) }
}

fn exp_checkpoint_limits(exp: u32) -> Vec<u64> {
    if exp <= 4 {
        return Vec::new();
    }
    // Ensure 10^exp == 1000 * 10^(exp-3), so k ranges 1..=1000.
    let step = match pow10_u64(exp - 3) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    let steps = 1000u64;
    if step == 0 {
        return Vec::new();
    }
    let mut v = Vec::with_capacity(steps as usize);
    for k in 1..=steps {
        v.push(k.saturating_mul(step));
    }
    v
}

fn reserve_k_range(
    next_k_low: &AtomicU64,
    segment_k: u64,
    k_end_excl: u64,
    checkpoint_ks: Option<&[u64]>,
) -> Option<(u64, u64)> {
    loop {
        let k_low = next_k_low.load(Ordering::Relaxed);
        if k_low >= k_end_excl {
            return None;
        }
        let mut k_high = k_low.saturating_add(segment_k).min(k_end_excl);
        if let Some(cks) = checkpoint_ks {
            if let Some(boundary) = cks.iter().copied().find(|&b| b > k_low) {
                if boundary < k_high {
                    k_high = boundary;
                }
            }
        }
        if next_k_low
            .compare_exchange(k_low, k_high, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            let k_count = k_high.saturating_sub(k_low);
            if k_count == 0 {
                return None;
            }
            return Some((k_low, k_count));
        }
    }
}

fn mod_mul_u64(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % (m as u128)) as u64
}

fn mod_pow_u64(mut a: u64, mut d: u64, m: u64) -> u64 {
    let mut r = 1u64;
    while d > 0 {
        if d & 1 == 1 {
            r = mod_mul_u64(r, a, m);
        }
        a = mod_mul_u64(a, a, m);
        d >>= 1;
    }
    r
}

// Deterministic Miller-Rabin for u64 using a fixed base set.
fn is_prime_u64(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    // Deterministic for 64-bit range with this base set.
    const BASES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    for &p in &BASES {
        if n == p {
            return true;
        }
        if n % p == 0 {
            return n == p;
        }
    }
    let mut d = n - 1;
    let mut s = 0u32;
    while d & 1 == 0 {
        d >>= 1;
        s += 1;
    }
    for &a in &BASES {
        if a % n == 0 {
            continue;
        }
        let mut x = mod_pow_u64(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        let mut witness = true;
        for _ in 1..s {
            x = mod_mul_u64(x, x, n);
            if x == n - 1 {
                witness = false;
                break;
            }
        }
        if witness {
            return false;
        }
    }
    true
}

struct ExpLog {
    writer: BufWriter<std::fs::File>,
    wheel_m: u32,
    part_mode: bool,
    k_start: u64,
    k_end_excl: u64,
    limits: Vec<u64>,
    k_floors: Vec<u64>,
    remainders: Vec<u64>,
    next_idx: usize,
    pending: std::collections::BTreeMap<u64, SegResult>,
    next_k_low: u64,
    sum: Kahan,
    count: u64,
    residues: Arc<Vec<u32>>,
    checkpoints: std::collections::BTreeMap<u64, ExpCheckpoint>,
}

#[derive(Clone, Copy)]
#[allow(dead_code)]
struct ExpCheckpoint {
    twins: u64,
    sum: f64,
    accum_err_bound: f64,
    term_eval_err_bound: f64,
}

impl ExpLog {
    fn new(
        path: &str,
        args: &Args,
        limit: u64,
        wheel_m: u32,
        segment_k: u64,
        residues: Arc<Vec<u32>>,
        gpu_names: &[String],
    ) -> Result<Option<Self>> {
        if args.benchmark {
            return Ok(None);
        }
        let k_start = args.k_start.unwrap_or(0);
        let k_end_excl = args.k_end.unwrap_or((limit / (wheel_m as u64)) + 2);
        let part_mode = args.k_start.is_some() || args.k_end.is_some();
        if !part_mode && args.exp.is_none() {
            return Ok(None);
        }
        let limits = match args.exp {
            Some(exp) => exp_checkpoint_limits(exp)
                .into_iter()
                .filter(|&lim| {
                    let k_floor = lim / (wheel_m as u64);
                    k_floor >= k_start && k_floor < k_end_excl
                })
                .collect::<Vec<_>>(),
            None => Vec::new(),
        };
        let mut k_floors = Vec::with_capacity(limits.len());
        let mut remainders = Vec::with_capacity(limits.len());
        for &lim in &limits {
            k_floors.push(lim / (wheel_m as u64));
            remainders.push(lim % (wheel_m as u64));
        }

        let mut writer = BufWriter::new(
            OpenOptions::new().create(true).write(true).truncate(true).open(path)?
        );
        let git_sha = git_sha_short().unwrap_or_else(|| "unknown".to_string());
        let meta = json!({
            "type": "meta",
            "format": "jsonl",
            "timestamp": timestamp(),
            "git_sha": git_sha,
            "cmdline": std::env::args().collect::<Vec<_>>().join(" "),
            "exp": args.exp,
            "limit": limit,
            "wheel_m": wheel_m,
            "part_mode": part_mode,
            "k_start": k_start,
            "k_end": k_end_excl,
            "segment_k": segment_k,
            "segment_mem_frac": args.segment_mem_frac,
            "auto_tune_seg": args.auto_tune_seg,
            "residues_len": residues.len(),
            "gpu_names": gpu_names,
            "note": if part_mode {
                "checkpoint sums are shard-local partial sums from k_start up to limit_requested within the shard; final excludes wheel-missed correction."
            } else {
                "checkpoint sums are exact for limit_requested, computed as full k<k_floor plus partial k_floor by residue filter; error bounds are IEEE-754 gamma_n upper bounds."
            }
        });
        writeln!(writer, "{}", meta)?;

        Ok(Some(Self {
            writer,
            wheel_m,
            part_mode,
            k_start,
            k_end_excl,
            limits,
            k_floors,
            remainders,
            next_idx: 0,
            pending: std::collections::BTreeMap::new(),
            next_k_low: k_start,
            sum: Kahan::default(),
            count: 0,
            residues,
            checkpoints: std::collections::BTreeMap::new(),
        }))
    }

    fn on_segment(&mut self, seg: SegResult) -> Result<()> {
        self.pending.insert(seg.k_low, seg);
        while let Some(seg) = self.pending.remove(&self.next_k_low) {
            self.sum.add(seg.sum);
            self.count += seg.count;
            self.next_k_low = self.next_k_low.saturating_add(seg.k_count);

            while self.next_idx < self.k_floors.len()
                && self.next_k_low >= self.k_floors[self.next_idx]
            {
                let k_floor = self.k_floors[self.next_idx];
                let rem = self.remainders[self.next_idx];
                let mut partial_sum = Kahan::default();
                let mut partial_count = 0u64;
                if k_floor > 0 {
                    let base = (self.wheel_m as u64).saturating_mul(k_floor);
                    for &r in self.residues.iter() {
                        let r_u = r as u64;
                        if r_u > rem {
                            break;
                        }
                        let p = base + r_u;
                        if p < 3 {
                            continue;
                        }
                        if is_prime_u64(p) && is_prime_u64(p + 2) {
                            partial_count += 1;
                            partial_sum.add(1.0 / (p as f64) + 1.0 / ((p + 2) as f64));
                        }
                    }
                }

                let total_count = self.count + partial_count;
                let total_sum = self.sum.value() + partial_sum.value();
                let accum_err = accum_error_bound(total_sum, total_count);
                let term_eval_err = gamma_n(5) * total_sum;
                let total_err = accum_err + term_eval_err;
                self.checkpoints.insert(
                    self.limits[self.next_idx],
                    ExpCheckpoint {
                        twins: total_count,
                        sum: total_sum,
                        accum_err_bound: accum_err,
                        term_eval_err_bound: term_eval_err,
                    },
                );
                let rec = json!({
                    "type": "checkpoint",
                    "limit_requested": self.limits[self.next_idx],
                    "k_start": self.k_start,
                    "k_end": self.k_end_excl,
                    "k_floor": k_floor,
                    "limit_covered": self.limits[self.next_idx],
                    "twins": total_count,
                    "sum": total_sum,
                    "sum_kahan_c": self.sum.c,
                    "partial_sum_kahan_c": partial_sum.c,
                    "accum_err_bound": accum_err,
                    "term_eval_err_bound": term_eval_err,
                    "total_err_bound": total_err
                });
                writeln!(self.writer, "{}", rec)?;
                self.next_idx += 1;
            }
        }
        Ok(())
    }

    fn write_final(
        &mut self,
        limit: u64,
        final_twins: u64,
        final_sum: f64,
        b2_star: f64,
        accum_err_bound: f64,
        term_eval_err_bound: f64,
        elapsed_secs: f64,
        missed_count: u64,
        missed_sum: f64,
    ) -> Result<()> {
        let total_err_bound = accum_err_bound + term_eval_err_bound;
        let rec = json!({
            "type": "final",
            "timestamp": timestamp(),
            "part_mode": self.part_mode,
            "limit": limit,
            "k_start": self.k_start,
            "k_end": self.k_end_excl,
            "twins": final_twins,
            "sum": final_sum,
            "b2_star": b2_star,
            "accum_err_bound": accum_err_bound,
            "term_eval_err_bound": term_eval_err_bound,
            "total_err_bound": total_err_bound,
            "elapsed_secs": elapsed_secs,
            "missed_count": missed_count,
            "missed_sum": missed_sum
        });
        writeln!(self.writer, "{}", rec)?;
        self.writer.flush()?;
        Ok(())
    }
}

fn shard_log_path(args: &Args) -> Option<String> {
    match (args.k_start, args.k_end) {
        (Some(k_start), Some(k_end)) => Some(format!("result_part[{k_start}-{k_end}].json")),
        _ => None,
    }
}

fn format_with_commas(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            result.push(',');
        }
        result.push(ch);
    }
    result
}

fn pick_segment_k(total_mem_bytes: usize, frac: f64, residues_len: usize) -> u64 {
    let mut f = frac;
    if !f.is_finite() {
        f = 0.25;
    }
    f = f.clamp(0.01, 0.50);

    // We allocate two bitsets: comp_p and comp_p2, 1 bit per candidate.
    // bytes(bitsets) ≈ 2 * candidates/8 = candidates/4 = (k * R) / 4
    let budget = (total_mem_bytes as f64 * f) as u64;
    let r = residues_len as u64;
    if r == 0 {
        return 1 << 20;
    }

    let max_k = segment_k_max(total_mem_bytes);
    let k = (budget.saturating_mul(4) / r).clamp(1 << 14, max_k);

    k
}

fn segment_k_max(total_mem_bytes: usize) -> u64 {
    if total_mem_bytes >= (8u64 << 30) as usize {
        1 << 23
    } else {
        1 << 22
    }
}

fn auto_tune_segment_k(
    wheel_m: u32,
    limit: u64,
    residues: Arc<Vec<u32>>,
    small_primes: Arc<Vec<u32>>,
    small_inv_m: Arc<Vec<u32>>,
    large_primes: Arc<Vec<u32>>,
    large_inv_m: Arc<Vec<u32>>,
    num_gpus: u32,
    min_vram: usize,
    args: &Args,
    mp: &MultiProgress,
) -> Result<u64> {
    let base = pick_segment_k(min_vram, args.segment_mem_frac, residues.len());
    let min_k = 1 << 14;
    let max_k = segment_k_max(min_vram);

    let mut base_pow2 = base.next_power_of_two();
    if base_pow2 < min_k {
        base_pow2 = min_k;
    } else if base_pow2 > max_k {
        base_pow2 = max_k;
    }

    let mut cands = Vec::new();
    if base_pow2 >= min_k {
        cands.push(base_pow2);
    }
    if base_pow2 / 2 >= min_k {
        cands.push(base_pow2 / 2);
    }
    if base_pow2 * 2 <= max_k {
        cands.push(base_pow2 * 2);
    }
    cands.sort_unstable();
    cands.dedup();

    let mut best_k = base_pow2;
    let mut best_speed = -1.0;

    for &k in &cands {
        let (_twins, _sum, elapsed, cand, _exp_log) = run_search(
            wheel_m,
            limit,
            0,
            (limit / (wheel_m as u64)) + 2,
            k,
            Arc::clone(&residues),
            Arc::clone(&small_primes),
            Arc::clone(&small_inv_m),
            Arc::clone(&large_primes),
            Arc::clone(&large_inv_m),
            num_gpus,
            mp,
            Some(Duration::from_secs(2)),
            None,
        )?;
        let speed = if elapsed > 0.0 { cand as f64 / elapsed } else { 0.0 };
        let msg = format!("Auto-tune seg_k={} => {:.2e} cand/s", format_with_commas(k), speed);
        print_mp_and_log(mp, &msg);
        if speed > best_speed {
            best_speed = speed;
            best_k = k;
        }
    }

    let msg = format!(
        "Auto-tune chose seg_k={} ({:.2e} cand/s)",
        format_with_commas(best_k),
        best_speed
    );
    print_mp_and_log(mp, &msg);
    Ok(best_k)
}

fn estimate_seconds_for_limit(
    limit: u64,
    wheel_m: u32,
    residues_len: usize,
    cand_per_sec: f64,
) -> Option<f64> {
    if cand_per_sec <= 0.0 || residues_len == 0 {
        return None;
    }
    let k_end_excl = (limit / (wheel_m as u64)) + 2;
    let total_candidates = (k_end_excl as u128) * (residues_len as u128);
    let total_candidates_f = total_candidates as f64;
    Some(total_candidates_f / cand_per_sec)
}

fn gpu_worker(
    device_id: u32,
    wheel_m: u32,
    residues: Arc<Vec<u32>>,
    small_primes: Arc<Vec<u32>>,
    small_inv_m_mod_p: Arc<Vec<u32>>,
    large_primes: Arc<Vec<u32>>,
    large_inv_m_mod_p: Arc<Vec<u32>>,
    segment_k: u64,
    k_end_excl: u64,
    limit: u64,
    next_k_low: Arc<AtomicU64>,
    pb: Arc<ProgressBar>,
    tx: mpsc::Sender<SegResult>,
    checkpoint_ks: Option<Arc<Vec<u64>>>,
) -> Result<()> {
    let device = Device::get_device(device_id)?;
    let _ctx = CudaContext::new(device)?;
    #[cfg(ptx_only)]
    let module = Module::from_ptx(PTX, &[])?;
    #[cfg(not(ptx_only))]
    let module = Module::from_fatbin(FATBIN, &[])?;

    let clear_kernel = module.get_function("clear_buffers")?;
    let sieve_small_kernel = module.get_function("sieve_wheel_primes_small")?;
    let sieve_kernel = module.get_function("sieve_wheel_primes")?;
    let twin_kernel = module.get_function("twin_sum_wheel")?;
    let reduce_kernel = module.get_function("reduce_block_results")?;

    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let res_len = residues.len();
    let small_prime_count = small_primes.len();
    let large_prime_count = large_primes.len();
    let sieve_shared = (res_len * size_of::<u32>()) as u32;
    let sieve_small_shared = (SMALL_RES_TILE * size_of::<u32>()) as u32;

    let d_residues = DeviceBuffer::from_slice(&residues)?;
    let d_small_primes = DeviceBuffer::from_slice(&small_primes)?;
    let d_small_inv = DeviceBuffer::from_slice(&small_inv_m_mod_p)?;
    let d_large_primes = DeviceBuffer::from_slice(&large_primes)?;
    let d_large_inv = DeviceBuffer::from_slice(&large_inv_m_mod_p)?;

    // bitsets: comp_p and comp_p2
    let candidates_per_seg = segment_k.saturating_mul(res_len as u64);
    let words = ((candidates_per_seg + 63) / 64) as usize;

    let d_comp_p = DeviceBuffer::<u64>::zeroed(words)?;
    let d_comp_p2 = DeviceBuffer::<u64>::zeroed(words)?;

    // kernel launch config — grid sized to the actual workload
    let block = 256u32;
    let sieve_pairs = large_prime_count as u64 * res_len as u64;
    let sieve_grid = ((sieve_pairs + block as u64 - 1) / block as u64).min(262144) as u32;
    let twin_grid = ((words as u64 + block as u64 - 1) / block as u64).min(262144) as u32;
    let clear_grid = ((words as u64 + block as u64 - 1) / block as u64).min(65535) as u32;
    let sieve_small_grid = ((res_len + SMALL_RES_TILE - 1) / SMALL_RES_TILE) as u32;

    let d_block_sums = DeviceBuffer::<f64>::zeroed(twin_grid as usize)?;
    let d_block_counts = DeviceBuffer::<u64>::zeroed(twin_grid as usize)?;
    let d_final_sum = DeviceBuffer::<f64>::zeroed(1)?;
    let d_final_count = DeviceBuffer::<u64>::zeroed(1)?;
    let mut h_final_sum = vec![0f64; 1];
    let mut h_final_count = vec![0u64; 1];

    loop {
        let (k_low, k_count) = match reserve_k_range(
            &next_k_low,
            segment_k,
            k_end_excl,
            checkpoint_ks.as_deref().map(|v| v.as_slice()),
        ) {
            Some(v) => v,
            None => break,
        };

        // Stream-ordered clear (avoids default-stream serialisation)
        unsafe {
            cust::launch!(
                clear_kernel<<<clear_grid, block, 0, stream>>>(
                    d_comp_p.as_device_ptr(),
                    d_comp_p2.as_device_ptr(),
                    words as u32
                )
            )?;
        }

        if small_prime_count > 0 {
            unsafe {
                cust::launch!(
                    sieve_small_kernel<<<sieve_small_grid, block, sieve_small_shared, stream>>>(
                        k_low as u64,
                        k_count as u64,
                        wheel_m as u32,
                        d_residues.as_device_ptr(),
                        res_len as i32,
                        d_small_primes.as_device_ptr(),
                        d_small_inv.as_device_ptr(),
                        small_prime_count as i32,
                        d_comp_p.as_device_ptr(),
                        d_comp_p2.as_device_ptr()
                    )
                )?;
            }
        }

        if large_prime_count > 0 {
            unsafe {
                cust::launch!(
                    sieve_kernel<<<sieve_grid, block, sieve_shared, stream>>>(
                        k_low as u64,
                        k_count as u64,
                        wheel_m as u32,
                        d_residues.as_device_ptr(),
                        res_len as i32,
                        d_large_primes.as_device_ptr(),
                        d_large_inv.as_device_ptr(),
                        large_prime_count as i32,
                        d_comp_p.as_device_ptr(),
                        d_comp_p2.as_device_ptr()
                    )
                )?;
            }
        }

        let (k_mask, k_shift) = if k_count.is_power_of_two() {
            (k_count - 1, k_count.trailing_zeros())
        } else {
            (0, 0)
        };

        unsafe {
            cust::launch!(
                twin_kernel<<<twin_grid, block, 0, stream>>>(
                    k_low as u64,
                    k_count as u64,
                    k_mask,
                    k_shift,
                    wheel_m as u32,
                    d_residues.as_device_ptr(),
                    res_len as i32,
                    limit as u64,
                    d_comp_p.as_device_ptr(),
                    d_comp_p2.as_device_ptr(),
                    d_block_sums.as_device_ptr(),
                    d_block_counts.as_device_ptr()
                )
            )?;
        }

        unsafe {
            cust::launch!(
                reduce_kernel<<<1, block, 0, stream>>>(
                    d_block_sums.as_device_ptr(),
                    d_block_counts.as_device_ptr(),
                    twin_grid as u32,
                    d_final_sum.as_device_ptr(),
                    d_final_count.as_device_ptr()
                )
            )?;
        }

        stream.synchronize()?;

        d_final_sum.copy_to(&mut h_final_sum)?;
        d_final_count.copy_to(&mut h_final_count)?;

        let mut seg_sum = Kahan::default();
        seg_sum.add(h_final_sum[0]);
        let seg_count: u64 = h_final_count[0];

        if tx
            .send(SegResult {
                k_low,
                k_count,
                sum: seg_sum.value(),
                count: seg_count,
            })
            .is_err()
        {
            break;
        }

        pb.inc((k_count as u64) * (res_len as u64));
    }

    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    let limit = if let Some(e) = args.exp {
        pow10_u64(e)?
    } else {
        args.limit
    };
    if limit < 5 {
        anyhow::bail!("limit must be >= 5");
    }
    if args.benchmark && args.test_wheels {
        anyhow::bail!("--benchmark and --test-wheels cannot be used together");
    }
    match (args.k_start, args.k_end) {
        (Some(k_start), Some(k_end)) => {
            if k_start >= k_end {
                anyhow::bail!("--k-start must be smaller than --k-end");
            }
            if args.test_wheels {
                anyhow::bail!("--test-wheels cannot be used with --k-start/--k-end");
            }
        }
        (None, None) => {}
        _ => anyhow::bail!("--k-start and --k-end must be specified together"),
    }

    let mp = MultiProgress::new();
    let _ = LOG.set(Mutex::new(BufWriter::new(
        OpenOptions::new()
            .create(true)
            .append(true)
            .open("run.log")?,
    )));
    let cmdline = std::env::args().collect::<Vec<_>>().join(" ");
    log_line("**********************************************************************");
    log_line(&format!("** Started: {cmdline}"));
    log_line("**********************************************************************");
    log_system_info(&mp);

    cust::init(CudaFlags::empty())?;
    let num_gpus = Device::num_devices()? as u32;
    if num_gpus == 0 {
        anyhow::bail!("No CUDA devices found");
    }

    let mut min_vram: usize = usize::MAX;
    let mut gpu_names: Vec<String> = Vec::new();
    for i in 0..num_gpus {
        let dev = Device::get_device(i)?;
        let mem = dev.total_memory()? as usize;
        let name = dev.name()?;
        gpu_names.push(name.clone());
        let msg = format!(
            "GPU {}: {} ({:.1} GB VRAM)",
            i,
            name,
            mem as f64 / (1u64 << 30) as f64
        );
        print_mp_and_log(&mp, &msg);
        min_vram = min_vram.min(mem);
    }

    let base_limit = ceil_sqrt(limit);

    let wheels_to_test = if args.test_wheels {
        let primes = [2, 3, 5, 7, 11, 13, 17];
        let mut v = vec![2]; // M=2 (parity only)
        let mut m = 2;
        for &p in &primes[1..] {
            m *= p;
            v.push(m);
        }
        v
    } else {
        vec![args.wheel]
    };

    let mut best_wheel = wheels_to_test[0];
    let mut best_speed = -1.0;

    for &w in &wheels_to_test {
        let pb_pre = mp.add(ProgressBar::new(0));
        pb_pre.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] [{bar:40.green/dim}] {pos}/{len} | {msg}")?
                .progress_chars("#>-"),
        );

        let residues = wheel_residues(w, Some(&pb_pre));
        if residues.is_empty() {
            pb_pre.finish_and_clear();
            let msg = format!("Skipping M={}: no valid twin residues", w);
            print_mp_and_log(&mp, &msg);
            continue;
        }
        let mut base_primes = sieve_odd_primes_u32(base_limit, Some(&pb_pre));
        pb_pre.finish_and_clear();

        let msg = format!(
            "Testing M={}: {} residues, {} base primes",
            w,
            residues.len(),
            base_primes.len()
        );
        print_mp_and_log(&mp, &msg);

        base_primes.retain(|&p| (w as u64) % (p as u64) != 0);
        let mut small_primes = Vec::new();
        let mut small_inv = Vec::new();
        let mut large_primes = Vec::new();
        let mut large_inv = Vec::new();
        for &p in &base_primes {
            let invp = modinv_u32(w % p, p);
            if p <= SMALL_PRIME_MAX {
                small_primes.push(p);
                small_inv.push(invp);
            } else {
                large_primes.push(p);
                large_inv.push(invp);
            }
        }

        let residues = Arc::new(residues);
        let small_primes = Arc::new(small_primes);
        let small_inv_m = Arc::new(small_inv);
        let large_primes = Arc::new(large_primes);
        let large_inv_m = Arc::new(large_inv);
        let residues_len = residues.len();

        let duration = if args.test_wheels {
            Some(Duration::from_secs(30))
        } else {
            None
        };

        let segment_k = if args.segment_k == 0 {
            if args.auto_tune_seg {
                auto_tune_segment_k(
                    w,
                    limit,
                    Arc::clone(&residues),
                    Arc::clone(&small_primes),
                    Arc::clone(&small_inv_m),
                    Arc::clone(&large_primes),
                    Arc::clone(&large_inv_m),
                    num_gpus,
                    min_vram,
                    &args,
                    &mp,
                )?
            } else {
                pick_segment_k(min_vram, args.segment_mem_frac, residues_len)
            }
        } else {
            args.segment_k
        };

        if args.benchmark {
            let duration = Duration::from_secs(args.benchmark_seconds.max(1));
            let (_twins, _sum, elapsed, cand, _exp_log) = run_search(
                w,
                limit,
                0,
                (limit / (w as u64)) + 2,
                segment_k,
                residues,
                small_primes,
                small_inv_m,
                large_primes,
                large_inv_m,
                num_gpus,
                &mp,
                Some(duration),
                None,
            )?;

            let cand_per_sec = if elapsed > 0.0 {
                cand as f64 / elapsed
            } else {
                0.0
            };
            let est =
                estimate_seconds_for_limit(args.benchmark_target, w, residues_len, cand_per_sec);

            let msg = format!(
                "Benchmark: M={} | {:.2e} cand/s | {:.2}s elapsed",
                w, cand_per_sec, elapsed
            );
            print_and_log(&msg);
            if let Some(sec) = est {
                let days = sec / 86400.0;
                let msg = format!(
                    "Estimate to reach limit {}: {:.2} days (≈{:.2e} s)",
                    format_with_commas(args.benchmark_target),
                    days,
                    sec
                );
                print_and_log(&msg);
            }
            return Ok(());
        }

        if !args.test_wheels {
            // If not testing, just run the final report directly and return.
            return final_report(
                w,
                limit,
                segment_k,
                residues,
                small_primes,
                small_inv_m,
                large_primes,
                large_inv_m,
                num_gpus,
                &mp,
                &args,
                gpu_names.clone(),
            );
        }

            let (_twins, _sum, elapsed, cand, _exp_log) = run_search(
            w,
            limit,
            0,
            (limit / (w as u64)) + 2,
            segment_k,
            residues,
            small_primes,
            small_inv_m,
            large_primes,
            large_inv_m,
            num_gpus,
            &mp,
            duration,
            None,
        )?;

        let speed = cand as f64 / elapsed;
        let msg = format!("  M={w}: {:.2e} cand/s", speed);
        print_mp_and_log(&mp, &msg);
        if speed > best_speed {
            best_speed = speed;
            best_wheel = w;
        }
    }

    if args.test_wheels {
        let msg = format!(
            "Best wheel found: M={best_wheel} ({:.2e} cand/s)",
            best_speed
        );
        print_mp_and_log(&mp, &msg);
        let msg = "Starting final search...".to_string();
        print_mp_and_log(&mp, &msg);

        let pb_pre = mp.add(ProgressBar::new(0));
        pb_pre.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] [{bar:40.green/dim}] {pos}/{len} | {msg}")?
                .progress_chars("#>-"),
        );
        let residues = wheel_residues(best_wheel, Some(&pb_pre));
        let mut base_primes = sieve_odd_primes_u32(base_limit, Some(&pb_pre));
        pb_pre.finish_and_clear();

        base_primes.retain(|&p| (best_wheel as u64) % (p as u64) != 0);
        let mut small_primes = Vec::new();
        let mut small_inv = Vec::new();
        let mut large_primes = Vec::new();
        let mut large_inv = Vec::new();
        for &p in &base_primes {
            let invp = modinv_u32(best_wheel % p, p);
            if p <= SMALL_PRIME_MAX {
                small_primes.push(p);
                small_inv.push(invp);
            } else {
                large_primes.push(p);
                large_inv.push(invp);
            }
        }

        let residues = Arc::new(residues);
        let small_primes = Arc::new(small_primes);
        let small_inv_m = Arc::new(small_inv);
        let large_primes = Arc::new(large_primes);
        let large_inv_m = Arc::new(large_inv);
        let residues_len = residues.len();

        let segment_k = if args.segment_k == 0 {
            if args.auto_tune_seg {
                auto_tune_segment_k(
                    best_wheel,
                    limit,
                    Arc::clone(&residues),
                    Arc::clone(&small_primes),
                    Arc::clone(&small_inv_m),
                    Arc::clone(&large_primes),
                    Arc::clone(&large_inv_m),
                    num_gpus,
                    min_vram,
                    &args,
                    &mp,
                )?
            } else {
                pick_segment_k(min_vram, args.segment_mem_frac, residues_len)
            }
        } else {
            args.segment_k
        };

        final_report(
            best_wheel,
            limit,
            segment_k,
            residues,
            small_primes,
            small_inv_m,
            large_primes,
            large_inv_m,
            num_gpus,
            &mp,
            &args,
            gpu_names.clone(),
        )?;
    }

    Ok(())
}

fn final_report(
    wheel_m: u32,
    limit: u64,
    segment_k: u64,
    residues: Arc<Vec<u32>>,
    small_primes: Arc<Vec<u32>>,
    small_inv_m: Arc<Vec<u32>>,
    large_primes: Arc<Vec<u32>>,
    large_inv_m: Arc<Vec<u32>>,
    num_gpus: u32,
    mp: &MultiProgress,
    args: &Args,
    gpu_names: Vec<String>,
) -> Result<()> {
    let k_start = args.k_start.unwrap_or(0);
    let full_k_end_excl = (limit / (wheel_m as u64)) + 2;
    let k_end_excl = args.k_end.unwrap_or(full_k_end_excl);
    let part_mode = args.k_start.is_some() || args.k_end.is_some();
    if k_end_excl > full_k_end_excl {
        anyhow::bail!(
            "--k-end={} exceeds max k_end={} for limit={} and wheel={}",
            k_end_excl,
            full_k_end_excl,
            limit,
            wheel_m
        );
    }

    let msg = if part_mode {
        format!(
            "Running shard search with M={wheel_m}, k=[{}, {})...",
            format_with_commas(k_start),
            format_with_commas(k_end_excl)
        )
    } else {
        format!("Running final search with M={wheel_m}...")
    };
    print_mp_and_log(mp, &msg);
    let (missed_count, missed_sum) = if part_mode {
        (0, 0.0)
    } else {
        count_wheel_missed_twins(wheel_m, limit)
    };
    if !part_mode && missed_count > 0 {
        let msg = format!(
            "Wheel-missed small twin pairs: {} (sum contribution: {:.17})",
            missed_count, missed_sum
        );
        print_mp_and_log(mp, &msg);
    }

    let log_path = shard_log_path(args)
        .unwrap_or_else(|| format!("exp_log_e{}.jsonl", args.exp.unwrap_or(0)));
    let exp_log = ExpLog::new(
        &log_path,
        args,
        limit,
        wheel_m,
        segment_k,
        Arc::clone(&residues),
        &gpu_names,
    )?;

    let (total_twins, total_sum, elapsed, _cand, mut exp_log) = run_search(
        wheel_m,
        limit,
        k_start,
        k_end_excl,
        segment_k,
        residues,
        small_primes,
        small_inv_m,
        large_primes,
        large_inv_m,
        num_gpus,
        mp,
        None,
        exp_log,
    )?;

    let final_twins = total_twins + missed_count;
    let final_sum = total_sum + missed_sum;

    let ln = (limit as f64).ln();
    let b2_star = final_sum + 4.0 * TWIN_PRIME_CONSTANT_C2 / ln;
    let accum_err_bound = accum_error_bound(final_sum, final_twins);
    // Per-term evaluation: p->f64, division, p+2->f64, division, and term add.
    // Use a rigorous gamma_5 bound for IEEE-754 round-to-nearest.
    let term_eval_err_bound = gamma_n(5) * final_sum;
    if let Some(ref mut log) = exp_log {
        log.write_final(
            limit,
            final_twins,
            final_sum,
            b2_star,
            accum_err_bound,
            term_eval_err_bound,
            elapsed,
            missed_count,
            missed_sum,
        )?;
    }

    let msg = format!(
        "Done. gpus={}, twins={}",
        num_gpus,
        format_with_commas(final_twins)
    );
    print_and_log(&msg);
    if !part_mode && let (Some(exp), Some(log)) = (args.exp, exp_log.as_ref()) {
        if exp >= 4 {
            let checkpoints = [exp - 3, exp - 2, exp - 1];
            for ce in checkpoints {
                if let Ok(lim) = pow10_u64(ce) {
                    if let Some(rec) = log.checkpoints.get(&lim) {
                        let (_missed_cnt, missed_sum) = count_wheel_missed_twins(wheel_m, lim);
                        let adj_sum = rec.sum + missed_sum;
                        let msg = format!(
                            "Brun partial sum up to {lim}: {:.17}",
                            adj_sum
                        );
                        print_and_log(&msg);
                    }
                }
            }
        }
    }
    let msg = if part_mode {
        format!(
            "Shard partial sum for k=[{}, {}): {:.17}",
            format_with_commas(k_start),
            format_with_commas(k_end_excl),
            final_sum
        )
    } else {
        format!("Brun partial sum up to {limit}: {:.17}", final_sum)
    };
    print_and_log(&msg);
    let msg = format!("Brun extrapolated  B2* (HL):    {:.17}", b2_star);
    print_and_log(&msg);
    let msg = format!(
        "Accumulation error bound (Kahan + reductions, excludes per-term rounding): <= {:.3e}",
        accum_err_bound
    );
    print_and_log(&msg);
    let msg = format!(
        "Per-term eval error bound (IEEE-754, gamma_5): <= {:.3e}",
        term_eval_err_bound
    );
    print_and_log(&msg);
    let msg = format!("Elapsed: {:.2}s", elapsed);
    print_and_log(&msg);
    Ok(())
}
