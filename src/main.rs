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
use std::sync::{Mutex, OnceLock};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::{Duration, Instant};
use time::{OffsetDateTime, format_description};

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernel.ptx"));

/// Twin prime constant C2 (OEIS A005597) — enough precision for f64 usage
const TWIN_PRIME_CONSTANT_C2: f64 = 0.6601618158468695739;
const SMALL_PRIME_MAX: u32 = 1 << 15;

static LOG: OnceLock<Mutex<BufWriter<std::fs::File>>> = OnceLock::new();

fn log_system_info(mp: &MultiProgress) {
    let logical = num_cpus::get();
    let physical = num_cpus::get_physical();
    print_mp_and_log(mp, &format!("CPU logical cores={logical} physical cores={physical}"));

    #[cfg(target_os = "linux")]
    {
        if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
            if let Some(line) = cpuinfo.lines().find(|l| l.starts_with("model name")) {
                if let Some((_, v)) = line.split_once(':') {
                    print_mp_and_log(mp, &format!("CPU model={}", v.trim()));
                }
            }
        }
        if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
            let mut total_kb: Option<u64> = None;
            let mut avail_kb: Option<u64> = None;
            for line in meminfo.lines() {
                if let Some((k, v)) = line.split_once(':') {
                    let val = v.trim().split_whitespace().next().and_then(|x| x.parse::<u64>().ok());
                    if k == "MemTotal" {
                        total_kb = val;
                    } else if k == "MemAvailable" {
                        avail_kb = val;
                    }
                }
            }
            if let Some(kb) = total_kb {
                let gb = kb as f64 / (1024.0 * 1024.0);
                print_mp_and_log(mp, &format!("MEM total={:.2} GB", gb));
            }
            if let Some(kb) = avail_kb {
                let gb = kb as f64 / (1024.0 * 1024.0);
                print_mp_and_log(mp, &format!("MEM available={:.2} GB", gb));
            }
        }
    }
}

fn timestamp() -> String {
    let format = format_description::parse("[year]-[month]-[day] [hour]:[minute]:[second]")
        .unwrap_or_else(|_| format_description::parse("[year]-[month]-[day] [hour]:[minute]:[second]").unwrap());
    let now = OffsetDateTime::now_local().unwrap_or_else(|_| OffsetDateTime::now_utc());
    now.format(&format).unwrap_or_else(|_| "0000-00-00 00:00:00".to_string())
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
/// Returns (total_twins, total_sum, elapsed_secs, candidates_processed).
fn run_search(
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
    max_duration: Option<Duration>,
) -> Result<(u64, f64, f64, u64)> {
    let k_end_excl = (limit / (wheel_m as u64)) + 2;
    let segment_k = if args.segment_k == 0 {
        pick_segment_k(min_vram, args.segment_mem_frac, residues.len())
    } else {
        args.segment_k
    };

    let total_candidates = (k_end_excl as u128) * (residues.len() as u128);
    let pb = mp.add(ProgressBar::new(
        total_candidates.min(u64::MAX as u128) as u64
    ));
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar:40.cyan/dim}] {percent:>3}% {pos}/{len} (ETA {eta}) | {msg}")?
            .progress_chars("#>-"),
    );
    let msg_prefix = if max_duration.is_some() { "testing" } else { "searching" };
    let num_segments = (k_end_excl + segment_k - 1) / segment_k;
    pb.set_message(format!(
        "{} M={} | seg_k={} | {} segments",
        msg_prefix, wheel_m, format_with_commas(segment_k), format_with_commas(num_segments)
    ));
    pb.enable_steady_tick(Duration::from_millis(200));

    let pb = Arc::new(pb);
    let next_k_low = Arc::new(AtomicU64::new(0));
    let (tx, rx) = mpsc::channel::<SegResult>();
    let run0 = Instant::now();

    let mut handles = Vec::with_capacity(num_gpus as usize);
    for gpu_id in 0..num_gpus {
        let residues = Arc::clone(&residues);
        let small_primes = Arc::clone(&small_primes);
        let small_inv_m_mod_p = Arc::clone(&small_inv_m);
        let large_primes = Arc::clone(&large_primes);
        let large_inv_m_mod_p = Arc::clone(&large_inv_m);
        let next_k_low = Arc::clone(&next_k_low);
        let pb = Arc::clone(&pb);
        let tx = tx.clone();
        let handle = thread::Builder::new()
            .name(format!("gpu-{}", gpu_id))
            .spawn(move || -> Result<()> {
                gpu_worker(
                    gpu_id, wheel_m, residues, small_primes, small_inv_m_mod_p,
                    large_primes, large_inv_m_mod_p, segment_k, k_end_excl, limit,
                    next_k_low, pb, tx,
                )
            })?;
        handles.push(handle);
    }
    drop(tx);

    let mut total = Kahan::default();
    let mut total_twins: u64 = 0;
    let mut last_msg = Instant::now();
    let mut last_log = Instant::now();

    for r in rx.iter() {
        total.add(r.sum);
        total_twins += r.count;
        
        if let Some(d) = max_duration {
            if run0.elapsed() >= d {
                // Signal termination by exhausting the atomic counter
                next_k_low.store(u64::MAX / 2, Ordering::SeqCst);
            }
        }

        if last_msg.elapsed().as_millis() >= 150 {
            let secs = run0.elapsed().as_secs_f64();
            let done = pb.position();
            let speed = if secs > 0.0 { done as f64 / secs } else { 0.0 };

            let msg = format!(
                "M={} | twins={} | B2_partial≈{:.12} | {:.2e} cand/s",
                wheel_m,
                format_with_commas(total_twins),
                total.value(),
                speed
            );
            pb.set_message(msg.clone());
            last_msg = Instant::now();
            if last_log.elapsed().as_secs() >= 60 {
                log_line(&format!("PROGRESS {msg}"));
                last_log = Instant::now();
            }
        }
    }

    for handle in handles {
        handle.join().map_err(|e| anyhow::anyhow!("GPU thread panicked: {:?}", e))??;
    }

    let elapsed = run0.elapsed().as_secs_f64();
    let cand_processed = pb.position();
    
    if max_duration.is_some() {
        pb.finish_and_clear();
    } else {
        pb.finish_with_message(format!("M={} search complete.", wheel_m));
    }

    Ok((total_twins, total.value(), elapsed, cand_processed))
}

/// Per-segment result sent back through the channel.
struct SegResult {
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
    // Find prime factors of M
    let mut factors = Vec::new();
    let mut m = wheel_m;
    let mut d = 2u32;
    while d * d <= m {
        if m % d == 0 {
            factors.push(d as u64);
            while m % d == 0 {
                m /= d;
            }
        }
        d += 1;
    }
    if m > 1 {
        factors.push(m as u64);
    }

    // For each prime factor q, candidate pairs are (q-2, q) and (q, q+2).
    let mut seen = std::collections::BTreeSet::new();
    let mut count = 0u64;
    let mut sum = 0.0f64;

    for &q in &factors {
        // pair (q-2, q)
        if q >= 5 && seen.insert(q - 2) {
            let p = q - 2;
            if p + 2 <= limit && is_prime_small(p) && is_prime_small(p + 2) {
                let pc = gcd_u64(p, wheel_m as u64) == 1;
                let p2c = gcd_u64(p + 2, wheel_m as u64) == 1;
                if !(pc && p2c) {
                    count += 1;
                    sum += 1.0 / (p as f64) + 1.0 / ((p + 2) as f64);
                }
            }
        }
        // pair (q, q+2)
        if q >= 3 && seen.insert(q) {
            let p = q;
            if p + 2 <= limit && is_prime_small(p) && is_prime_small(p + 2) {
                let pc = gcd_u64(p, wheel_m as u64) == 1;
                let p2c = gcd_u64(p + 2, wheel_m as u64) == 1;
                if !(pc && p2c) {
                    count += 1;
                    sum += 1.0 / (p as f64) + 1.0 / ((p + 2) as f64);
                }
            }
        }
    }

    (count, sum)
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
    
    let k = (budget.saturating_mul(4) / r).clamp(1 << 14, 1 << 22);

    k
}

fn estimate_seconds_for_limit(limit: u64, wheel_m: u32, residues_len: usize, cand_per_sec: f64) -> Option<f64> {
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
) -> Result<()> {
    let device = Device::get_device(device_id)?;
    let _ctx = CudaContext::new(device)?;
    let module = Module::from_ptx(PTX, &[])?;

    let clear_kernel = module.get_function("clear_buffers")?;
    let sieve_small_kernel = module.get_function("sieve_wheel_primes_small")?;
    let sieve_kernel = module.get_function("sieve_wheel_primes")?;
    let twin_kernel = module.get_function("twin_sum_wheel")?;

    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let res_len = residues.len();
    let small_prime_count = small_primes.len();
    let large_prime_count = large_primes.len();
    let sieve_shared = (res_len * size_of::<u32>()) as u32;

    let d_residues = DeviceBuffer::from_slice(&residues)?;
    let d_small_primes = DeviceBuffer::from_slice(&small_primes)?;
    let d_small_inv = DeviceBuffer::from_slice(&small_inv_m_mod_p)?;
    let d_large_primes = DeviceBuffer::from_slice(&large_primes)?;
    let d_large_inv = DeviceBuffer::from_slice(&large_inv_m_mod_p)?;

    // bitsets: comp_p and comp_p2
    let candidates_per_seg = segment_k.saturating_mul(res_len as u64);
    let words = ((candidates_per_seg + 31) / 32) as usize;

    let d_comp_p = DeviceBuffer::<u32>::zeroed(words)?;
    let d_comp_p2 = DeviceBuffer::<u32>::zeroed(words)?;

    // kernel launch config — grid sized to the actual workload
    let block = 256u32;
    let sieve_pairs = large_prime_count as u64 * res_len as u64;
    let sieve_grid = ((sieve_pairs + block as u64 - 1) / block as u64).min(262144) as u32;
    let twin_grid = ((words as u64 + block as u64 - 1) / block as u64).min(262144) as u32;
    let clear_grid = ((words as u64 + block as u64 - 1) / block as u64).min(65535) as u32;
    let sieve_small_grid = small_prime_count as u32;

    let d_block_sums = DeviceBuffer::<f64>::zeroed(twin_grid as usize)?;
    let d_block_counts = DeviceBuffer::<u64>::zeroed(twin_grid as usize)?;
    let mut h_block_sums = vec![0f64; twin_grid as usize];
    let mut h_block_counts = vec![0u64; twin_grid as usize];

    loop {
        let k_low = next_k_low.fetch_add(segment_k, Ordering::Relaxed);
        if k_low >= k_end_excl {
            break;
        }

        let k_count = (k_end_excl - k_low).min(segment_k);
        if k_count == 0 {
            break;
        }

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
                    sieve_small_kernel<<<sieve_small_grid, block, sieve_shared, stream>>>(
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

        unsafe {
            cust::launch!(
                twin_kernel<<<twin_grid, block, 0, stream>>>(
                    k_low as u64,
                    k_count as u64,
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

        stream.synchronize()?;

        d_block_sums.copy_to(&mut h_block_sums)?; // CopyDestination in scope
        d_block_counts.copy_to(&mut h_block_counts)?;

        let mut seg_sum = Kahan::default();
        let mut seg_count: u64 = 0;
        for i in 0..(twin_grid as usize) {
            seg_sum.add(h_block_sums[i]);
            seg_count = seg_count.wrapping_add(h_block_counts[i]);
        }

        if tx
            .send(SegResult {
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

    let mp = MultiProgress::new();
    let _ = LOG.set(Mutex::new(BufWriter::new(
        OpenOptions::new().create(true).append(true).open("run.log")?
    )));
    log_system_info(&mp);
    
    cust::init(CudaFlags::empty())?;
    let num_gpus = Device::num_devices()? as u32;
    if num_gpus == 0 {
        anyhow::bail!("No CUDA devices found");
    }

    let mut min_vram: usize = usize::MAX;
    for i in 0..num_gpus {
        let dev = Device::get_device(i)?;
        let mem = dev.total_memory()? as usize;
        let name = dev.name()?;
        let msg = format!(
            "GPU {}: {} ({:.1} GB VRAM)",
            i, name, mem as f64 / (1u64 << 30) as f64
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

        let msg = format!("Testing M={}: {} residues, {} base primes", w, residues.len(), base_primes.len());
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

        if args.benchmark {
            let duration = Duration::from_secs(args.benchmark_seconds.max(1));
            let (_twins, _sum, elapsed, cand) = run_search(
                w, limit, residues, small_primes, small_inv_m, large_primes, large_inv_m,
                num_gpus, min_vram, &args, &mp, Some(duration)
            )?;

            let cand_per_sec = if elapsed > 0.0 { cand as f64 / elapsed } else { 0.0 };
            let est = estimate_seconds_for_limit(args.benchmark_target, w, residues_len, cand_per_sec);

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
                w, limit, residues, small_primes, small_inv_m, large_primes, large_inv_m,
                num_gpus, min_vram, &args, &mp
            );
        }

        let (_twins, _sum, elapsed, cand) = run_search(
            w, limit, residues, small_primes, small_inv_m, large_primes, large_inv_m,
            num_gpus, min_vram, &args, &mp, duration
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
        let msg = format!("Best wheel found: M={best_wheel} ({:.2e} cand/s)", best_speed);
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

        final_report(
            best_wheel, limit, residues, small_primes, small_inv_m, large_primes, large_inv_m,
            num_gpus, min_vram, &args, &mp
        )?;
    }

    Ok(())
}

fn final_report(
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
) -> Result<()> {
    let msg = format!("Running final search with M={wheel_m}...");
    print_mp_and_log(mp, &msg);
    let (missed_count, missed_sum) = count_wheel_missed_twins(wheel_m, limit);
    if missed_count > 0 {
        let msg = format!(
            "Wheel-missed small twin pairs: {} (sum contribution: {:.15})",
            missed_count, missed_sum
        );
        print_mp_and_log(mp, &msg);
    }

    let (total_twins, total_sum, elapsed, _cand) = run_search(
        wheel_m, limit, residues, small_primes, small_inv_m, large_primes, large_inv_m,
        num_gpus, min_vram, args, mp, None
    )?;

    let final_twins = total_twins + missed_count;
    let final_sum = total_sum + missed_sum;

    let ln = (limit as f64).ln();
    let b2_star = final_sum + 4.0 * TWIN_PRIME_CONSTANT_C2 / ln;

    let msg = format!(
        "Done. gpus={}, twins={}",
        num_gpus,
        format_with_commas(final_twins)
    );
    print_and_log(&msg);
    let msg = format!("Brun partial sum up to {limit}: {:.15}", final_sum);
    print_and_log(&msg);
    let msg = format!("Brun extrapolated  B2* (HL):    {:.15}", b2_star);
    print_and_log(&msg);
    let msg = format!("Elapsed: {:.2}s", elapsed);
    print_and_log(&msg);
    Ok(())
}
