use anyhow::Result;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{Duration, Instant};

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

fn mark_prime_list(
    wheel_m: u32,
    k_low: u64,
    k_count: u64,
    residues: &[u32],
    primes: &[u32],
    inv_m_mod_p: &[u32],
    comp_p: &mut [u64],
    comp_p2: &mut [u64],
) {
    let k_end = k_low + k_count;
    for (pi, &q) in primes.iter().enumerate() {
        let inv_m = inv_m_mod_p[pi] as u64;
        let q_u = q as u64;
        let qq = q_u * q_u;
        for (ridx, &r) in residues.iter().enumerate() {
            let r_u = r as u64;

            // comp_p
            let neg_r = (q_u - (r_u % q_u)) % q_u;
            let k0 = (neg_r * inv_m) % q_u;
            let k_mod = k_low % q_u;
            let need = if k0 >= k_mod { k0 - k_mod } else { q_u + k0 - k_mod };
            let mut first_k = k_low + need;
            while first_k < k_end {
                let p = (wheel_m as u64) * first_k + r_u;
                if p >= qq {
                    break;
                }
                first_k += q_u;
            }
            let mut kk = first_k;
            while kk < k_end {
                let idx = (ridx as u64) * k_count + (kk - k_low);
                let w = (idx >> 6) as usize;
                let b = (idx & 63) as u32;
                comp_p[w] |= 1u64 << b;
                kk += q_u;
            }

            // comp_p2
            let rp2 = r_u + 2;
            let neg_rp2 = (q_u - (rp2 % q_u)) % q_u;
            let k1 = (neg_rp2 * inv_m) % q_u;
            let k_mod = k_low % q_u;
            let need = if k1 >= k_mod { k1 - k_mod } else { q_u + k1 - k_mod };
            let mut first_k = k_low + need;
            while first_k < k_end {
                let p = (wheel_m as u64) * first_k + r_u;
                let p2 = p + 2;
                if p2 >= qq {
                    break;
                }
                first_k += q_u;
            }
            let mut kk = first_k;
            while kk < k_end {
                let idx = (ridx as u64) * k_count + (kk - k_low);
                let w = (idx >> 6) as usize;
                let b = (idx & 63) as u32;
                comp_p2[w] |= 1u64 << b;
                kk += q_u;
            }
        }
    }
}

struct SegResult {
    sum: f64,
    count: u64,
    candidates: u64,
}

pub fn run_search_cpu(
    wheel_m: u32,
    limit: u64,
    segment_k: u64,
    residues: &[u32],
    small_primes: &[u32],
    small_inv_m: &[u32],
    large_primes: &[u32],
    large_inv_m: &[u32],
    num_threads: usize,
    mp: &MultiProgress,
    max_duration: Option<Duration>,
) -> Result<(u64, f64, f64, u64)> {
    let k_end_excl = (limit / (wheel_m as u64)) + 2;
    let total_candidates = (k_end_excl as u128) * (residues.len() as u128);

    let pb = mp.add(ProgressBar::new(
        total_candidates.min(u64::MAX as u128) as u64,
    ));
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar:40.cyan/dim}] {percent:>3}% {pos}/{len} | {msg}")?
            .progress_chars("#>-"),
    );
    let msg_prefix = if max_duration.is_some() { "testing" } else { "searching" };
    let num_segments = (k_end_excl + segment_k - 1) / segment_k;
    pb.set_message(format!(
        "{}(cpu) M={} | seg_k={} | {} segments | threads={}",
        msg_prefix,
        wheel_m,
        format_with_commas(segment_k),
        format_with_commas(num_segments),
        num_threads
    ));
    pb.enable_steady_tick(Duration::from_millis(200));

    let run0 = Instant::now();
    let mut total = Kahan::default();
    let mut total_twins: u64 = 0;
    let mut segments_done: u64 = 0;
    let mut last_msg = Instant::now();
    let mut ema_speed: f64 = 0.0;

    let next_k_low = Arc::new(AtomicU64::new(0));
    let (tx, rx) = mpsc::channel::<SegResult>();

    let residues = Arc::new(residues.to_vec());
    let small_primes = Arc::new(small_primes.to_vec());
    let small_inv_m = Arc::new(small_inv_m.to_vec());
    let large_primes = Arc::new(large_primes.to_vec());
    let large_inv_m = Arc::new(large_inv_m.to_vec());

    let mut handles = Vec::with_capacity(num_threads);
    for tid in 0..num_threads {
        let residues = Arc::clone(&residues);
        let small_primes = Arc::clone(&small_primes);
        let small_inv_m = Arc::clone(&small_inv_m);
        let large_primes = Arc::clone(&large_primes);
        let large_inv_m = Arc::clone(&large_inv_m);
        let next_k_low = Arc::clone(&next_k_low);
        let tx = tx.clone();

        let handle = thread::Builder::new()
            .name(format!("cpu-{}", tid))
            .spawn(move || {
                loop {
                    let k_low = next_k_low.fetch_add(segment_k, Ordering::Relaxed);
                    if k_low >= k_end_excl {
                        break;
                    }
                    let k_count = (k_end_excl - k_low).min(segment_k);
                    if k_count == 0 {
                        break;
                    }

                    let candidates_per_seg = k_count * (residues.len() as u64);
                    let words = ((candidates_per_seg + 63) / 64) as usize;
                    let mut comp_p = vec![0u64; words];
                    let mut comp_p2 = vec![0u64; words];

                    if !small_primes.is_empty() {
                        mark_prime_list(
                            wheel_m,
                            k_low,
                            k_count,
                            &residues,
                            &small_primes,
                            &small_inv_m,
                            &mut comp_p,
                            &mut comp_p2,
                        );
                    }
                    if !large_primes.is_empty() {
                        mark_prime_list(
                            wheel_m,
                            k_low,
                            k_count,
                            &residues,
                            &large_primes,
                            &large_inv_m,
                            &mut comp_p,
                            &mut comp_p2,
                        );
                    }

                    let mut seg_sum = Kahan::default();
                    let mut seg_count: u64 = 0;

                    for (w, (&cp, &c2)) in comp_p.iter().zip(comp_p2.iter()).enumerate() {
                        let mut ok = !(cp | c2);
                        let base_idx = (w as u64) * 64;
                        let remaining = if candidates_per_seg > base_idx {
                            candidates_per_seg - base_idx
                        } else {
                            0
                        };
                        if remaining == 0 {
                            ok = 0;
                        } else if remaining < 64 {
                            ok &= (1u64 << (remaining as u32)) - 1;
                        }

                        while ok != 0 {
                            let bit = ok.trailing_zeros();
                            ok &= ok - 1;

                            let idx = base_idx + (bit as u64);
                            let kk = k_low + (idx % k_count);
                            let ridx = (idx / k_count) as usize;
                            let p = (wheel_m as u64) * kk + (residues[ridx] as u64);
                            if p < 3 || p + 2 > limit {
                                continue;
                            }
                            seg_sum.add(1.0 / (p as f64) + 1.0 / ((p + 2) as f64));
                            seg_count = seg_count.wrapping_add(1);
                        }
                    }

                    if tx
                        .send(SegResult {
                            sum: seg_sum.value(),
                            count: seg_count,
                            candidates: candidates_per_seg,
                        })
                        .is_err()
                    {
                        break;
                    }
                }
            })?;
        handles.push(handle);
    }
    drop(tx);

    for r in rx.iter() {
        total.add(r.sum);
        total_twins += r.count;
        segments_done += 1;
        pb.inc(r.candidates);

        if let Some(d) = max_duration {
            if run0.elapsed() >= d {
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
            let eta_secs = if ema_speed > 0.0 { remaining as f64 / ema_speed } else { 0.0 };
            let msg = format!(
                "seg {}/{} | twins={} | B2_partial≈{:.12} | {:.2e} cand/s | ETA {:.1} min",
                format_with_commas(segments_done),
                format_with_commas(num_segments),
                format_with_commas(total_twins),
                total.value(),
                ema_speed,
                eta_secs / 60.0
            );
            pb.set_message(msg);
            last_msg = Instant::now();
        }
    }

    for handle in handles {
        let _ = handle.join();
    }

    let elapsed = run0.elapsed().as_secs_f64();
    let cand_processed = pb.position();
    if max_duration.is_some() {
        pb.finish_and_clear();
    } else {
        pb.finish_with_message(format!("M={} search complete. (cpu)", wheel_m));
    }

    Ok((total_twins, total.value(), elapsed, cand_processed))
}
