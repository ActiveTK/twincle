#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

/*
Wheel-based twin-candidate representation:

We only represent candidate starts p such that:
  gcd(p, M) = 1 and gcd(p+2, M) = 1
where M is wheel modulus (30 / 210 / 30030).
Such candidates can be enumerated as:
  p = M * k + r
for k >= 0, r in residues[] (precomputed on CPU).

We maintain two composite bitsets:
  comp_p  : p is composite (divisible by some q)
  comp_p2 : p+2 is composite
A twin pair is valid if both bits are 0 and p >= 3 and p+2 <= limit.

Bitset indexing:
  idx = ridx * k_count + (k - k_low)
where R = residues_len, k in [k_low, k_low + k_count).
*/

__device__ __forceinline__ unsigned int bit_word(unsigned long long idx)
{
    return (unsigned int)(idx >> 6ULL);
}
__device__ __forceinline__ unsigned long long bit_mask(unsigned long long idx)
{
    return 1ULL << (unsigned int)(idx & 63ULL);
}

/* Extended Euclid for modular inverse (device) — only used if you choose to compute inverses on GPU.
   In this version, inverses invM_mod_p[] are computed on CPU and uploaded. */
static __device__ __forceinline__ unsigned int mul_mod_u32(unsigned int a, unsigned int b, unsigned int mod)
{
    return (unsigned long long)a * (unsigned long long)b % (unsigned long long)mod;
}

/* Stream-ordered buffer clear.
   Replaces cuMemsetD32_v2 (default-stream, blocks host) so that the clear
   is enqueued on the same non-blocking stream as the sieve/twin kernels. */
extern "C" __global__ void clear_buffers(
    unsigned long long *comp_p,
    unsigned long long *comp_p2,
    unsigned int  word_count)
{
    unsigned int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = tid; i < word_count; i += stride)
    {
        comp_p[i]  = 0ULL;
        comp_p2[i] = 0ULL;
    }
}

/* Pair-parallel wheel sieve kernel.
   OLD: each thread owned one prime and looped over ALL R residues — the thread
   handling the smallest prime (q=17) did millions of marks while others were
   idle, causing catastrophic warp-level load imbalance.

   NEW: the iteration space is (prime_index, residue_index) pairs.
   Total pairs = prime_count * R.  Each thread picks pairs via a stride loop,
   so the heavy marking work for small primes is spread across many threads. */
extern "C" __global__ void sieve_wheel_primes(
    unsigned long long k_low,
    unsigned long long k_count,
    unsigned int M,
    const unsigned int *__restrict__ residues, // length R
    int R,
    const unsigned int *__restrict__ primes,
    const unsigned int *__restrict__ invM_mod_p,
    int prime_count,
    unsigned long long *__restrict__ comp_p_words,
    unsigned long long *__restrict__ comp_p2_words)
{
    extern __shared__ unsigned int s_res[];
    for (int i = threadIdx.x; i < R; i += (int)blockDim.x)
        s_res[i] = residues[i];
    __syncthreads();

    unsigned long long total_pairs = (unsigned long long)prime_count * (unsigned long long)R;
    unsigned long long tid    = (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x
                              + (unsigned long long)threadIdx.x;
    unsigned long long stride = (unsigned long long)blockDim.x * (unsigned long long)gridDim.x;

    for (unsigned long long pair = tid; pair < total_pairs; pair += stride)
    {
        unsigned int pi   = (unsigned int)(pair / (unsigned long long)R);
        unsigned int ridx = (unsigned int)(pair % (unsigned long long)R);

        unsigned int q    = primes[pi];
        unsigned int invM = invM_mod_p[pi];
        unsigned int r    = s_res[ridx];

        unsigned long long qq  = (unsigned long long)q * (unsigned long long)q;
        unsigned long long mod = (unsigned long long)q;
        unsigned long long k_end = k_low + k_count;

        /* --- comp_p: mark k where M*k + r ≡ 0 (mod q) --- */
        {
            unsigned int neg_r = (q - (r % q)) % q;
            unsigned int k0    = mul_mod_u32(neg_r, invM, q);

            unsigned long long k_mod = k_low % mod;
            unsigned long long need  = (k0 >= k_mod) ? (k0 - k_mod) : (mod + k0 - k_mod);
            unsigned long long first_k = k_low + need;

            while (first_k < k_end)
            {
                unsigned long long p = (unsigned long long)M * first_k + (unsigned long long)r;
                if (p >= qq) break;
                first_k += mod;
            }

            unsigned int cur_word = 0;
            unsigned long long cur_mask = 0ULL;
            bool has_word = false;
            for (unsigned long long kk = first_k; kk < k_end; kk += mod)
            {
                unsigned long long idx = (unsigned long long)ridx * (unsigned long long)k_count
                                       + (kk - k_low);
                unsigned int w = bit_word(idx);
                unsigned long long m = bit_mask(idx);
                if (!has_word)
                {
                    cur_word = w;
                    cur_mask = m;
                    has_word = true;
                }
                else if (w == cur_word)
                {
                    cur_mask |= m;
                }
                else
                {
                    atomicOr(&comp_p_words[cur_word], cur_mask);
                    cur_word = w;
                    cur_mask = m;
                }
            }
            if (has_word)
            {
                atomicOr(&comp_p_words[cur_word], cur_mask);
            }
        }

        /* --- comp_p2: mark k where M*k + r + 2 ≡ 0 (mod q) --- */
        {
            unsigned int rp2     = r + 2U;
            unsigned int neg_rp2 = (q - (rp2 % q)) % q;
            unsigned int k1      = mul_mod_u32(neg_rp2, invM, q);

            unsigned long long k_mod = k_low % mod;
            unsigned long long need  = (k1 >= k_mod) ? (k1 - k_mod) : (mod + k1 - k_mod);
            unsigned long long first_k = k_low + need;

            while (first_k < k_end)
            {
                unsigned long long p  = (unsigned long long)M * first_k + (unsigned long long)r;
                unsigned long long p2 = p + 2ULL;
                if (p2 >= qq) break;
                first_k += mod;
            }

            unsigned int cur_word = 0;
            unsigned long long cur_mask = 0ULL;
            bool has_word = false;
            for (unsigned long long kk = first_k; kk < k_end; kk += mod)
            {
                unsigned long long idx = (unsigned long long)ridx * (unsigned long long)k_count
                                       + (kk - k_low);
                unsigned int w = bit_word(idx);
                unsigned long long m = bit_mask(idx);
                if (!has_word)
                {
                    cur_word = w;
                    cur_mask = m;
                    has_word = true;
                }
                else if (w == cur_word)
                {
                    cur_mask |= m;
                }
                else
                {
                    atomicOr(&comp_p2_words[cur_word], cur_mask);
                    cur_word = w;
                    cur_mask = m;
                }
            }
            if (has_word)
            {
                atomicOr(&comp_p2_words[cur_word], cur_mask);
            }
        }
    }
}

/* Prime-major sieve for small primes.
   One block per prime, threads iterate residues. */
extern "C" __global__ void sieve_wheel_primes_small(
    unsigned long long k_low,
    unsigned long long k_count,
    unsigned int M,
    const unsigned int *__restrict__ residues, // length R
    int R,
    const unsigned int *__restrict__ primes,
    const unsigned int *__restrict__ invM_mod_p,
    int prime_count,
    unsigned long long *__restrict__ comp_p_words,
    unsigned long long *__restrict__ comp_p2_words)
{
    unsigned int pi = (unsigned int)blockIdx.x;
    if ((int)pi >= prime_count) return;

    extern __shared__ unsigned int s_res[];
    for (int i = threadIdx.x; i < R; i += (int)blockDim.x)
        s_res[i] = residues[i];
    __syncthreads();

    unsigned int q = primes[pi];
    unsigned int invM = invM_mod_p[pi];
    unsigned long long qq  = (unsigned long long)q * (unsigned long long)q;
    unsigned long long mod = (unsigned long long)q;
    unsigned long long k_end = k_low + k_count;

    for (int ridx = threadIdx.x; ridx < R; ridx += (int)blockDim.x)
    {
        unsigned int r = s_res[ridx];

        /* --- comp_p: mark k where M*k + r ≡ 0 (mod q) --- */
        {
            unsigned int neg_r = (q - (r % q)) % q;
            unsigned int k0    = mul_mod_u32(neg_r, invM, q);

            unsigned long long k_mod = k_low % mod;
            unsigned long long need  = (k0 >= k_mod) ? (k0 - k_mod) : (mod + k0 - k_mod);
            unsigned long long first_k = k_low + need;

            while (first_k < k_end)
            {
                unsigned long long p = (unsigned long long)M * first_k + (unsigned long long)r;
                if (p >= qq) break;
                first_k += mod;
            }

            unsigned int cur_word = 0;
            unsigned long long cur_mask = 0ULL;
            bool has_word = false;
            for (unsigned long long kk = first_k; kk < k_end; kk += mod)
            {
                unsigned long long idx = (unsigned long long)ridx * (unsigned long long)k_count
                                       + (kk - k_low);
                unsigned int w = bit_word(idx);
                unsigned long long m = bit_mask(idx);
                if (!has_word)
                {
                    cur_word = w;
                    cur_mask = m;
                    has_word = true;
                }
                else if (w == cur_word)
                {
                    cur_mask |= m;
                }
                else
                {
                    atomicOr(&comp_p_words[cur_word], cur_mask);
                    cur_word = w;
                    cur_mask = m;
                }
            }
            if (has_word)
            {
                atomicOr(&comp_p_words[cur_word], cur_mask);
            }
        }

        /* --- comp_p2: mark k where M*k + r + 2 ≡ 0 (mod q) --- */
        {
            unsigned int rp2     = r + 2U;
            unsigned int neg_rp2 = (q - (rp2 % q)) % q;
            unsigned int k1      = mul_mod_u32(neg_rp2, invM, q);

            unsigned long long k_mod = k_low % mod;
            unsigned long long need  = (k1 >= k_mod) ? (k1 - k_mod) : (mod + k1 - k_mod);
            unsigned long long first_k = k_low + need;

            while (first_k < k_end)
            {
                unsigned long long p  = (unsigned long long)M * first_k + (unsigned long long)r;
                unsigned long long p2 = p + 2ULL;
                if (p2 >= qq) break;
                first_k += mod;
            }

            unsigned int cur_word = 0;
            unsigned long long cur_mask = 0ULL;
            bool has_word = false;
            for (unsigned long long kk = first_k; kk < k_end; kk += mod)
            {
                unsigned long long idx = (unsigned long long)ridx * (unsigned long long)k_count
                                       + (kk - k_low);
                unsigned int w = bit_word(idx);
                unsigned long long m = bit_mask(idx);
                if (!has_word)
                {
                    cur_word = w;
                    cur_mask = m;
                    has_word = true;
                }
                else if (w == cur_word)
                {
                    cur_mask |= m;
                }
                else
                {
                    atomicOr(&comp_p2_words[cur_word], cur_mask);
                    cur_word = w;
                    cur_mask = m;
                }
            }
            if (has_word)
            {
                atomicOr(&comp_p2_words[cur_word], cur_mask);
            }
        }
    }
}

/* Twin sum over wheel candidates.
   A candidate idx corresponds to:
     ridx = idx / k_count
     k    = k_low + (idx % k_count)
     p    = M*k + residues[ridx]
   It's a twin pair iff comp_p[idx]==0 and comp_p2[idx]==0 and p>=3 and p+2<=limit. */

static __device__ __forceinline__ double warp_reduce_sum_f64(double val)
{
    unsigned int lo = __double2loint(val);
    unsigned int hi = __double2hiint(val);
    for (int off = 16; off > 0; off >>= 1)
    {
        unsigned int lo2 = __shfl_down_sync(0xffffffffU, lo, off);
        unsigned int hi2 = __shfl_down_sync(0xffffffffU, hi, off);
        double other = __hiloint2double(hi2, lo2);
        val += other;
        lo = __double2loint(val);
        hi = __double2hiint(val);
    }
    return val;
}

static __device__ __forceinline__ unsigned long long warp_reduce_sum_u64(unsigned long long val)
{
    unsigned int lo = (unsigned int)val;
    unsigned int hi = (unsigned int)(val >> 32);
    for (int off = 16; off > 0; off >>= 1)
    {
        unsigned int lo2 = __shfl_down_sync(0xffffffffU, lo, off);
        unsigned int hi2 = __shfl_down_sync(0xffffffffU, hi, off);
        unsigned long long other = ((unsigned long long)hi2 << 32) | lo2;
        val += other;
        lo = (unsigned int)val;
        hi = (unsigned int)(val >> 32);
    }
    return val;
}

extern "C" __global__ void twin_sum_wheel(
    unsigned long long k_low,
    unsigned long long k_count,
    unsigned int M,
    const unsigned int *__restrict__ residues,
    int R,
    unsigned long long limit,
    const unsigned long long *__restrict__ comp_p_words,
    const unsigned long long *__restrict__ comp_p2_words,
    double *__restrict__ block_sums,
    unsigned long long *__restrict__ block_counts)
{
    unsigned long long total_candidates = k_count * (unsigned long long)R;
    unsigned int total_words = (unsigned int)((total_candidates + 63ULL) / 64ULL);

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    double local_sum = 0.0;
    unsigned long long local_count = 0ULL;

    for (unsigned int w = tid; w < total_words; w += stride)
    {
        unsigned long long cp = comp_p_words[w];
        unsigned long long c2 = comp_p2_words[w];
        unsigned long long ok = ~(cp | c2); // 1 bits are "both not composite"
        // Clamp last word
        unsigned long long base_idx = (unsigned long long)w * 64ULL;
        unsigned long long remaining = (total_candidates > base_idx) ? (total_candidates - base_idx) : 0ULL;
        if (remaining == 0ULL)
        {
            ok = 0ULL;
        }
        else if (remaining < 64ULL)
        {
            ok &= (1ULL << (unsigned int)remaining) - 1ULL;
        }

        while (ok != 0ULL)
        {
            int bit = __ffsll((long long)ok) - 1;
            ok &= ok - 1ULL;

            unsigned long long idx = base_idx + (unsigned int)bit;
            unsigned long long kk = k_low + (idx % (unsigned long long)k_count);
            unsigned int ridx = (unsigned int)(idx / (unsigned long long)k_count);
            unsigned long long p = (unsigned long long)M * kk + (unsigned long long)residues[ridx];

            if (p < 3ULL)
                continue;
            if (p + 2ULL > limit)
                continue;

            local_sum += 1.0 / (double)p + 1.0 / (double)(p + 2ULL);
            local_count += 1ULL;
        }
    }

    local_sum = warp_reduce_sum_f64(local_sum);
    local_count = warp_reduce_sum_u64(local_count);

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    __shared__ double s_sum[8];
    __shared__ unsigned long long s_cnt[8];

    if (lane == 0)
    {
        s_sum[warp_id] = local_sum;
        s_cnt[warp_id] = local_count;
    }
    __syncthreads();

    if (warp_id == 0)
    {
        int num_warps = (int)(blockDim.x >> 5);
        local_sum = (lane < num_warps) ? s_sum[lane] : 0.0;
        local_count = (lane < num_warps) ? s_cnt[lane] : 0ULL;

        local_sum = warp_reduce_sum_f64(local_sum);
        local_count = warp_reduce_sum_u64(local_count);

        if (lane == 0)
        {
            block_sums[blockIdx.x] = local_sum;
            block_counts[blockIdx.x] = local_count;
        }
    }
}

extern "C" __global__ void reduce_block_results(
    const double *__restrict__ block_sums,
    const unsigned long long *__restrict__ block_counts,
    unsigned int n,
    double *__restrict__ out_sum,
    unsigned long long *__restrict__ out_count)
{
    double sum = 0.0;
    unsigned long long cnt = 0ULL;

    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x)
    {
        sum += block_sums[i];
        cnt += block_counts[i];
    }

    sum = warp_reduce_sum_f64(sum);
    cnt = warp_reduce_sum_u64(cnt);

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    __shared__ double s_sum[8];
    __shared__ unsigned long long s_cnt[8];

    if (lane == 0)
    {
        s_sum[warp_id] = sum;
        s_cnt[warp_id] = cnt;
    }
    __syncthreads();

    if (warp_id == 0)
    {
        int num_warps = (int)(blockDim.x >> 5);
        sum = (lane < num_warps) ? s_sum[lane] : 0.0;
        cnt = (lane < num_warps) ? s_cnt[lane] : 0ULL;

        sum = warp_reduce_sum_f64(sum);
        cnt = warp_reduce_sum_u64(cnt);

        if (lane == 0)
        {
            *out_sum = sum;
            *out_count = cnt;
        }
    }
}
