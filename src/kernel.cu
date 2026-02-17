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
  idx = (k - k_low) * R + ridx
where R = residues_len, k in [k_low, k_low + k_count).
*/

__device__ __forceinline__ unsigned int bit_word(unsigned long long idx)
{
    return (unsigned int)(idx >> 5ULL);
}
__device__ __forceinline__ unsigned int bit_mask(unsigned long long idx)
{
    return 1U << (unsigned int)(idx & 31ULL);
}

/* Extended Euclid for modular inverse (device) — only used if you choose to compute inverses on GPU.
   In this version, inverses invM_mod_p[] are computed on CPU and uploaded. */
static __device__ __forceinline__ unsigned int mul_mod_u32(unsigned int a, unsigned int b, unsigned int mod)
{
    return (unsigned long long)a * (unsigned long long)b % (unsigned long long)mod;
}

/* Prime-parallel wheel sieve kernel.
   Each thread owns a prime q and marks candidate indices where:
     p     = M*k + r ≡ 0 (mod q)  => comp_p
     p + 2 = M*k + r + 2 ≡ 0 (mod q) => comp_p2
   Start is bumped to >= q*q to avoid marking the prime itself. */
extern "C" __global__ void sieve_wheel_primes(
    unsigned long long k_low,
    unsigned long long k_count,
    unsigned int M,
    const unsigned int *__restrict__ residues, // length R
    int R,
    const unsigned int *__restrict__ primes,     // base primes excluding factors of M and excluding 2
    const unsigned int *__restrict__ invM_mod_p, // invM_mod_p[i] = M^{-1} mod primes[i]
    int prime_count,
    unsigned int *__restrict__ comp_p_words,
    unsigned int *__restrict__ comp_p2_words)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int pi = tid; pi < (unsigned int)prime_count; pi += stride)
    {
        unsigned int q = primes[pi];
        unsigned int invM = invM_mod_p[pi];

        unsigned long long qq = (unsigned long long)q * (unsigned long long)q;

        // For each residue class r, solve k ≡ (-r)*invM (mod q) for p divisible
        // and k ≡ (-(r+2))*invM (mod q) for p+2 divisible.
        for (int ridx = 0; ridx < R; ++ridx)
        {
            unsigned int r = residues[ridx];

            // k0 for p divisible by q
            unsigned int neg_r = (q - (r % q)) % q;
            unsigned int k0 = mul_mod_u32(neg_r, invM, q);

            // k1 for (p+2) divisible by q
            unsigned int rp2 = r + 2U;
            unsigned int neg_rp2 = (q - (rp2 % q)) % q;
            unsigned int k1 = mul_mod_u32(neg_rp2, invM, q);

            // Find first k >= k_low with k ≡ k0 (mod q)
            unsigned long long k = k_low;
            unsigned long long mod = (unsigned long long)q;
            unsigned long long k_mod = k % mod;
            unsigned long long need = (k0 >= k_mod) ? (k0 - k_mod) : (mod + k0 - k_mod);
            unsigned long long first_k = k + need;

            // bump until p >= q*q
            // p = M*first_k + r
            while (first_k < k_low + k_count)
            {
                unsigned long long p = (unsigned long long)M * first_k + (unsigned long long)r;
                if (p >= qq)
                    break;
                first_k += mod;
            }

            for (unsigned long long kk2 = first_k; kk2 < k_low + k_count; kk2 += mod)
            {
                unsigned long long idx = (kk2 - k_low) * (unsigned long long)R + (unsigned long long)ridx;
                atomicOr(&comp_p_words[bit_word(idx)], bit_mask(idx));
            }

            // Now for p+2 divisible:
            k_mod = k % mod;
            need = (k1 >= k_mod) ? (k1 - k_mod) : (mod + k1 - k_mod);
            first_k = k + need;

            while (first_k < k_low + k_count)
            {
                unsigned long long p = (unsigned long long)M * first_k + (unsigned long long)r;
                unsigned long long p2 = p + 2ULL;
                if (p2 >= qq)
                    break;
                first_k += mod;
            }

            for (unsigned long long kk2 = first_k; kk2 < k_low + k_count; kk2 += mod)
            {
                unsigned long long idx = (kk2 - k_low) * (unsigned long long)R + (unsigned long long)ridx;
                atomicOr(&comp_p2_words[bit_word(idx)], bit_mask(idx));
            }
        }
    }
}

/* Twin sum over wheel candidates.
   A candidate idx corresponds to p = M*(k_low + idx/R) + residues[idx%R].
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
    const unsigned int *__restrict__ comp_p_words,
    const unsigned int *__restrict__ comp_p2_words,
    double *__restrict__ d_sum,
    unsigned long long *__restrict__ d_count)
{
    unsigned long long total_candidates = k_count * (unsigned long long)R;
    unsigned int total_words = (unsigned int)((total_candidates + 31ULL) / 32ULL);

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    double local_sum = 0.0;
    unsigned long long local_count = 0ULL;

    for (unsigned int w = tid; w < total_words; w += stride)
    {
        unsigned int cp = comp_p_words[w];
        unsigned int c2 = comp_p2_words[w];
        unsigned int ok = ~(cp | c2); // 1 bits are "both not composite"
        // Clamp last word
        unsigned long long base_idx = (unsigned long long)w * 32ULL;
        unsigned long long remaining = (total_candidates > base_idx) ? (total_candidates - base_idx) : 0ULL;
        if (remaining == 0ULL)
        {
            ok = 0U;
        }
        else if (remaining < 32ULL)
        {
            ok &= (1U << (unsigned int)remaining) - 1U;
        }

        while (ok != 0U)
        {
            int bit = __ffs((int)ok) - 1;
            ok &= ok - 1U;

            unsigned long long idx = base_idx + (unsigned int)bit;
            unsigned long long kk = k_low + idx / (unsigned long long)R;
            unsigned int ridx = (unsigned int)(idx % (unsigned long long)R);
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
            atomicAdd(d_sum, local_sum);
            atomicAdd(d_count, local_count);
        }
    }
}
