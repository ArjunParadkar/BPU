/* bpu_fixed.h -- the contract constants and the fixed-point device model (docs/reference_contract.md, bpusim/fixed.py).
 * Integer-only.  Shared by the PS firmware (write-verify) and the host C model of the PL read path. */
#ifndef BPU_FIXED_H
#define BPU_FIXED_H
#include <stdint.h>
#include <stddef.h>

#define BPU_ROWS        32
#define BPU_COLS        32
#define BPU_GMIN_Q      655
#define BPU_GSTEP       254
#define BPU_ETA_FRAC    14
#define BPU_ETA_ONE     (1 << BPU_ETA_FRAC)
#define BPU_ETA_MAX     (BPU_ETA_ONE - 1)
#define BPU_ZLUT_BITS   10
#define BPU_ZLUT_FRAC   13
#define BPU_N_GEN       64
#define BPU_VERIFY_CODE 64
#define BPU_WS_FRAC     15
#define BPU_WS_ONE      (1 << BPU_WS_FRAC)
#define BPU_GAIN_FRAC   14
#define BPU_GAIN_ONE    (1 << BPU_GAIN_FRAC)
#define BPU_GAIN_MIN    819
#define BPU_LUT_BITS    10
#define BPU_LUT_N       (1 << BPU_LUT_BITS)
#define BPU_TOL_Q       655
#define BPU_DAMP_Q      11469
#define BPU_VERIFY_READS 8
#define BPU_MAX_ITER    100

/* ---- read path (contract section 3) ---- */
static inline uint32_t bpu_conductance_q(uint32_t w_q) { return BPU_GMIN_Q + BPU_GSTEP * w_q; }
static inline uint32_t bpu_noisy_conductance_q(uint32_t g_q, int32_t eta_q) {
    return (uint32_t)(((uint64_t)g_q * (uint64_t)(BPU_ETA_ONE + eta_q) + 8192u) >> BPU_ETA_FRAC);
}
/* round-half-to-even arithmetic shift (the SAR ADC) */
static inline int64_t bpu_rne_shift(int64_t x, unsigned s) {
    if (s == 0) return x;
    int64_t r = x >> s;                       /* arithmetic (floor) on two's complement */
    uint64_t rem = (uint64_t)x - ((uint64_t)r << s);
    uint64_t half = 1ull << (s - 1);
    if (rem > half || (rem == half && (r & 1))) r += 1;
    return r;
}
static inline int32_t bpu_sat8(int64_t q) { return q > 127 ? 127 : (q < -128 ? -128 : (int32_t)q); }

/* ---- PRNG (contract section 5B) ---- */
typedef struct { uint32_t s[4]; } bpu_xoshiro_t;
static inline uint32_t bpu_rotl32(uint32_t x, int k) { return (x << k) | (x >> (32 - k)); }
static inline uint32_t bpu_xoshiro_next(bpu_xoshiro_t *g) {
    uint32_t *s = g->s;
    uint32_t result = bpu_rotl32(s[1] * 5u, 7) * 9u;
    uint32_t t = s[1] << 9;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = bpu_rotl32(s[3], 11);
    return result;
}
static inline uint64_t bpu_splitmix64(uint64_t *state) {
    *state += 0x9E3779B97F4A7C15ull;
    uint64_t z = *state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static inline void bpu_prng_seed(bpu_xoshiro_t *g, uint64_t seed64, unsigned instance) {
    uint64_t st = seed64 + (uint64_t)instance * 4ull * 0x9E3779B97F4A7C15ull;
    int allzero = 1;
    for (int j = 0; j < 4; j++) { g->s[j] = (uint32_t)bpu_splitmix64(&st); if (g->s[j]) allzero = 0; }
    if (allzero) g->s[0] = 1;
}
extern const int16_t bpu_zlut[BPU_LUT_N];     /* generated: ps/src/bpu_tables.c */
static inline int32_t bpu_prng_eta(bpu_xoshiro_t *g, uint32_t sigma_q) {
    uint32_t w = bpu_xoshiro_next(g);
    int32_t z = bpu_zlut[w >> (32 - BPU_ZLUT_BITS)];
    int64_t e = ((int64_t)sigma_q * z + 4096) >> BPU_ZLUT_FRAC;
    if (e > BPU_ETA_MAX) e = BPU_ETA_MAX;
    if (e < -BPU_ETA_MAX) e = -BPU_ETA_MAX;
    return (int32_t)e;
}

/* ---- write path (contract section 6) ---- */
extern const uint16_t bpu_step_up[BPU_LUT_N], bpu_step_dn[BPU_LUT_N], bpu_nup[BPU_LUT_N], bpu_ndn[BPU_LUT_N];
static inline unsigned bpu_lut_index(int32_t w_s) { int32_t k = w_s >> (BPU_WS_FRAC - BPU_LUT_BITS); return k > BPU_LUT_N - 1 ? BPU_LUT_N - 1 : (unsigned)k; }
static inline uint32_t bpu_ws_to_wq(int32_t w_s) { int32_t q = (w_s + 64) >> (BPU_WS_FRAC - 8); return q > 255 ? 255 : (uint32_t)q; }
static inline int32_t bpu_step_q(int32_t w_s, uint32_t gain_q, int32_t c2c_q, int dir) {
    unsigned k = bpu_lut_index(w_s);
    uint64_t base = dir > 0 ? bpu_step_up[k] : bpu_step_dn[k];
    uint64_t prod = base * gain_q * (uint64_t)(BPU_ETA_ONE + c2c_q);
    return (int32_t)((prod + (1ull << (BPU_GAIN_FRAC + BPU_ETA_FRAC - 1))) >> (BPU_GAIN_FRAC + BPU_ETA_FRAC));
}
/* one device, n signed pulses, each with a fresh c2c sample from gen */
static inline int32_t bpu_pulse(int32_t w_s, int32_t n, uint32_t gain_q, bpu_xoshiro_t *c2c_gen, uint32_t sigma_c2c_q) {
    while (n != 0) {
        int dir = n > 0 ? 1 : -1;
        int32_t c = bpu_prng_eta(c2c_gen, sigma_c2c_q);
        int32_t s = bpu_step_q(w_s, gain_q, c, dir);
        w_s += dir * s;
        if (w_s < 0) w_s = 0;
        if (w_s > BPU_WS_ONE) w_s = BPU_WS_ONE;
        n -= dir;
    }
    return w_s;
}
static inline int32_t bpu_pulses_for(int32_t w_s, int32_t w_target_s) {
    unsigned i0 = bpu_lut_index(w_s), i1 = bpu_lut_index(w_target_s);
    if (w_target_s >= w_s) return (int32_t)bpu_rne_shift((int64_t)bpu_nup[i1] - (int64_t)bpu_nup[i0], 8);
    return -(int32_t)bpu_rne_shift((int64_t)bpu_ndn[i1] - (int64_t)bpu_ndn[i0], 8);
}
static inline int32_t bpu_w_meas_from_sum(int64_t g_sum, int n_reads) {
    int64_t den = (int64_t)n_reads * BPU_VERIFY_CODE * BPU_GSTEP * 255;
    int64_t num = (g_sum - (int64_t)n_reads * BPU_VERIFY_CODE * BPU_GMIN_Q) * BPU_WS_ONE;
    if (num < 0) num = 0;
    int64_t w = (num + den / 2) / den;
    return w > BPU_WS_ONE ? BPU_WS_ONE : (int32_t)w;
}
#endif
