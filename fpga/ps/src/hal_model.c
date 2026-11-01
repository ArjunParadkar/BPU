/* hal_model.c -- a HOST C model of the PL behind the HAL: the same register map and stream encodings, the read
 * datapath ported from the contract.  It is the third independent implementation (Python reference, RTL, C) and
 * lets the firmware's sequencer / write-verify run and be tested on the host without a board. */
#include <stdlib.h>
#include <string.h>
#include "bpu_hal.h"
#include "bpu_fixed.h"

struct bpu_hal {
    uint32_t cfg, load_sel, sigma, ld_count, clipped, reads, status;
    uint8_t  w_p[32][32], w_m[32][32]; uint32_t mask[32]; int16_t act[32]; int16_t eta[2048];
    bpu_xoshiro_t gen[64];
    uint64_t res[32]; int res_mode_at_run;
};
static struct bpu_hal *M(bpu_hal_t *h) { return h; }

bpu_hal_t *bpu_hal_model_open(void) {
    struct bpu_hal *m = calloc(1, sizeof *m);
    for (int r = 0; r < 32; r++) m->mask[r] = 0xFFFFFFFFu;
    return m;
}
static void model_run(struct bpu_hal *m) {
    unsigned S = m->cfg & 31, two = (m->cfg >> 5) & 1, prng = (m->cfg >> 6) & 1, rmode = (m->cfg >> 8) & 3;
    if (prng) {                                     /* fill phase: 64 generators, one sample per row */
        for (int r = 0; r < 32; r++) for (int c = 0; c < 32; c++) for (int d = 0; d < 2; d++)
            m->eta[(r * 32 + c) * 2 + d] = (int16_t)bpu_prng_eta(&m->gen[2 * c + d], m->sigma & 0xFFFF);
    }
    int64_t Y[32] = {0}; int32_t Ip[32] = {0}, Im[32] = {0};
    for (unsigned p = 0; p < (two ? 2u : 1u); p++) {
        memset(Ip, 0, sizeof Ip); memset(Im, 0, sizeof Im);
        for (int r = 0; r < 32; r++) {
            int32_t x = m->act[r], d;
            if (!two) d = (int8_t)(x & 0xFF); else if (p == 0) d = x & 0xFF; else d = (int8_t)((x >> 8) & 0xFF);
            for (int c = 0; c < 32; c++) {
                if (!(m->mask[r] >> c & 1)) continue;
                uint32_t gp = bpu_noisy_conductance_q(bpu_conductance_q(m->w_p[r][c]), m->eta[(r * 32 + c) * 2]);
                uint32_t gm = bpu_noisy_conductance_q(bpu_conductance_q(m->w_m[r][c]), m->eta[(r * 32 + c) * 2 + 1]);
                Ip[c] += d * (int32_t)gp; Im[c] += d * (int32_t)gm;
            }
        }
        for (int c = 0; c < 32; c++) Y[c] += ((int64_t)Ip[c] - Im[c]) << (8 * p);
    }
    for (int c = 0; c < 32; c++) {
        int64_t q = bpu_rne_shift(Y[c], S);
        int32_t y = bpu_sat8(q);
        if (q < -128 || q > 127) m->clipped++;
        if (rmode == RES_VERIFY) m->res[c] = (uint64_t)(uint32_t)Ip[c] | ((uint64_t)(uint32_t)Im[c] << 32);
        else m->res[c] = ((uint64_t)(uint8_t)y << 56) | ((uint64_t)Y[c] & ((1ull << 40) - 1));
    }
    m->res_mode_at_run = (int)rmode; m->reads++; m->status = 2;
}
uint32_t bpu_reg_read(bpu_hal_t *h, uint32_t off) {
    struct bpu_hal *m = M(h);
    switch (off) {
        case REG_STATUS: return m->status; case REG_CONFIG: return m->cfg; case REG_LOAD_SEL: return m->load_sel;
        case REG_SIGMA: return m->sigma; case REG_LD_COUNT: return m->ld_count; case REG_CLIPPED: return m->clipped;
        case REG_READS: return m->reads; case REG_VERSION: return BPU_VERSION_ID; default: return 0xDEAD0000u;
    }
}
void bpu_reg_write(bpu_hal_t *h, uint32_t off, uint32_t v) {
    struct bpu_hal *m = M(h);
    switch (off) {
        case REG_CTRL: if (v & 1) { m->status = 0; model_run(m); } if (v & 2) m->clipped = 0; if (v & 8) m->ld_count = 0; break;
        case REG_CONFIG: m->cfg = v; break;
        case REG_LOAD_SEL: m->load_sel = v; m->ld_count = 0; break;
        case REG_SIGMA: m->sigma = v; break;
        default: break;
    }
}
int bpu_load(bpu_hal_t *h, int sel, const uint64_t *beats, size_t n) {
    struct bpu_hal *m = M(h);
    bpu_reg_write(h, REG_LOAD_SEL, (uint32_t)sel);
    for (size_t b = 0; b < n; b++, m->ld_count++) {
        uint64_t w = beats[b];
        switch (sel) {
            case SEL_WEIGHTS: for (int j = 0; j < 4; j++) { size_t k = 4 * b + j; m->w_p[k >> 5][k & 31] = (uint8_t)(w >> (16 * j)); m->w_m[k >> 5][k & 31] = (uint8_t)(w >> (16 * j + 8)); } break;
            case SEL_MASK: m->mask[2 * b] = (uint32_t)w; m->mask[2 * b + 1] = (uint32_t)(w >> 32); break;
            case SEL_ACT: for (int j = 0; j < 4; j++) m->act[4 * b + j] = (int16_t)(w >> (16 * j)); break;
            case SEL_NOISE: for (int j = 0; j < 4; j++) m->eta[4 * b + j] = (int16_t)(w >> (16 * j)); break;
            case SEL_PRNG: m->gen[b >> 2].s[b & 3] = (uint32_t)w; break;
            default: return -1;
        }
    }
    return 0;
}
int bpu_results(bpu_hal_t *h, uint64_t *beats, size_t n) {
    struct bpu_hal *m = M(h);
    unsigned rmode = (m->cfg >> 8) & 3;
    if (rmode == RES_NOISE) {
        if (n != 512) return -1;
        for (size_t b = 0; b < 512; b++) { beats[b] = 0; for (int j = 0; j < 4; j++) beats[b] |= (uint64_t)(uint16_t)m->eta[4 * b + j] << (16 * j); }
        return 0;
    }
    if (n != 32) return -1;
    memcpy(beats, m->res, sizeof m->res);
    return 0;
}
void bpu_hal_model_close(bpu_hal_t *h) { free(h); }
