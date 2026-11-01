#include <string.h>
#include <stdio.h>
#include "bpu_tile.h"

int bpu_tile_init(bpu_tile_t *t, bpu_hal_t *h) {
    memset(t, 0, sizeof *t); t->h = h;
    for (int r = 0; r < BPU_ROWS; r++) t->mask[r] = 0xFFFFFFFFu;
    return bpu_reg_read(h, REG_VERSION) == BPU_VERSION_ID ? 0 : -1;
}
int bpu_tile_configure(bpu_tile_t *t) {
    bpu_reg_write(t->h, REG_CONFIG, (t->adc_shift & 31) | (t->two_pass << 5) | (t->mode_prng << 6) | ((t->result_mode & 3) << 8));
    bpu_reg_write(t->h, REG_SIGMA, t->sigma_q & 0xFFFF);
    return 0;
}
int bpu_tile_load_weights(bpu_tile_t *t) {
    uint64_t beats[256];
    for (int k = 0; k < 1024; k += 4) {
        uint64_t b = 0;
        for (int j = 0; j < 4; j++) {
            int r = (k + j) >> 5, c = (k + j) & 31;
            b |= ((uint64_t)t->w_p[r][c] | ((uint64_t)t->w_m[r][c] << 8)) << (16 * j);
        }
        beats[k >> 2] = b;
    }
    if (bpu_load(t->h, SEL_WEIGHTS, beats, 256)) return -1;
    uint64_t mb[16];
    for (int b = 0; b < 16; b++) mb[b] = (uint64_t)t->mask[2 * b] | ((uint64_t)t->mask[2 * b + 1] << 32);
    return bpu_load(t->h, SEL_MASK, mb, 16);
}
int bpu_tile_load_act(bpu_tile_t *t, const int16_t x[BPU_ROWS]) {
    uint64_t beats[8];
    for (int b = 0; b < 8; b++) {
        beats[b] = 0;
        for (int j = 0; j < 4; j++) beats[b] |= (uint64_t)(uint16_t)x[4 * b + j] << (16 * j);
    }
    return bpu_load(t->h, SEL_ACT, beats, 8);
}
int bpu_tile_load_noise(bpu_tile_t *t, const int16_t eta[2048]) {
    static uint64_t beats[512];
    for (int b = 0; b < 512; b++) {
        beats[b] = 0;
        for (int j = 0; j < 4; j++) beats[b] |= (uint64_t)(uint16_t)eta[4 * b + j] << (16 * j);
    }
    return bpu_load(t->h, SEL_NOISE, beats, 512);
}
int bpu_tile_seed_prng(bpu_tile_t *t, uint64_t seed64) {
    uint64_t beats[256];
    for (unsigned m = 0; m < BPU_N_GEN; m++) {
        bpu_xoshiro_t g; bpu_prng_seed(&g, seed64, m);
        for (int j = 0; j < 4; j++) beats[4 * m + j] = g.s[j];
    }
    return bpu_load(t->h, SEL_PRNG, beats, 256);
}
int bpu_tile_read(bpu_tile_t *t, uint64_t out[BPU_COLS]) {
    bpu_reg_write(t->h, REG_CTRL, 1);
    for (long i = 0; i < 1000000; i++)
        if (bpu_reg_read(t->h, REG_STATUS) & 2) { bpu_reg_write(t->h, REG_CTRL, 4); return bpu_results(t->h, out, BPU_COLS); }
    return -2;
}
int bpu_tile_read_noise_dump(bpu_tile_t *t, int16_t eta[2048]) {
    static uint64_t beats[512];
    unsigned saved = t->result_mode; t->result_mode = RES_NOISE; bpu_tile_configure(t);
    bpu_reg_write(t->h, REG_CTRL, 4);
    int rc = bpu_results(t->h, beats, 512);
    t->result_mode = saved; bpu_tile_configure(t);
    for (int b = 0; b < 512; b++) for (int j = 0; j < 4; j++) eta[4 * b + j] = (int16_t)(beats[b] >> (16 * j));
    return rc;
}

/* ---- write-verify (contract section 6), one word line of one device array ---- */
static int verify_row_sum(bpu_tile_t *t, int dev, int row, int64_t sum[BPU_COLS]) {
    int16_t x[BPU_ROWS] = {0}; x[row] = BPU_VERIFY_CODE;
    unsigned saved_mode = t->result_mode, saved_two = t->two_pass;
    t->result_mode = RES_VERIFY; t->two_pass = 0; bpu_tile_configure(t);
    if (bpu_tile_load_act(t, x)) return -1;
    memset(sum, 0, sizeof(int64_t) * BPU_COLS);
    for (int n = 0; n < BPU_VERIFY_READS; n++) {
        uint64_t res[BPU_COLS];
        if (bpu_tile_read(t, res)) return -1;
        for (int c = 0; c < BPU_COLS; c++) sum[c] += (int32_t)(dev ? (res[c] >> 32) : (res[c] & 0xFFFFFFFFu));
    }
    t->result_mode = saved_mode; t->two_pass = saved_two; bpu_tile_configure(t);
    return 0;
}
int bpu_write_verify_row(bpu_tile_t *t, int dev, int row, const int32_t w_target_s[BPU_COLS], int32_t w_s[BPU_COLS],
                         const uint32_t gain_q[BPU_COLS], bpu_xoshiro_t *c2c_gen, uint32_t sigma_c2c_q, bpu_wv_report_t *rep) {
    uint8_t (*wq)[BPU_COLS] = dev ? t->w_m : t->w_p;
    long pulses = 0; int it;
    rep->converged = 0; rep->max_abs_err_s = 0;
    for (it = 1; it <= BPU_MAX_ITER; it++) {
        for (int c = 0; c < BPU_COLS; c++) wq[row][c] = (uint8_t)bpu_ws_to_wq(w_s[c]);
        if (bpu_tile_load_weights(t)) return -1;
        int64_t sum[BPU_COLS];
        if (verify_row_sum(t, dev, row, sum)) return -1;
        int todo_any = 0; int32_t n[BPU_COLS]; int32_t maxerr = 0;
        for (int c = 0; c < BPU_COLS; c++) {
            int32_t meas = bpu_w_meas_from_sum(sum[c], BPU_VERIFY_READS);
            int32_t err = w_target_s[c] - meas;
            int32_t ae = err < 0 ? -err : err; if (ae > maxerr) maxerr = ae;
            unsigned k = bpu_lut_index(w_s[c]);
            uint32_t st = bpu_step_up[k] > bpu_step_dn[k] ? bpu_step_up[k] : bpu_step_dn[k];
            int32_t res = (int32_t)(((uint64_t)st * gain_q[c]) >> (BPU_GAIN_FRAC + 1));
            int32_t thr = res > BPU_TOL_Q ? res : BPU_TOL_Q;
            int todo = (ae > thr) && (t->mask[row] >> c & 1);
            n[c] = 0;
            if (todo) {
                todo_any = 1;
                int64_t aim = (int64_t)w_s[c] + (((int64_t)BPU_DAMP_Q * err) >> BPU_GAIN_FRAC);
                if (aim < 0) aim = 0;
                if (aim > BPU_WS_ONE) aim = BPU_WS_ONE;
                n[c] = bpu_pulses_for(w_s[c], (int32_t)aim);
                if (n[c] == 0) n[c] = err > 0 ? 1 : -1;
            }
        }
        rep->max_abs_err_s = maxerr;
        if (!todo_any) { rep->converged = 1; it--; break; }
        for (int c = 0; c < BPU_COLS; c++) {
            pulses += n[c] < 0 ? -n[c] : n[c];
            w_s[c] = bpu_pulse(w_s[c], n[c], gain_q[c], c2c_gen, sigma_c2c_q);
        }
    }
    for (int c = 0; c < BPU_COLS; c++) wq[row][c] = (uint8_t)bpu_ws_to_wq(w_s[c]);
    bpu_tile_load_weights(t);
    rep->iterations = it > BPU_MAX_ITER ? BPU_MAX_ITER : it; rep->pulses = pulses;
    return 0;
}
