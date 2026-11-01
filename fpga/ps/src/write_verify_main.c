/* write_verify_main.c -- gate test 3: sweep target weights x sigma, program every row of one device array by
 * write-verify through the tile, report iterations / pulses / final band per (target, sigma).  JSON lines out. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bpu_tile.h"

int main(int argc, char **argv) {
    const char *hal = argc > 1 ? argv[1] : "model";
    uint64_t seed = argc > 2 ? strtoull(argv[2], NULL, 0) : 20261101ull;
    bpu_hal_t *h = bpu_hal_open(hal); if (!h) return 1;
    bpu_tile_t t; if (bpu_tile_init(&t, h)) return 1;
    const double targets[] = {0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0};
    const double sigmas[]  = {0.0, 0.02, 0.05, 0.1, 0.2};
    int fail = 0;
    for (size_t si = 0; si < sizeof sigmas / sizeof *sigmas; si++) {
        uint32_t sq = (uint32_t)(sigmas[si] * 16384 + 0.5);
        for (size_t ti = 0; ti < sizeof targets / sizeof *targets; ti++) {
            int32_t tgt = (int32_t)(targets[ti] * BPU_WS_ONE + 0.5);
            t.mode_prng = 1; t.sigma_q = sq; t.adc_shift = 0; bpu_tile_configure(&t);
            bpu_tile_seed_prng(&t, seed + 1000 * si + ti);
            bpu_xoshiro_t c2c; bpu_prng_seed(&c2c, seed ^ 0xC2C, 64 + (unsigned)(si * 16 + ti));
            bpu_xoshiro_t d2d; bpu_prng_seed(&d2d, seed ^ 0xD2D, 128 + (unsigned)(si * 16 + ti));
            long it_sum = 0, it_max = 0, pulses = 0; int conv = 0; int32_t worst = 0; int rows = 8;
            for (int r = 0; r < rows; r++) {
                int32_t w_s[32], wt[32]; uint32_t gain[32];
                for (int c = 0; c < 32; c++) {
                    w_s[c] = (int32_t)(bpu_xoshiro_next(&d2d) % (BPU_WS_ONE + 1)); wt[c] = tgt;
                    int32_t g = BPU_GAIN_ONE + bpu_prng_eta(&d2d, sq);               /* d2d gain ~ 1 + sigma N(0,1) */
                    gain[c] = (uint32_t)(g < BPU_GAIN_MIN ? BPU_GAIN_MIN : g);
                }
                bpu_wv_report_t rep;
                if (bpu_write_verify_row(&t, 0, r, wt, w_s, gain, &c2c, sq, &rep)) return 1;
                it_sum += rep.iterations; if (rep.iterations > it_max) it_max = rep.iterations; pulses += rep.pulses; conv += rep.converged;
                for (int c = 0; c < 32; c++) { int32_t e = w_s[c] - tgt; if (e < 0) e = -e; if (e > worst) worst = e; }
            }
            int band_ok = worst <= BPU_TOL_Q + (int32_t)(3 * sigmas[si] / 8.0 * BPU_WS_ONE / 2.83) + 1300;   /* tol + 3 SE of 8 averaged reads + one coarse pulse */
            int ok = (conv == rows) && band_ok;
            if (!ok) fail++;
            printf("{\"sigma\": %.3f, \"target\": %.3f, \"rows\": %d, \"converged_rows\": %d, \"iter_mean\": %.2f, \"iter_max\": %ld, \"pulses_per_device\": %.1f, \"worst_abs_err_w\": %.4f, \"pass\": %s}\n",
                   sigmas[si], targets[ti], rows, conv, (double)it_sum / rows, it_max, (double)pulses / (rows * 32), (double)worst / BPU_WS_ONE, ok ? "true" : "false");
        }
    }
    bpu_hal_close(h);
    return fail ? 1 : 0;
}
