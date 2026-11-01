/* bpu_tile.h -- tile sequencer + write-verify on top of the HAL. */
#ifndef BPU_TILE_H
#define BPU_TILE_H
#include "bpu_hal.h"
#include "bpu_fixed.h"
typedef struct {
    bpu_hal_t *h;
    uint8_t  w_p[BPU_ROWS][BPU_COLS], w_m[BPU_ROWS][BPU_COLS];
    uint32_t mask[BPU_ROWS];
    unsigned adc_shift, two_pass, mode_prng, result_mode;
    uint32_t sigma_q;
} bpu_tile_t;
typedef struct { int iterations; long pulses; int converged; int32_t max_abs_err_s; } bpu_wv_report_t;

int  bpu_tile_init(bpu_tile_t *t, bpu_hal_t *h);
int  bpu_tile_load_weights(bpu_tile_t *t);                      /* w_p/w_m/mask -> PL */
int  bpu_tile_load_act(bpu_tile_t *t, const int16_t x[BPU_ROWS]);
int  bpu_tile_load_noise(bpu_tile_t *t, const int16_t eta[2048]);
int  bpu_tile_seed_prng(bpu_tile_t *t, uint64_t seed64);
int  bpu_tile_configure(bpu_tile_t *t);
int  bpu_tile_read(bpu_tile_t *t, uint64_t out[BPU_COLS]);      /* start, wait done, fetch 32 result words */
int  bpu_tile_read_noise_dump(bpu_tile_t *t, int16_t eta[2048]);
/* write-verify one word line (all 32 columns of one device array) toward targets w_target_s (Q1.15);
 * noise for the verify reads comes from the PL's PRNG (seeded), c2c from c2c_gen.  dev: 0 = G+, 1 = G-. */
int  bpu_write_verify_row(bpu_tile_t *t, int dev, int row, const int32_t w_target_s[BPU_COLS], int32_t w_s[BPU_COLS],
                          const uint32_t gain_q[BPU_COLS], bpu_xoshiro_t *c2c_gen, uint32_t sigma_c2c_q, bpu_wv_report_t *rep);
#endif
