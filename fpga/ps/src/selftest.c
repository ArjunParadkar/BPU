/* selftest.c -- prints the C implementation's answers for a few contract probes so the host test can diff them
 * against bpusim.fixed (PRNG stream, pulse law, pulses_for, w_meas, rne_shift). */
#include <stdio.h>
#include <stdlib.h>
#include "bpu_fixed.h"
int main(int argc, char **argv) {
    uint64_t seed = argc > 1 ? strtoull(argv[1], NULL, 0) : 0xC0FFEEull;
    uint32_t sq = argc > 2 ? (uint32_t)strtoul(argv[2], NULL, 0) : 328;
    bpu_xoshiro_t g[64]; for (unsigned m = 0; m < 64; m++) bpu_prng_seed(&g[m], seed, m);
    printf("prng");
    for (int r = 0; r < 32; r++) for (int c = 0; c < 32; c++) for (int d = 0; d < 2; d++) printf(" %d", bpu_prng_eta(&g[2 * c + d], sq));
    printf("\n");
    printf("rne");
    for (int s = 0; s < 32; s += 3) for (int64_t x = -300; x <= 300; x += 37) printf(" %lld", (long long)bpu_rne_shift(x, s));
    printf("\n");
    printf("pulses_for");
    for (int32_t w = 0; w <= BPU_WS_ONE; w += 1234) for (int32_t t2 = 0; t2 <= BPU_WS_ONE; t2 += 2345) printf(" %d", bpu_pulses_for(w, t2));
    printf("\n");
    printf("step");
    for (int32_t w = 0; w <= BPU_WS_ONE; w += 777) printf(" %d %d", bpu_step_q(w, 16384, 0, 1), bpu_step_q(w, 15000, -1000, -1));
    printf("\n");
    printf("wmeas");
    for (int64_t s = 0; s < 8 * 64 * 70000; s += 1234567) printf(" %d", bpu_w_meas_from_sum(s, 8));
    printf("\n");
    bpu_xoshiro_t c2c; bpu_prng_seed(&c2c, seed, 64);
    printf("pulse");
    int32_t w = 1000; for (int k = 0; k < 40; k++) { w = bpu_pulse(w, (k % 7) - 3, 16000, &c2c, sq); printf(" %d", w); }
    printf("\n");
    return 0;
}
