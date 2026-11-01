/* runner.c -- host runner / on-target runner: load golden vectors (the gen_vectors.py hex set), run the tile,
 * diff against the expectation bit for bit, report.  Same binary on the board ("mmap") and on the host ("model"). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bpu_tile.h"

static uint64_t *read_hex(const char *dir, const char *name, size_t *n) {
    char p[1024]; snprintf(p, sizeof p, "%s/%s", dir, name);
    FILE *f = fopen(p, "r"); if (!f) { perror(p); exit(2); }
    size_t cap = 1 << 16, k = 0; uint64_t *v = malloc(cap * 8); char line[64];
    while (fgets(line, sizeof line, f)) { if (k == cap) { cap *= 2; v = realloc(v, cap * 8); } v[k++] = strtoull(line, NULL, 16); }
    fclose(f); *n = k; return v;
}
int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <model|mmap> <vector dir> [n]\n", argv[0]); return 2; }
    bpu_hal_t *h = bpu_hal_open(argv[1]); if (!h) return 1;
    bpu_tile_t t; if (bpu_tile_init(&t, h)) { fprintf(stderr, "tile VERSION mismatch\n"); return 1; }
    size_t nw, nm, nc, na, nn, ne;
    uint64_t *wset = read_hex(argv[2], "wset.hex", &nw), *mask = read_hex(argv[2], "mask.hex", &nm), *cfg = read_hex(argv[2], "cfg.hex", &nc),
             *act = read_hex(argv[2], "act.hex", &na), *noise = read_hex(argv[2], "noise.hex", &nn), *exp = read_hex(argv[2], "expect.hex", &ne);
    size_t n = argc > 3 ? (size_t)atol(argv[3]) : nc; if (n > nc) n = nc;
    long mism = 0, shown = 0, cur = -1;
    for (size_t i = 0; i < n; i++) {
        long ws = (long)(cfg[i] & 0xFFFF);
        if (ws != cur) { bpu_load(h, SEL_WEIGHTS, wset + ws * 256, 256); bpu_load(h, SEL_MASK, mask + ws * 16, 16); cur = ws; }
        bpu_load(h, SEL_ACT, act + i * 8, 8); bpu_load(h, SEL_NOISE, noise + i * 512, 512);
        t.adc_shift = (cfg[i] >> 16) & 31; t.two_pass = (cfg[i] >> 21) & 1; t.mode_prng = 0; t.result_mode = RES_NORMAL; bpu_tile_configure(&t);
        uint64_t out[32]; if (bpu_tile_read(&t, out)) { fprintf(stderr, "read %zu timed out\n", i); return 1; }
        for (int c = 0; c < 32; c++) if (out[c] != exp[i * 32 + c]) { mism++; if (shown++ < 20) printf("MISMATCH vec=%zu col=%d got=%016llx exp=%016llx\n", i, c, (unsigned long long)out[c], (unsigned long long)exp[i * 32 + c]); }
    }
    printf("%s vectors=%zu columns=%zu mismatches=%ld clipped=%u reads=%u\n", mism ? "GATE FAIL" : "GATE PASS", n, n * 32, mism,
           bpu_reg_read(h, REG_CLIPPED), bpu_reg_read(h, REG_READS));
    bpu_hal_close(h);
    return mism ? 1 : 0;
}
