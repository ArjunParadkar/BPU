/* measure.c -- board power through the KV260 carrier's INA260 (hwmon) while the tile runs a sustained workload.
 * Prints idle and loaded power (mean of N samples), so dynamic = loaded - idle.  Linux hwmon paths vary by image;
 * BPU_HWMON overrides the directory.  NOT YET RUN ON HARDWARE. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <time.h>
#include "bpu_tile.h"

static const char *find_ina260(char *buf, size_t n) {
    const char *env = getenv("BPU_HWMON"); if (env) return env;
    DIR *d = opendir("/sys/class/hwmon"); if (!d) return NULL;
    struct dirent *e;
    while ((e = readdir(d))) {
        char p[512]; snprintf(p, sizeof p, "/sys/class/hwmon/%s/name", e->d_name);
        FILE *f = fopen(p, "r"); if (!f) continue;
        char name[64] = {0}; if (fgets(name, sizeof name, f) && strstr(name, "ina260")) { fclose(f); snprintf(buf, n, "/sys/class/hwmon/%s", e->d_name); closedir(d); return buf; }
        fclose(f);
    }
    closedir(d); return NULL;
}
static double read_power_uW(const char *dir) {
    char p[512]; snprintf(p, sizeof p, "%s/power1_input", dir);
    FILE *f = fopen(p, "r"); if (!f) return -1; double v; if (fscanf(f, "%lf", &v) != 1) v = -1; fclose(f); return v;
}
static double sample(const char *dir, int n) { double s = 0; for (int i = 0; i < n; i++) { s += read_power_uW(dir); usleep(10000); } return s / n * 1e-6; }

int main(int argc, char **argv) {
    char dirbuf[256]; const char *dir = find_ina260(dirbuf, sizeof dirbuf);
    if (!dir) { fprintf(stderr, "no ina260 hwmon found\n"); return 2; }
    bpu_hal_t *h = bpu_hal_open(argc > 1 ? argv[1] : "mmap"); if (!h) return 1;
    bpu_tile_t t; if (bpu_tile_init(&t, h)) { fprintf(stderr, "tile not found\n"); return 1; }
    t.adc_shift = 18; t.mode_prng = 1; t.sigma_q = 328; bpu_tile_configure(&t); bpu_tile_seed_prng(&t, 1);
    for (int r = 0; r < 32; r++) for (int c = 0; c < 32; c++) { t.w_p[r][c] = (r * 7 + c) & 255; t.w_m[r][c] = (r * 3 + 5 * c) & 255; }
    bpu_tile_load_weights(&t);
    int16_t x[32]; for (int i = 0; i < 32; i++) x[i] = (int16_t)(i * 8 - 128); bpu_tile_load_act(&t, x);
    double idle = sample(dir, 200);
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    long reads = 0; double loaded = 0; int ns = 0;
    while (1) {
        uint64_t out[32];
        for (int i = 0; i < 1000; i++) { bpu_tile_read(&t, out); reads++; }
        loaded += read_power_uW(dir) * 1e-6; ns++;
        clock_gettime(CLOCK_MONOTONIC, &t1);
        if ((t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9 > 20.0) break;
    }
    double secs = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    loaded /= ns;
    printf("{\"idle_W\": %.4f, \"loaded_W\": %.4f, \"dynamic_W\": %.4f, \"reads\": %ld, \"reads_per_s\": %.1f, \"J_per_read_dynamic\": %.3e, \"note\": \"PS-driven loop: includes DMA + register polling overhead; the PL-only figure needs the hardware counter run\"}\n",
           idle, loaded, loaded - idle, reads, reads / secs, (loaded - idle) * secs / reads);
    bpu_hal_close(h);
    return 0;
}
