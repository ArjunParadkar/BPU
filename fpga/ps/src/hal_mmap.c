/* hal_mmap.c -- the board HAL for Linux on the Kria PS (Ubuntu / PetaLinux): registers through /dev/mem,
 * streams through an AXI DMA (xilinx axi_dma IP, simple mode, MM2S for loads, S2MM for results) using a
 * physically contiguous buffer from /dev/udmabuf0 (or a reserved-memory u-dma-buf).  Addresses come from the
 * block design in fpga/scripts/vivado_bd.tcl; override with BPU_TILE_BASE / BPU_DMA_BASE / BPU_BUF_PHYS.
 * NOT YET RUN ON HARDWARE (no board on the bench, 2026-11-01): see docs/RESULTS.md. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include "bpu_hal.h"

#define DMA_MM2S_CR 0x00
#define DMA_MM2S_SR 0x04
#define DMA_MM2S_SA 0x18
#define DMA_MM2S_LEN 0x28
#define DMA_S2MM_CR 0x30
#define DMA_S2MM_SR 0x34
#define DMA_S2MM_DA 0x48
#define DMA_S2MM_LEN 0x58

struct bpu_hal {
    volatile uint32_t *tile; volatile uint32_t *dma; volatile uint64_t *buf; uint64_t buf_phys; size_t buf_len; int fd_mem, fd_buf;
};
static uint64_t env_u64(const char *n, uint64_t d) { const char *s = getenv(n); return s ? strtoull(s, NULL, 0) : d; }

bpu_hal_t *bpu_hal_mmap_open(void) {
    struct bpu_hal *h = calloc(1, sizeof *h);
    uint64_t tile_base = env_u64("BPU_TILE_BASE", 0xA0000000ull), dma_base = env_u64("BPU_DMA_BASE", 0xA0010000ull);
    h->fd_mem = open("/dev/mem", O_RDWR | O_SYNC);
    if (h->fd_mem < 0) { perror("/dev/mem"); free(h); return NULL; }
    h->tile = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, h->fd_mem, tile_base);
    h->dma  = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, h->fd_mem, dma_base);
    h->buf_len = 1 << 16;
    h->fd_buf = open("/dev/udmabuf0", O_RDWR | O_SYNC);
    if (h->fd_buf < 0) { perror("/dev/udmabuf0 (load the u-dma-buf module, see scripts/board_setup.sh)"); free(h); return NULL; }
    h->buf = mmap(NULL, h->buf_len, PROT_READ | PROT_WRITE, MAP_SHARED, h->fd_buf, 0);
    h->buf_phys = env_u64("BPU_BUF_PHYS", 0);
    if (!h->buf_phys) {                      /* u-dma-buf exports it in sysfs */
        FILE *f = fopen("/sys/class/u-dma-buf/udmabuf0/phys_addr", "r");
        if (f) { if (fscanf(f, "%lx", &h->buf_phys) != 1) h->buf_phys = 0; fclose(f); }
    }
    if (h->tile == MAP_FAILED || h->dma == MAP_FAILED || h->buf == MAP_FAILED || !h->buf_phys) { fprintf(stderr, "hal_mmap: mapping failed\n"); free(h); return NULL; }
    h->dma[DMA_MM2S_CR / 4] = 4; h->dma[DMA_S2MM_CR / 4] = 4;      /* reset both channels */
    while ((h->dma[DMA_MM2S_CR / 4] & 4) || (h->dma[DMA_S2MM_CR / 4] & 4)) ;
    return h;
}
void bpu_hal_mmap_close(bpu_hal_t *h) { if (!h) return; munmap((void *)h->buf, h->buf_len); close(h->fd_buf); close(h->fd_mem); free(h); }
uint32_t bpu_reg_read(bpu_hal_t *h, uint32_t off) { return h->tile[off / 4]; }
void bpu_reg_write(bpu_hal_t *h, uint32_t off, uint32_t v) { h->tile[off / 4] = v; }

static int dma_wait(volatile uint32_t *sr) {
    for (long i = 0; i < 100000000L; i++) { uint32_t s = *sr; if (s & 0x1000) return 0; if (s & 0x70) return -1; }
    return -2;
}
int bpu_load(bpu_hal_t *h, int sel, const uint64_t *beats, size_t n) {
    if (n * 8 > h->buf_len) return -1;
    bpu_reg_write(h, REG_LOAD_SEL, (uint32_t)sel);
    memcpy((void *)h->buf, beats, n * 8);
    __sync_synchronize();
    h->dma[DMA_MM2S_CR / 4] = 0x1001;                                 /* run, IOC irq enable (polled) */
    h->dma[DMA_MM2S_SA / 4] = (uint32_t)h->buf_phys; h->dma[(DMA_MM2S_SA + 4) / 4] = (uint32_t)(h->buf_phys >> 32);
    h->dma[DMA_MM2S_LEN / 4] = (uint32_t)(n * 8);
    int rc = dma_wait(&h->dma[DMA_MM2S_SR / 4]);
    h->dma[DMA_MM2S_SR / 4] = 0x1000;
    return rc;
}
int bpu_results(bpu_hal_t *h, uint64_t *beats, size_t n) {
    uint64_t phys = h->buf_phys + h->buf_len / 2;
    volatile uint64_t *dst = h->buf + h->buf_len / 16;
    h->dma[DMA_S2MM_CR / 4] = 0x1001;
    h->dma[DMA_S2MM_DA / 4] = (uint32_t)phys; h->dma[(DMA_S2MM_DA + 4) / 4] = (uint32_t)(phys >> 32);
    h->dma[DMA_S2MM_LEN / 4] = (uint32_t)(n * 8);
    bpu_reg_write(h, REG_CTRL, 4);                                      /* emit */
    int rc = dma_wait(&h->dma[DMA_S2MM_SR / 4]);
    h->dma[DMA_S2MM_SR / 4] = 0x1000;
    __sync_synchronize();
    memcpy(beats, (const void *)dst, n * 8);
    return rc;
}
