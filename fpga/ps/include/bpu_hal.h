/* bpu_hal.h -- what the firmware needs from the platform: the tile's registers, the load DMA, the result DMA. */
#ifndef BPU_HAL_H
#define BPU_HAL_H
#include <stdint.h>
#include <stddef.h>
enum { REG_CTRL = 0x00, REG_STATUS = 0x04, REG_CONFIG = 0x08, REG_LOAD_SEL = 0x0C, REG_SIGMA = 0x10,
       REG_LD_COUNT = 0x14, REG_CLIPPED = 0x18, REG_READS = 0x1C, REG_VERSION = 0x20 };
enum { SEL_WEIGHTS = 0, SEL_MASK = 1, SEL_ACT = 2, SEL_NOISE = 3, SEL_PRNG = 4 };
enum { RES_NORMAL = 0, RES_VERIFY = 1, RES_NOISE = 2 };
#define BPU_VERSION_ID 0xB9000001u

typedef struct bpu_hal bpu_hal_t;
bpu_hal_t *bpu_hal_open(const char *spec);          /* "mmap" (board) or "model" (host C model) */
void       bpu_hal_close(bpu_hal_t *h);
uint32_t   bpu_reg_read(bpu_hal_t *h, uint32_t off);
void       bpu_reg_write(bpu_hal_t *h, uint32_t off, uint32_t v);
int        bpu_load(bpu_hal_t *h, int sel, const uint64_t *beats, size_t n);     /* AXI-Stream in, via DMA */
int        bpu_results(bpu_hal_t *h, uint64_t *beats, size_t n);                 /* AXI-Stream out, via DMA */
#endif
