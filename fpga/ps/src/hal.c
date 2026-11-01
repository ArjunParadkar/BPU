#include <string.h>
#include "bpu_hal.h"
bpu_hal_t *bpu_hal_model_open(void); void bpu_hal_model_close(bpu_hal_t *);
#ifdef BPU_HAVE_MMAP
bpu_hal_t *bpu_hal_mmap_open(void); void bpu_hal_mmap_close(bpu_hal_t *);
#endif
static int is_model;
bpu_hal_t *bpu_hal_open(const char *spec) {
#ifdef BPU_HAVE_MMAP
    if (spec && strcmp(spec, "mmap") == 0) { is_model = 0; return bpu_hal_mmap_open(); }
#endif
    (void)spec; is_model = 1; return bpu_hal_model_open();
}
void bpu_hal_close(bpu_hal_t *h) {
#ifdef BPU_HAVE_MMAP
    if (!is_model) { bpu_hal_mmap_close(h); return; }
#endif
    bpu_hal_model_close(h);
}
