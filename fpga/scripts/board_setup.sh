#!/usr/bin/env bash
# board_setup.sh -- on the Kria (Ubuntu 22.04 / 24.04 for Kria, or PetaLinux): load the bitstream as a Kria app and
# give the firmware a DMA buffer.  Run as root.  NOT YET RUN (no board on the bench, 2026-11-01).
#   1. copy  <out_dir>/bpu_kv260.runs/impl_1/bd_wrapper.bit, the .xsa-derived bd.dtbo (xsct: createdts / dtc),
#      and scripts/shell.json into /lib/firmware/xilinx/bpu_tile/   (bit must be renamed bpu_tile.bit.bin via bootgen)
#   2. sudo xmutil unloadapp && sudo xmutil loadapp bpu_tile
#   3. this script loads u-dma-buf and checks the register ID
set -euo pipefail
modprobe u-dma-buf udmabuf0=1048576 || insmod ./u-dma-buf.ko udmabuf0=1048576
ls /dev/udmabuf0 /sys/class/u-dma-buf/udmabuf0/phys_addr
ID=$(devmem2 0xA0000020 w 2>/dev/null | tail -1 || busybox devmem 0xA0000020)
echo "tile VERSION register: $ID (expect 0xB9000001)"
