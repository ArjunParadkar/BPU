# Pinned tool versions

| tool | version | used for |
|---|---|---|
| OSS CAD Suite | 2026-11-01 build (`~/.local/opt/oss-cad-suite`) | Verilator 5.053 (cocotb + gate tb), Icarus 13 (lint only: its bundled libc cannot load the venv's libpython for cocotb), Yosys (synthesisability check) |
| cocotb | 2.1.0 | unit / core / AXI tests |
| Python | 3.12.13 (uv), numpy 2.5.4, scipy 1.18.1, pytest 9.2.0 | reference, generator, harness (`fpga/requirements.txt`) |
| gcc | system (14.x) | PS firmware host build |
| Vivado / Vitis | **2024.2 intended** (board file `kv260_som` 1.4, axi_dma 7.1, zynq_ultra_ps_e 3.5) | bitstream -- not run yet |
| Kria image | Ubuntu 22.04 for Kria (xmutil, u-dma-buf) | board bring-up -- not run yet |
