#!/usr/bin/env python
"""Emit the Gaussian inverse-CDF ROM (contract section 2/5B) as a $readmemh file from bpusim.fixed."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from bpusim import fixed as F
out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parents[1] / "hdl" / "zlut_q13.mem"
lut = F.gaussian_lut()
out.write_text("".join(f"{int(v) & 0xFFFF:04x}\n" for v in lut))
print(out, len(lut), "entries")
