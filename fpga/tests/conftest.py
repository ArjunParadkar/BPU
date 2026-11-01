import os, pathlib, shutil, subprocess, sys
import pytest

FPGA = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FPGA.parent))          # bpusim (the reference) from the repo root, no install needed
HDL = FPGA / "hdl"
SOURCES = [HDL / f for f in ("bpu_pkg.sv", "wl_dac.sv", "gcond.sv", "xbar_tile.sv", "col_integrate.sv", "shift_add.sv",
                              "sar_adc.sv", "eta_prng.sv", "tile_core.sv", "bpu_tile_axi.sv")]
OSS = pathlib.Path.home() / ".local/opt/oss-cad-suite/bin"


def tool(name):
    p = shutil.which(name) or (str(OSS / name) if (OSS / name).exists() else None)
    if p is None:
        pytest.skip(f"{name} not installed (OSS CAD Suite expected in {OSS})")
    return p


@pytest.fixture(scope="session")
def zlut():
    f = HDL / "zlut_q13.mem"
    if not f.exists():
        subprocess.run([os.environ.get("PYTHON", "python"), str(FPGA / "scripts/gen_zlut.py"), str(f)], check=True)
    return f


def pytest_addoption(parser):
    parser.addoption("--gate-vectors", type=int, default=int(os.environ.get("BPU_GATE_N", "512")),
                     help="random vectors in the bit-exact gate (10000 = the registered full run)")
    parser.addoption("--sim", default=os.environ.get("BPU_SIM", "verilator"), help="verilator | icarus for the gate")
