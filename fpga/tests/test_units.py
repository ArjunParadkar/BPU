"""Gate test 1: per-module cocotb unit tests (Icarus) against bpusim.fixed, bit-exact."""
import os, pathlib
import pytest
from cocotb_tools.runner import get_runner
from conftest import SOURCES, HDL, tool

BUILD = pathlib.Path(__file__).resolve().parent / "_build"
BUILD_ARGS = ["-g2012"] if os.environ.get("BPU_UNIT_SIM") == "icarus" else ["-Wno-fatal", "-Wno-WIDTHEXPAND", "-Wno-WIDTHTRUNC", "--x-assign", "fast", "--x-initial", "fast"]


def run_cocotb(toplevel, module, testcase, zlut, env=None):
    sim = os.environ.get("BPU_UNIT_SIM", "verilator")
    os.environ["PATH"] = str(pathlib.Path(tool("verilator" if sim == "verilator" else "iverilog")).parent) + os.pathsep + os.environ["PATH"]
    runner = get_runner(os.environ.get("BPU_UNIT_SIM", "verilator"))
    build_dir = BUILD / f"{os.environ.get('BPU_UNIT_SIM', 'verilator')}_{toplevel}"
    params = {"ZLUT_FILE": f'"{zlut}"'} if toplevel in ("tile_core", "eta_prng", "bpu_tile_axi") else {}
    runner.build(sources=SOURCES, hdl_toplevel=toplevel, build_dir=build_dir, always=False, parameters=params,
                 build_args=BUILD_ARGS, includes=[HDL], timescale=("1ns", "1ps"))
    return runner.test(hdl_toplevel=toplevel, test_module=module, testcase=testcase, build_dir=build_dir,
                       test_dir=HDL / "tb", extra_env=dict(env or {}), results_xml=str(build_dir / f"{testcase}.xml"))


@pytest.mark.parametrize("toplevel,testcase", [
    ("wl_dac", "wl_dac_matches_reference"),
    ("gcond", "gcond_matches_reference"),
    ("sar_adc", "sar_adc_matches_reference"),
    ("eta_prng", "eta_prng_matches_reference"),
])
def test_unit(toplevel, testcase, zlut):
    from cocotb_tools.check_results import get_results
    xml = run_cocotb(toplevel, "cocotb_units", testcase, zlut)
    total, failed = get_results(xml)
    assert total >= 1 and failed == 0, f"{toplevel}: {failed} of {total} cocotb tests failed (see {xml})"


@pytest.mark.parametrize("testcase", [
    "stream_mode_random_and_corners",
    "verify_result_mode_gives_last_pass_currents",
    "prng_mode_matches_reference_stream_and_output",
])
def test_core(testcase, zlut):
    from cocotb_tools.check_results import get_results
    xml = run_cocotb("tile_core", "cocotb_core", testcase, zlut)
    total, failed = get_results(xml)
    assert total >= 1 and failed == 0, f"tile_core/{testcase}: {failed} of {total} failed (see {xml})"


def test_axi_wrapper(zlut):
    """The AXI4-Lite / AXI4-Stream wrapper end to end (what the PS talks to)."""
    from cocotb_tools.check_results import get_results
    xml = run_cocotb("bpu_tile_axi", "cocotb_axi", "axi_two_vectors", zlut)
    total, failed = get_results(xml)
    assert total >= 1 and failed == 0
