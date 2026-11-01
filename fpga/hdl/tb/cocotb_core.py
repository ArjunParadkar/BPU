"""cocotb tests of tile_core: integrators + shift-add + ADC through the whole read (small random set, both
noise modes, verify result mode, noise dump) against bpusim.fixed, bit-exact."""
import os, pathlib, sys
import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ReadOnly

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[2] / "vectors"))
from bpusim import fixed as F
import gen_vectors as G

N = int(os.environ.get("BPU_CORE_N", "24"))


def to_signed(v, bits):
    v = int(v) & ((1 << bits) - 1)
    return v - (1 << bits) if v >> (bits - 1) else v


async def load(dut, sel, words):
    dut.ld_sel.value = sel; dut.ld_restart.value = 1
    await RisingEdge(dut.clk); dut.ld_restart.value = 0
    for w in words:
        dut.ld_valid.value = 1; dut.ld_data.value = int(w)
        await RisingEdge(dut.clk)
    dut.ld_valid.value = 0


async def run_read(dut):
    dut.start.value = 1; await RisingEdge(dut.clk); dut.start.value = 0
    for _ in range(400):
        await RisingEdge(dut.clk); await ReadOnly()
        if int(dut.done.value): break
    else:
        raise AssertionError("tile_core did not finish")
    await RisingEdge(dut.clk)


async def read_results(dut, n):
    out = []
    for a in range(n):
        dut.res_addr.value = a; await RisingEdge(dut.clk); await ReadOnly()
        out.append(int(dut.res_data.value)); await RisingEdge(dut.clk)
    return out


async def setup(dut):
    cocotb.start_soon(Clock(dut.clk, 10, "ns").start())
    for s in ("ld_valid", "ld_restart", "start", "clear_clipped", "mode_prng", "result_mode", "sigma_q", "res_addr", "two_pass", "adc_shift"):
        getattr(dut, s).value = 0
    dut.rst_n.value = 0
    for _ in range(3): await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    for _ in range(2): await RisingEdge(dut.clk)


async def program(dut, v):
    await load(dut, 0, G.pack_weights(v["wp"], v["wm"]))
    await load(dut, 1, G.pack_mask(v["mask"]))
    await load(dut, 2, G.pack_act(v["x"]))
    dut.adc_shift.value = int(v["S"]); dut.two_pass.value = int(v["passes"] == 2)


@cocotb.test()
async def stream_mode_random_and_corners(dut):
    await setup(dut)
    vecs = G.vectors(777, N) + G.corner_vectors()
    for v in vecs:
        await program(dut, v)
        await load(dut, 3, G.pack_noise(v["eta"]))
        dut.result_mode.value = 0
        await run_read(dut)
        res = await read_results(dut, 32)
        for c, w in enumerate(res):
            y, Y = G.unpack_result(w)
            assert (y, Y) == (int(v["y"][c]), int(v["Y"][c])), f"{v.get('name', v['seed'])} col {c}: got y={y} Y={Y} want y={v['y'][c]} Y={v['Y'][c]}"
    # the DUT's clip counter equals the reference's
    assert int(dut.clipped_count.value) == sum(int(v["clipped"]) for v in vecs)
    assert int(dut.read_count.value) == len(vecs)


@cocotb.test()
async def verify_result_mode_gives_last_pass_currents(dut):
    await setup(dut)
    for v in G.vectors(31, 6):
        await program(dut, v)
        await load(dut, 3, G.pack_noise(v["eta"]))
        dut.result_mode.value = 1
        await run_read(dut)
        res = await read_results(dut, 32)
        for c, w in enumerate(res):
            ip = to_signed(w & 0xFFFFFFFF, 32); im = to_signed(w >> 32, 32)
            assert (ip, im) == (int(v["Ip"][c]), int(v["Im"][c])), f"seed {v['seed']} col {c}: got {ip},{im} want {v['Ip'][c]},{v['Im'][c]}"
    # a verify read proper: one-hot row 5 at code 64 -> I+ = 64 * Ghat exactly
    v = G.vectors(31, 1)[0]
    x = np.zeros(32, np.int64); x[5] = F.VERIFY_CODE
    v = dict(v, x=x, passes=1)
    await program(dut, v)
    await load(dut, 3, G.pack_noise(v["eta"]))
    await run_read(dut)
    res = await read_results(dut, 32)
    want = F.verify_read(v["wp"], v["mask"][5], v["eta"].reshape(32, 32, 2)[5, :, 0], 5)
    for c, w in enumerate(res):
        assert to_signed(w & 0xFFFFFFFF, 32) == int(want[c])


@cocotb.test()
async def prng_mode_matches_reference_stream_and_output(dut):
    await setup(dut)
    seed, sigma = 0xBADC0DE, 0.05
    sigma_q = F.sigma_to_code(sigma)
    words = []
    for m in range(64):
        words += F.prng_seed_words(seed, m)
    await load(dut, 4, words)
    dut.sigma_q.value = sigma_q; dut.mode_prng.value = 1
    ref = F.EtaPRNG(seed, sigma_q)
    for v in G.vectors(99, 4):
        await program(dut, v)
        eta = ref.read_block()
        dut.result_mode.value = 0
        await run_read(dut)
        r = F.tile_read(v["wp"], v["wm"], v["mask"], v["x"], eta, v["S"], v["passes"])
        res = await read_results(dut, 32)
        for c, w in enumerate(res):
            assert G.unpack_result(w) == (int(r.y_adc[c]), int(r.Y[c])), f"prng vec seed {v['seed']} col {c}"
        # the noise the PL generated for this read == the reference stream, word for word
        dut.result_mode.value = 2
        dump = await read_results(dut, 512)
        got = np.array([to_signed(w >> (16 * k), 16) for w in dump for k in range(4)])
        assert np.array_equal(got, eta), "PL prng stream != reference EtaPRNG stream"
    # reproducibility: reseed, same reads -> identical outputs
    await load(dut, 4, words)
    ref2 = F.EtaPRNG(seed, sigma_q)
    dut.result_mode.value = 2
    v = G.vectors(99, 1)[0]
    await program(dut, v); await run_read(dut)
    dump = await read_results(dut, 512)
    got = np.array([to_signed(w >> (16 * k), 16) for w in dump for k in range(4)])
    assert np.array_equal(got, ref2.read_block())
