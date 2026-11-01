"""cocotb unit tests: every module against its bpusim.fixed micro-reference, bit-exact (gate test 1)."""
import os, pathlib, sys
import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer, ReadOnly

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
from bpusim import fixed as F

N = int(os.environ.get("BPU_UNIT_N", "2000"))


def s16(v):  return int(v) & 0xFFFF
def to_signed(v, bits):
    v = int(v) & ((1 << bits) - 1)
    return v - (1 << bits) if v >> (bits - 1) else v


# ------------------------------------------------------------------ wl_dac
@cocotb.test()
async def wl_dac_matches_reference(dut):
    rng = np.random.default_rng(1)
    for _ in range(N):
        two = int(rng.integers(0, 2)); p = int(rng.integers(0, 2)) if two else 0
        x = int(rng.integers(-32768, 32768)) if two else int(rng.integers(-128, 128))
        dut.x_code.value = s16(x); dut.two_pass.value = two; dut.pass_idx.value = p
        await Timer(1, "ns")
        want = int(F.dac_pass_codes(np.array([x]), 2 if two else 1)[p][0])
        got = to_signed(dut.d.value, 9)
        assert got == want, f"wl_dac x={x} two={two} pass={p}: got {got} want {want}"


# ------------------------------------------------------------------ gcond
@cocotb.test()
async def gcond_matches_reference(dut):
    rng = np.random.default_rng(2)
    cases = [(w, e) for w in (0, 1, 127, 254, 255) for e in (-16383, -1, 0, 1, 16383)]
    cases += [(int(rng.integers(0, 256)), int(rng.integers(-16383, 16384))) for _ in range(N)]
    for w, e in cases:
        dut.w_q.value = w; dut.eta_q.value = s16(e)
        await Timer(1, "ns")
        want = int(F.noisy_conductance_q(F.conductance_q(np.array([w])), np.array([e]))[0])
        got = int(dut.ghat.value)
        assert got == want, f"gcond w={w} eta={e}: got {got} want {want}"


# ------------------------------------------------------------------ sar_adc
@cocotb.test()
async def sar_adc_matches_reference(dut):
    cocotb.start_soon(Clock(dut.clk, 10, "ns").start())
    rng = np.random.default_rng(3)
    dut.start.value = 0; dut.shift.value = 0; dut.y.value = 0
    for _ in range(3): await RisingEdge(dut.clk)
    cases = []
    for S in (0, 1, 2, 3, 7, 16, 31):
        for Y in (0, 1, -1, 127, 128, -128, -129, (1 << 39) - 1, -(1 << 39), (3 << (max(S, 1) - 1)), -(3 << (max(S, 1) - 1)),
                  (5 << (max(S, 1) - 1)), 127 << S, 128 << S, -128 << S, -129 << S):
            cases.append((S, Y))
    for _ in range(N):
        S = int(rng.integers(0, 32)); mag = int(rng.integers(0, 40))
        Y = int(rng.integers(-(1 << mag), (1 << mag))) if mag else 0
        cases.append((S, Y))
    for S, Y in cases:
        dut.shift.value = S; dut.y.value = int(Y) & ((1 << 40) - 1); dut.start.value = 1
        await RisingEdge(dut.clk); dut.start.value = 0
        for _ in range(10):
            await RisingEdge(dut.clk)
            await ReadOnly()
            if int(dut.done.value): break
        else:
            raise AssertionError("sar_adc did not finish")
        q = int(F.rne_shift(np.array([Y]), S)[0])
        want = max(-128, min(127, q)); want_clip = int(q < -128 or q > 127)
        got = to_signed(dut.code.value, 8); clip = int(dut.clipped.value)
        assert got == want and clip == want_clip, f"sar_adc Y={Y} S={S}: got {got} clip {clip} want {want} clip {want_clip}"
        await RisingEdge(dut.clk)


# ------------------------------------------------------------------ eta_prng
@cocotb.test()
async def eta_prng_matches_reference(dut):
    cocotb.start_soon(Clock(dut.clk, 10, "ns").start())
    dut.ld_en.value = 0; dut.step.value = 0
    for seed, sigma in ((0xC0FFEE, 0.02), (1, 0.2), (0xDEADBEEFCAFEF00D, 0.1)):
        sigma_q = F.sigma_to_code(sigma)
        dut.sigma_q.value = sigma_q
        for m in range(64):
            for j, w in enumerate(F.prng_seed_words(seed, m)):
                dut.ld_en.value = 1; dut.ld_idx.value = m * 4 + j; dut.ld_word.value = w
                await RisingEdge(dut.clk)
        dut.ld_en.value = 0
        ref = F.EtaPRNG(seed, sigma_q)
        for read in range(3):
            block = ref.read_block().reshape(32, 32, 2)
            for r in range(32):
                dut.step.value = 1; await RisingEdge(dut.clk); dut.step.value = 0
                await RisingEdge(dut.clk); await RisingEdge(dut.clk); await ReadOnly()   # 2-cycle latency (sync ROM)
                got = int(dut.eta.value)
                for c in range(32):
                    for d in (0, 1):
                        g = to_signed(got >> (16 * (2 * c + d)), 16)
                        assert g == block[r, c, d], f"prng seed={seed:x} read={read} row={r} col={c} dev={d}: got {g} want {block[r,c,d]}"
                await RisingEdge(dut.clk)
