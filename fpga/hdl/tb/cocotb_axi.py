"""cocotb test of the AXI wrapper: registers, load stream, start, result stream; two vectors bit-exact."""
import pathlib, sys
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ReadOnly, Timer

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3])); sys.path.insert(0, str(HERE.parents[2] / "vectors"))
import gen_vectors as G


async def axil_write(dut, addr, data):
    dut.s_axil_awaddr.value = addr; dut.s_axil_awvalid.value = 1
    dut.s_axil_wdata.value = data; dut.s_axil_wstrb.value = 0xF; dut.s_axil_wvalid.value = 1
    dut.s_axil_bready.value = 1
    while True:
        await RisingEdge(dut.aclk)
        await ReadOnly()
        if int(dut.s_axil_awready.value): break
    await RisingEdge(dut.aclk)
    dut.s_axil_awvalid.value = 0; dut.s_axil_wvalid.value = 0
    while True:
        await ReadOnly()
        if int(dut.s_axil_bvalid.value): break
        await RisingEdge(dut.aclk)
    await RisingEdge(dut.aclk)


async def axil_read(dut, addr):
    dut.s_axil_araddr.value = addr; dut.s_axil_arvalid.value = 1; dut.s_axil_rready.value = 1
    while True:
        await RisingEdge(dut.aclk); await ReadOnly()
        if int(dut.s_axil_arready.value): break
    await RisingEdge(dut.aclk); dut.s_axil_arvalid.value = 0
    while True:
        await ReadOnly()
        if int(dut.s_axil_rvalid.value): break
        await RisingEdge(dut.aclk)
    v = int(dut.s_axil_rdata.value)
    await RisingEdge(dut.aclk)
    return v


async def axis_load(dut, sel, words):
    await axil_write(dut, 0x0C, sel)
    for w in words:
        dut.s_axis_tdata.value = int(w); dut.s_axis_tvalid.value = 1
        while True:                                   # the beat is taken at the edge where tvalid & tready
            await RisingEdge(dut.aclk); await ReadOnly()
            if int(dut.s_axis_tready.value): break
        await Timer(1, "ns")                          # leave the read-only phase before driving the next beat
    dut.s_axis_tvalid.value = 0


async def _sink(dut, out):
    for _ in range(4000):
        await RisingEdge(dut.aclk); await ReadOnly()
        if int(dut.m_axis_tvalid.value) and int(dut.m_axis_tready.value):
            out.append(int(dut.m_axis_tdata.value))
            if int(dut.m_axis_tlast.value): return


async def collect(dut, n):
    out = []
    dut.m_axis_tready.value = 1
    sink = cocotb.start_soon(_sink(dut, out))       # listen BEFORE the emit command lands
    await axil_write(dut, 0x00, 0x4)
    await sink
    await RisingEdge(dut.aclk)                      # the sink finishes in the read-only phase
    assert len(out) == n, f"result stream gave {len(out)} beats, want {n}"
    return out


@cocotb.test()
async def axi_two_vectors(dut):
    cocotb.start_soon(Clock(dut.aclk, 10, "ns").start())
    for s in ("s_axil_awvalid", "s_axil_wvalid", "s_axil_bready", "s_axil_arvalid", "s_axil_rready", "s_axis_tvalid", "s_axis_tlast", "m_axis_tready"):
        getattr(dut, s).value = 0
    dut.aresetn.value = 0
    for _ in range(3): await RisingEdge(dut.aclk)
    dut.aresetn.value = 1
    for _ in range(2): await RisingEdge(dut.aclk)
    assert await axil_read(dut, 0x20) == 0xB9000001
    for v in G.vectors(4242, 2):
        await axis_load(dut, 0, G.pack_weights(v["wp"], v["wm"]))
        await axis_load(dut, 1, G.pack_mask(v["mask"]))
        await axis_load(dut, 2, G.pack_act(v["x"]))
        await axis_load(dut, 3, G.pack_noise(v["eta"]))
        assert await axil_read(dut, 0x14) == 512
        await axil_write(dut, 0x08, int(v["S"]) | (int(v["passes"] == 2) << 5))
        await axil_write(dut, 0x00, 0x1)
        for _ in range(400):
            if (await axil_read(dut, 0x04)) & 0x2: break
        else:
            raise AssertionError("done never set")
        res = await collect(dut, 32)
        for c, w in enumerate(res):
            assert G.unpack_result(w) == (int(v["y"][c]), int(v["Y"][c])), f"seed {v['seed']} col {c}"
    assert await axil_read(dut, 0x1C) == 2
