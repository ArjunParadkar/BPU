"""Gate test 2 (the headline): >= 10k pre-registered random (weights, activations, seed) vectors from bpusim.fixed,
PL output == reference on every bit of every column.  Default run is 512 vectors (+ corners) so CI stays fast;
`pytest --gate-vectors 10000` (or BPU_GATE_N=10000) is the registered full run.  Any mismatch fails and the first
divergent (vector, column) lines are in the report.  Gate test 5 (corners) rides along: the corner vectors are
appended to every run."""
import json, os, pathlib, re, subprocess, sys
import pytest
from conftest import FPGA, HDL, tool

sys.path.insert(0, str(FPGA / "vectors"))
import gen_vectors as G

SEED_BASE = 20261101          # registered in vectors/REGISTERED.md before the first RTL run


@pytest.fixture(scope="session")
def gate_sim(zlut):
    tool("verilator")
    out = FPGA / "tests/_build/gate_verilator"
    exe = out / "Vtb_tile_gate"
    srcs = [HDL / "tb/tb_tile_gate.sv", *[HDL / f for f in os.listdir(HDL) if f.endswith(".sv")]]
    if not exe.exists() or any(s.stat().st_mtime > exe.stat().st_mtime for s in srcs):
        subprocess.run([str(FPGA / "scripts/build_gate_sim.sh"), str(out)], check=True)
    return exe


def run_gate(exe, vecdir, n):
    r = subprocess.run([str(exe), f"+dir={vecdir}", f"+n={n}"], cwd=HDL, capture_output=True, text=True)
    return r.stdout + r.stderr


def test_bit_exact_tile_gate(gate_sim, request, tmp_path_factory):
    n = request.config.getoption("--gate-vectors")
    vecdir = tmp_path_factory.mktemp("gate") / f"vec{n}"
    vecs = G.vectors(SEED_BASE, n)
    for c in G.corner_vectors():
        c["wset_new"] = True; vecs.append(c)
    meta = G.write_set(vecdir, vecs)
    out = run_gate(gate_sim, vecdir, meta["n_vectors"])
    (FPGA / "tests/_build").mkdir(exist_ok=True)
    (FPGA / f"tests/_build/gate_{n}.log").write_text(out)
    m = re.search(r"GATE (PASS|FAIL) vectors=(\d+)(?: columns=(\d+))? mismatches=(\d+)", out)
    assert m, out[-2000:]
    mism = [l for l in out.splitlines() if l.startswith("MISMATCH")]
    assert m.group(1) == "PASS" and int(m.group(4)) == 0, "first divergences:\n" + "\n".join(mism[:20])
    assert int(m.group(2)) == meta["n_vectors"]
    cyc = re.search(r"CYCLES per read: min=(\d+) max=(\d+) mean=(\d+)", out)
    print(out.splitlines()[-4:])
    sha = G.sha256_of(vecs[:n])
    (FPGA / f"tests/_build/gate_{n}.json").write_text(json.dumps(
        {"n_random": n, "n_total": meta["n_vectors"], "seed_base": SEED_BASE, "sha256": sha,
         "cycles_min": int(cyc.group(1)), "cycles_max": int(cyc.group(2)), "cycles_mean": int(cyc.group(3)),
         "verdict": "PASS"}))
