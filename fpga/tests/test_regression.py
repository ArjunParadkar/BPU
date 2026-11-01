"""Gate test 6: the frozen fixture.  The registered 10k seeds' expected outputs (and the SHA-256 of stimulus +
expected) are committed; this test regenerates them from the current bpusim.fixed and fails if anything moved --
a change to the reference, the generator or numpy's stream shows up here before it can silently re-baseline the
RTL gate.  (The RTL gate itself is tests/test_gate.py; CI runs both.)"""
import pathlib, sys
import numpy as np
from conftest import FPGA

sys.path.insert(0, str(FPGA / "vectors"))
import gen_vectors as G

FIX = FPGA / "vectors/fixtures/gate_10k.npz"


def test_fixture_matches_current_reference():
    f = np.load(FIX)
    seed_base, n = int(f["seed_base"]), int(f["n"])
    vecs = G.vectors(seed_base, n)
    y = np.stack([v["y"] for v in vecs]); Y = np.stack([v["Y"] for v in vecs])
    first_bad = np.argwhere(y != f["y"])
    assert first_bad.size == 0, f"expected outputs moved; first at vector {first_bad[0][0]} column {first_bad[0][1]}"
    assert np.array_equal(Y, f["Y"])
    assert G.sha256_of(vecs) == str(f["sha256"]), "stimulus or expected outputs differ from the frozen fixture"
