"""PS firmware on the host: the C implementation of the contract against bpusim.fixed (gate tests 1/3/4 on the
PS side).  The C model of the PL (ps/src/hal_model.c) is the third independent implementation; the firmware's
sequencer and write-verify run on it unchanged."""
import json, os, pathlib, subprocess, sys
import numpy as np
import pytest
from conftest import FPGA

sys.path.insert(0, str(FPGA / "vectors"))
from bpusim import fixed as F
import gen_vectors as G
PS = FPGA / "ps"


@pytest.fixture(scope="session")
def host_build():
    subprocess.run(["make", "-s", "host"], cwd=PS, check=True)
    return PS / "build/host"


def test_c_model_bit_exact_on_random_vectors(host_build, tmp_path):
    vecs = G.vectors(31337, 300) + G.corner_vectors()
    for c in vecs[300:]:
        c["wset_new"] = True
    meta = G.write_set(tmp_path, vecs)
    r = subprocess.run([str(host_build / "runner"), "model", str(tmp_path)], capture_output=True, text=True)
    assert "GATE PASS" in r.stdout, r.stdout[-3000:]
    assert f"vectors={meta['n_vectors']}" in r.stdout


def test_c_primitives_match_reference(host_build):
    out = subprocess.run([str(host_build / "selftest"), "0xC0FFEE", "328"], capture_output=True, text=True, check=True).stdout
    got = {l.split()[0]: np.array([int(v) for v in l.split()[1:]]) for l in out.strip().splitlines()}
    # prng stream (mode B) == EtaPRNG
    np.testing.assert_array_equal(got["prng"], F.EtaPRNG(0xC0FFEE, 328).read_block())
    # rne shift
    want = [int(F.rne_shift(np.array([x]), s)[0]) for s in range(0, 32, 3) for x in range(-300, 301, 37)]
    np.testing.assert_array_equal(got["rne"], want)
    # pulses_for
    want = [int(F.pulses_for(np.array([w]), np.array([t]))[0]) for w in range(0, F.WS_ONE + 1, 1234) for t in range(0, F.WS_ONE + 1, 2345)]
    np.testing.assert_array_equal(got["pulses_for"], want)
    # step sizes
    want = []
    for w in range(0, F.WS_ONE + 1, 777):
        want += [int(F.step_q(np.array([w]), np.array([16384]), np.array([0]), np.array([1]))[0]),
                 int(F.step_q(np.array([w]), np.array([15000]), np.array([-1000]), np.array([-1]))[0])]
    np.testing.assert_array_equal(got["step"], want)
    # w_meas
    want = [int(F.w_meas_from_sum(np.array([s]))[0]) for s in range(0, 8 * 64 * 70000, 1234567)]
    np.testing.assert_array_equal(got["wmeas"], want)
    # pulse law with the c2c generator (instance 64 of the same seed)
    gen = F.Xoshiro128ss(*F.prng_seed_words(0xC0FFEE, 64)); lut = F.zlut()
    def c2c(shape):
        word = gen.next(); z = int(lut[word >> 22]); e = (328 * z + 4096) >> 13
        return np.array([max(-16383, min(16383, e))])
    w = np.array([1000]); want = []
    for k in range(40):
        w = F.pulse(w, np.array([(k % 7) - 3]), np.array([16000]), c2c); want.append(int(w[0]))
    np.testing.assert_array_equal(got["pulse"], want)


def predicted_convergent(sigma_r, target, tol=0.02, reads=8):
    """The verify circuit averages `reads` reads of a device whose relative read noise is sigma_r, so the measured
    state has standard error sigma_r * (target + G_min/G_range) / sqrt(reads).  The loop certifies |err| <= tol on a
    single averaged measurement, which it can only do reliably when that error is at most about tol / 2; above it
    the loop keeps pulsing (ping-pong) and never certifies, in the float model exactly as in fixed point.  At the
    top rail the measurement is clipped at w = 1, so the error is one-sided and the factor halves."""
    k = 1 if target >= 1.0 else 2
    return k * sigma_r * (target + 0.0101) / reads ** 0.5 <= tol


def test_write_verify_sweep_converges(host_build):
    """Gate test 3: targets {0..1} x sigma_r {0..0.2}, 8 rows x 32 devices each, from random initial states.
    Asserts: (a) every cell at the contract's operating point sigma_r <= 0.02 converges inside MAX_ITER with the
    true state in the band; (b) the convergence region over the whole sweep equals the resolution prediction
    (`predicted_convergent`) cell for cell -- i.e. the loop fails only where the band is below the verify
    circuit's own read resolution, and nowhere else; (c) the run is deterministic.  The full table is written to
    tests/_build/write_verify_sweep.json for docs/RESULTS.md."""
    r = subprocess.run([str(host_build / "write_verify"), "model", "20261101"], capture_output=True, text=True)
    rows = [json.loads(l) for l in r.stdout.strip().splitlines() if l.startswith("{")]
    (FPGA / "tests/_build").mkdir(exist_ok=True)
    (FPGA / "tests/_build/write_verify_sweep.json").write_text(json.dumps(rows, indent=1))
    assert len(rows) == 45
    op = [x for x in rows if x["sigma"] <= 0.02]
    assert all(x["pass"] for x in op), json.dumps([x for x in op if not x["pass"]], indent=1)
    assert max(x["iter_max"] for x in op) <= 20 and max(x["pulses_per_device"] for x in op) < 120
    mismatch = [x for x in rows if x["pass"] != predicted_convergent(x["sigma"], x["target"])]
    assert not mismatch, "convergence region differs from the resolution prediction:\n" + json.dumps(mismatch, indent=1)
    r2 = subprocess.run([str(host_build / "write_verify"), "model", "20261101"], capture_output=True, text=True)
    assert r2.stdout == r.stdout


def test_noise_reproducible_same_seed_twice(host_build, tmp_path):
    """Gate test 4 on the PS side: the same seed programmed twice gives identical reads (prng mode) and the
    PL-side stream equals bpusim's EtaPRNG stream -- checked through the C model here and through the RTL in
    cocotb_core.prng_mode_matches_reference_stream_and_output."""
    out1 = subprocess.run([str(host_build / "selftest"), "0xBADC0DE", "1638"], capture_output=True, text=True, check=True).stdout
    out2 = subprocess.run([str(host_build / "selftest"), "0xBADC0DE", "1638"], capture_output=True, text=True, check=True).stdout
    assert out1 == out2
    prng = np.array([int(v) for v in out1.splitlines()[0].split()[1:]])
    np.testing.assert_array_equal(prng, F.EtaPRNG(0xBADC0DE, 1638).read_block())
