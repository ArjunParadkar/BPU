#!/usr/bin/env python
"""Golden-vector generator for the BPU tile gate (contract section 8).  Reference-first: everything here is
computed by bpusim.fixed BEFORE the RTL sees it; the RTL is never consulted.

A vector is fully determined by its seed:
  rng = PCG64(seed)
  passes = 2 if seed is odd else 1
  S      = {17,18,19}[seed % 3] (+8 for two passes)
  sigma_r = {0.0, 0.02, 0.05, 0.1, 0.2}[seed % 5]
  w+ ~ U[0,1] -> w_q ; w- likewise ; mask = all ones unless seed % 4 == 3 (then Bernoulli 0.8) ;
  x ~ U[-1,1) -> x_q (int8 / int16) ; eta stream (2048 x Q1.14) from the same rng
Weight sets are shared by VEC_PER_WSET consecutive vectors (the first vector of a group draws them) so a
10k-vector run reloads weights 100 times, like a real tile would.
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from bpusim import fixed as F

VEC_PER_WSET = 100
SHIFTS = (17, 18, 19)
SIGMAS = (0.0, 0.02, 0.05, 0.1, 0.2)


def draw_weights(rng, seed):
    wp = F.w_to_code(rng.uniform(0, 1, (32, 32)))
    wm = F.w_to_code(rng.uniform(0, 1, (32, 32)))
    mask = np.ones((32, 32), bool) if seed % 4 != 3 else (rng.random((32, 32)) < 0.8)
    return wp, wm, mask


def make_vector(seed: int, weights=None):
    rng = np.random.default_rng(seed)
    passes = 2 if seed % 2 else 1
    S = SHIFTS[seed % 3] + (8 if passes == 2 else 0)
    sigma = SIGMAS[seed % 5]
    wp, wm, mask = draw_weights(rng, seed) if weights is None else weights
    if weights is not None:
        draw_weights(rng, seed)                       # keep the draw order identical either way
    x = F.act_to_code(rng.uniform(-1, 1, 32), passes)
    eta = F.eta_stream_from_numpy(rng, sigma)[0]
    r = F.tile_read(wp, wm, mask, x, eta, S, passes)
    return dict(seed=seed, passes=passes, S=S, sigma=sigma, wp=wp, wm=wm, mask=mask, x=x, eta=eta,
                y=r.y_adc.astype(np.int64), Y=r.Y, Ip=r.I_plus[-1], Im=r.I_minus[-1], clipped=r.clipped)


def corner_vectors():
    """Contract corner cases (test 5): rails, all-zero / all-one rows, max |I+ - I-|, saturation, S = 0 and 31,
    extreme noise, masked-out tile."""
    ones = np.ones((32, 32), bool)
    z = np.zeros(2048, np.int64)
    out = []
    def add(name, wp, wm, mask, x, eta, S, passes=1, sigma=None):
        r = F.tile_read(wp, wm, mask, x, eta, S, passes)
        out.append(dict(seed=-1, name=name, passes=passes, S=S, sigma=sigma, wp=np.asarray(wp), wm=np.asarray(wm), mask=mask,
                        x=np.asarray(x, np.int64), eta=eta, y=r.y_adc.astype(np.int64), Y=r.Y, Ip=r.I_plus[-1], Im=r.I_minus[-1],
                        clipped=r.clipped))
    full = np.full((32, 32), 255); zero = np.zeros((32, 32), np.int64)
    xmax = np.full(32, 127); xmin = np.full(32, -128)
    add("w_all_one_plus_xmax", full, zero, ones, xmax, z, 0)                  # max positive current, S=0 saturates
    add("w_all_one_minus_xmax", zero, full, ones, xmax, z, 0)                 # max negative
    add("w_all_one_plus_xmin", full, zero, ones, xmin, z, 0)
    add("max_diff_exact", full, zero, ones, xmax, z, 21)                      # 127*65425*32 = 265,887,200 -> >>21
    add("all_zero_rows", full, zero, ones, np.zeros(32, np.int64), z, 0)
    add("single_row_one", full, zero, ones, np.eye(32, dtype=np.int64)[5] * 64, z, 6)   # verify-style read: 64*G/64 = G
    add("balanced_pair_zero", full, full, ones, xmax, z, 0)                   # G+ == G- -> exactly zero
    add("balanced_pair_noise", full, full, ones, xmax, np.where(np.arange(2048) % 2 == 0, 16383, -16383), 0)
    add("eta_max_all", full, zero, ones, xmax, np.full(2048, 16383), 22)      # Ghat max 130846
    add("eta_min_all", full, zero, ones, xmax, np.full(2048, -16383), 0)      # Ghat -> ~4
    add("mask_all_off", full, zero, np.zeros((32, 32), bool), xmax, z, 0)
    add("S31", full, zero, ones, xmax, z, 31)
    add("two_pass_rails", full, zero, ones, np.array([32767, -32768] * 16), z, 0, passes=2)
    add("two_pass_lowbyte_only", full, zero, ones, np.full(32, 255), z, 0, passes=2)   # hi byte 0, lo 255 unsigned
    add("two_pass_minus_one", full, zero, ones, np.full(32, -1), z, 8, passes=2)       # lo 255, hi -1 -> x = -1
    # round-half-even probes on the ADC: single synapse, Y = x * (G+ - G-) lands exactly on ties
    # Y = x * (G+ - G-) with a single synapse: choose x=64, w+ = 1, w- = 0 -> 64*254 = 16256 ; S=7: 127.0 exactly
    wp1 = zero.copy(); wp1[0, 0] = 1
    add("adc_tie_even", wp1, zero, ones, np.eye(32, dtype=np.int64)[0] * 64, z, 7)      # 16256/128 = 127 exact
    wp2 = zero.copy(); wp2[0, 0] = 1
    x2 = np.eye(32, dtype=np.int64)[0] * 1
    add("adc_tie_rne_1", wp2, zero, ones, x2, z, 1)                                      # 254/2 = 127 exact
    add("adc_tie_rne_2", wp2, zero, ones, x2, z, 2)                                      # 254/4 = 63.5 -> 64 (even)
    add("adc_tie_rne_3", wp2, zero, ones, x2, z, 3)                                      # 254/8 = 31.75 -> 32
    add("adc_tie_rne_neg", wp2, zero, ones, -x2, z, 2)                                   # -63.5 -> -64 (even)
    wp3 = zero.copy(); wp3[0, 0] = 3                                                     # 762/4 = 190.5 -> 190 (even), sat 127
    add("adc_tie_rne_half_even_190", wp3, zero, ones, x2, z, 4)                          # 762/16 = 47.625 -> 48
    return out


def vectors(seed_base: int, n: int):
    out = []
    weights = None
    for i in range(n):
        seed = seed_base + i
        if i % VEC_PER_WSET == 0:
            weights = None
        v = make_vector(seed, weights)
        weights = (v["wp"], v["wm"], v["mask"])
        v["wset_new"] = (i % VEC_PER_WSET == 0)
        out.append(v)
    return out


# ------------------------------------------------------------------ encodings shared with the RTL / PS
def pack_weights(wp, wm):
    """256 x 64-bit beats: synapse k (row-major) -> 16 bits {w-, w+}, 4 per beat."""
    words = (wm.astype(np.uint64) << 8) | wp.astype(np.uint64)
    words = words.reshape(-1, 4)
    return words[:, 0] | (words[:, 1] << 16) | (words[:, 2] << 32) | (words[:, 3] << 48)


def pack_mask(mask):
    bits = mask.astype(np.uint64).reshape(-1)
    beats = np.zeros(16, np.uint64)
    for k in range(1024):
        beats[k // 64] |= bits[k] << np.uint64(k % 64)
    return beats


def pack_act(x):
    w = (np.asarray(x, np.int64) & 0xFFFF).astype(np.uint64).reshape(-1, 4)
    return w[:, 0] | (w[:, 1] << 16) | (w[:, 2] << 32) | (w[:, 3] << 48)


def pack_noise(eta):
    w = (np.asarray(eta, np.int64) & 0xFFFF).astype(np.uint64).reshape(-1, 4)
    return w[:, 0] | (w[:, 1] << 16) | (w[:, 2] << 32) | (w[:, 3] << 48)


def pack_expect(y, Y):
    return ((np.asarray(y, np.int64) & 0xFF).astype(np.uint64) << 56) | (np.asarray(Y, np.int64) & ((1 << 40) - 1)).astype(np.uint64)


def pack_cfg(v, wset_idx):
    return np.uint64(wset_idx & 0xFFFF) | (np.uint64(v["S"]) << 16) | (np.uint64(v["passes"] == 2) << 21)


def unpack_result(word):
    word = int(word)
    y = (word >> 56) & 0xFF; y = y - 256 if y >= 128 else y
    Y = word & ((1 << 40) - 1); Y = Y - (1 << 40) if Y >= (1 << 39) else Y
    return y, Y


def write_hex(path, arr):
    with open(path, "w") as f:
        for w in np.asarray(arr, np.uint64).reshape(-1):
            f.write(f"{int(w):016x}\n")


def write_set(outdir: pathlib.Path, vecs):
    outdir.mkdir(parents=True, exist_ok=True)
    wsets, masks, cfg, act, noise, exp = [], [], [], [], [], []
    wset_idx = -1
    for v in vecs:
        if v.get("wset_new", True):
            wset_idx += 1
            wsets.append(pack_weights(v["wp"], v["wm"])); masks.append(pack_mask(v["mask"]))
        cfg.append(pack_cfg(v, wset_idx)); act.append(pack_act(v["x"])); noise.append(pack_noise(v["eta"]))
        exp.append(pack_expect(v["y"], v["Y"]))
    write_hex(outdir / "wset.hex", np.concatenate(wsets)); write_hex(outdir / "mask.hex", np.concatenate(masks))
    write_hex(outdir / "cfg.hex", np.array(cfg)); write_hex(outdir / "act.hex", np.concatenate(act))
    write_hex(outdir / "noise.hex", np.concatenate(noise)); write_hex(outdir / "expect.hex", np.concatenate(exp))
    meta = {"n_vectors": len(vecs), "n_wsets": wset_idx + 1,
            "names": [v.get("name", str(v["seed"])) for v in vecs]}
    (outdir / "meta.json").write_text(json.dumps(meta))
    return meta


def sha256_of(vecs):
    h = hashlib.sha256()
    for v in vecs:
        for k in ("wp", "wm", "mask", "x", "eta", "y", "Y"):
            h.update(np.ascontiguousarray(np.asarray(v[k], np.int64)).tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-base", type=int, default=20261101)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--corners", action="store_true", help="append the corner vectors")
    ap.add_argument("--fixture", type=pathlib.Path, help="also write the frozen npz (seeds + expected + sha)")
    a = ap.parse_args()
    vecs = vectors(a.seed_base, a.n)
    if a.corners:
        for c in corner_vectors():
            c["wset_new"] = True
            vecs.append(c)
    meta = write_set(a.out, vecs)
    digest = sha256_of(vecs[:a.n])
    print(json.dumps({**meta, "sha256_stimulus_and_expected": digest, "seed_base": a.seed_base}))
    if a.fixture:
        np.savez_compressed(a.fixture, seed_base=a.seed_base, n=a.n,
                            y=np.stack([v["y"] for v in vecs[:a.n]]), Y=np.stack([v["Y"] for v in vecs[:a.n]]),
                            sha256=digest)


if __name__ == "__main__":
    main()
