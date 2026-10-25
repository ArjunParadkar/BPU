"""E15 checkpoint round trip (2026-10-25): every chained segment crashed on resume because the checkpoint was loaded
with map_location=device and the RNG states then sat on CUDA.  The resume path must load on the CPU and hand
torch.set_rng_state a CPU ByteTensor; a paused + resumed run must reproduce the uninterrupted run exactly."""
import argparse
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "experiments"))
import e15_scratch_lm as e15  # noqa: E402


def _setup(tmp_path):
    cfg = dict(vocab=512, d=16, layers=1, heads=2, block=16)
    rng = np.random.default_rng(0)
    for n, sz in (("train", 20000), ("val", 5000)):
        rng.integers(0, 512, sz, dtype=np.uint16).tofile(tmp_path / f"{n}.bin")
    args = argparse.Namespace(device="cpu", lr=1e-3, batch_seqs=4, micro=2, tokens=12 * 4 * 16, curve_batches=1, eval_batches=1,
                              relpert_batches=1, data_in_ram=True)
    data = {"train": e15.Tokens(tmp_path / "train.bin", in_ram=True), "val": e15.Tokens(tmp_path / "val.bin")}
    return cfg, args, data


def test_pause_resume_reproduces_and_loads_rng_on_cpu(tmp_path, monkeypatch):
    cfg, args, data = _setup(tmp_path)
    monkeypatch.setattr(e15.json, "dump", lambda *a, **k: None)
    seen = {}
    real_load = torch.load

    def load(*a, **k):
        seen["map_location"] = k.get("map_location")
        c = real_load(*a, **k)
        c["rng_cpu"] = c["rng_cpu"].to(torch.int64)   # a wrongly typed state, as a CUDA-mapped one would be unusable
        return c
    monkeypatch.setattr(e15.torch, "load", load)
    results = {}

    def capture(model, *a, **k):
        results["w"] = {k2: v.clone() for k2, v in model.state_dict().items()}
        return {"eval": {"val": {"clean": 0.0}}}
    monkeypatch.setattr(e15, "finish", capture)
    monkeypatch.setattr(e15, "ROOT", tmp_path)
    monkeypatch.setattr(e15, "OUT", "a")
    assert e15.train_run("tiles", 0.04, 0, args, cfg, data, deadline=float("inf"), heartbeat=lambda: None) == "done"
    ref = results.pop("w")
    monkeypatch.setattr(e15, "OUT", "b")
    assert e15.train_run("tiles", 0.04, 0, args, cfg, data, deadline=0.0, heartbeat=lambda: None) == "paused"
    assert e15.train_run("tiles", 0.04, 0, args, cfg, data, deadline=float("inf"), heartbeat=lambda: None) == "done"
    assert seen["map_location"] == "cpu"
    got = results.pop("w")
    for k, v in ref.items():
        assert torch.equal(v, got[k]), k
