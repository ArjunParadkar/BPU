"""Shared helpers for the experiment scripts: arguments, timing, results.json, figures."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def base_parser(name: str, default_seeds: int = 3) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=name)
    ap.add_argument("--seeds", type=int, default=default_seeds, help="number of seeds")
    ap.add_argument("--full", action="store_true", help="paper budget (10 seeds, longer runs)")
    ap.add_argument("--out", type=str, default=None, help="results directory")
    ap.add_argument("--no-fig", action="store_true")
    return ap


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o) if np.isfinite(o) else None
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _out_dir(name: str, out: str | None) -> pathlib.Path:
    """``--out`` paths are taken relative to the repository root, not the shell's directory."""
    if not out:
        return RESULTS / name
    d = pathlib.Path(out)
    return d if d.is_absolute() else ROOT / d


def save_results(name: str, results: dict, out: str | None = None) -> pathlib.Path:
    d = _out_dir(name, out)
    d.mkdir(parents=True, exist_ok=True)
    results = dict(results)
    results.setdefault("meta", {})
    results["meta"].update({"experiment": name, "git": git_hash(), "python": platform.python_version(),
                            "numpy": np.__version__, "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
    path = d / "results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=1, cls=NumpyEncoder)
    print(f"[{name}] wrote {path}")
    return path


def load_results(name: str, out: str | None = None) -> dict:
    d = _out_dir(name, out)
    with open(d / "results.json") as f:
        return json.load(f)


def figure_path(name: str) -> pathlib.Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    return FIGURES / f"{name}.png"


class Timer:
    def __init__(self, label=""):
        self.label = label

    def __enter__(self):
        self.t0 = time.time(); return self

    def __exit__(self, *a):
        self.dt = time.time() - self.t0
        if self.label:
            print(f"  {self.label}: {self.dt:.1f} s")


def sign_test(x, y):
    """Two-sided exact sign test that paired x < y; returns p-value and the win count."""
    from scipy.stats import binomtest
    x = np.asarray(x, float); y = np.asarray(y, float)
    wins = int(np.sum(x < y)); n = int(np.sum(x != y))
    if n == 0:
        return 1.0, wins
    return float(binomtest(wins, n, 0.5).pvalue), wins
