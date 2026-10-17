"""Regenerate one PNG per experiment from results/<e>/results.json (no recomputation)."""
import importlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from common import load_results, figure_path  # noqa: E402

MODULES = {"e1": ("e1_gpt2_noise", "e1_gpt2_noise"), "e2": ("e2_grow_vs_rigl", "e2_grow_vs_rigl"),
           "e3": ("e3_drift", "e3_drift"), "e4": ("e4_ring_selforg", "e4_ring_selforg"),
           "e5": ("e5_grid_selforg", "e5_grid_selforg"), "e6": ("e6_memory", "e6_memory"),
           "e7": ("e7_modularity", "e7_modularity"), "e8": ("e8_nanogpt_tiles", "e8_nanogpt_tiles"),
           "e8b": ("e8b_energy_model", "e8b_energy"), "e9": ("e9_trainability", "e9_trainability"),
           "e10": ("e10_continual", "e10_continual"), "e2_cifar": ("e2_grow_vs_rigl", "e2_cifar")}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(MODULES)
    for e in wanted:
        mod_name, fig_name = MODULES[e]
        try:
            res = load_results(e)
        except FileNotFoundError:
            print(f"[{e}] no results.json yet, skipped"); continue
        importlib.import_module(mod_name).plot(res, figure_path(fig_name))
