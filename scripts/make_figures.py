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
           "e10": ("e10_continual", "e10_continual"), "e2_cifar": ("e2_grow_vs_rigl", "e2_cifar"),
           "e3_tracking": ("e3_tracking", "e3_tracking"), "e4_level": ("e4_ring_selforg", "e4_level"),
           "e4_level_capped": ("e4_ring_selforg", "e4_level_capped"), "e7_ref": ("e7_modularity", "e7_ref"),
           "e3_cue": ("e3_tracking", "e3_cue"), "e4_level_t1": ("e4_ring_selforg", "e4_level_t1"), "e4_level_t05": ("e4_ring_selforg", "e4_level_t05"),
           "e4_level_t025": ("e4_ring_selforg", "e4_level_t025"), "e9_pulse": ("e9_trainability", "e9_pulse"),
           # round three (2026-10-19): variability programme + nearest-pulse reruns
           "e9_pareto": ("e9_trainability", "e9_pareto"), "e4_np": ("e4_ring_selforg", "e4_np"), "e5_selforg_np": ("e5_grid_selforg", "e5_selforg_np"),
           "e6_k32": ("e6_memory", "e6_k32"), "e3_final": ("e3_tracking", "e3_final"), "e11": ("e11_regularization", "e11_regularization"),
           "e12": ("e12_exploration", "e12_exploration"), "e13": ("e13_stochastic_resonance", "e13_stochastic_resonance"), "e14": ("e14_sampling", "e14_sampling"),
           # HPCC (Pinnacle) runs, 2026-10-22/23: the MONDAY queue and its bracket extensions
           "e11_f32_bracket": ("e11_regularization", "e11_f32_bracket"), "e11_full_tiles": ("e11_regularization", "e11_full_tiles"),
           "e11_full_f32": ("e11_regularization", "e11_full_f32"), "e12_long": ("e12_exploration", "e12_long"),
           "e3_final50": ("e3_tracking", "e3_final50"), "e5_ref20": ("e5_grid_selforg", "e5_ref20"),
           "e5_selforg_np20": ("e5_grid_selforg", "e5_selforg_np20"), "e1_full": ("e1_gpt2_noise", "e1_full")}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(MODULES)
    for e in wanted:
        mod_name, fig_name = MODULES[e]
        try:
            res = load_results(e)
        except FileNotFoundError:
            print(f"[{e}] no results.json yet, skipped"); continue
        importlib.import_module(mod_name).plot(res, figure_path(fig_name))
