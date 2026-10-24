"""E15 data: tokenise the training stream and the shift sets once, to uint16 token files (CPU job).

Writes ``data/e15/`` (git-ignored; stays on the cluster):
  train.bin        ~0.85B GPT-2 tokens of OpenWebText (Skylion007/openwebtext, streamed); if it cannot be
                   loaded, FineWeb-Edu sample-10BT (HuggingFaceFW/fineweb-edu) — meta.json says which
  val.bin          4M tokens from documents NOT in train.bin (the first documents of the stream)
  tinystories.bin  up to 4M tokens of roneneldan/TinyStories (validation split): narrative shift set
  python.bin       up to 4M tokens of Python (codeparrot/codeparrot-clean-valid): code shift set
  meta.json        sources, token counts, tokenizer, the EOT separator (50256 between documents)

Usage: python e15_data.py --train-tokens 850000000 [--out data/e15] [--procs 32]; --smoke writes a few
thousand tokens of each to data/e15_smoke from the first documents.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

import numpy as np

from common import ROOT

EOT = 50256


def stream(name, config=None, split="train", text_key="text"):
    from datasets import load_dataset
    kw = dict(split=split, streaming=True)
    ds = load_dataset(name, config, **kw) if config else load_dataset(name, **kw)
    for ex in ds:
        t = ex.get(text_key)
        if t:
            yield t


def web_stream():
    errs = []
    for name, cfg in (("Skylion007/openwebtext", None), ("HuggingFaceFW/fineweb-edu", "sample-10BT")):
        try:
            it = stream(name, cfg)
            first = next(it)
            def gen(first=first, it=it):
                yield first
                yield from it
            return name + (f"/{cfg}" if cfg else ""), gen(), errs
        except Exception as e:   # noqa: BLE001 - record and fall back
            errs.append(f"{name}: {type(e).__name__}: {e}"[:400])
    raise RuntimeError("no web corpus could be loaded: " + " | ".join(errs))


def write_tokens(texts, tok, path, limit, batch=1000, log=""):
    batch = min(batch, max(8, limit // 2000))
    """Tokenise ``texts`` in batches until ``limit`` tokens; documents separated by EOT. Returns n tokens."""
    n = 0; t0 = time.time(); buf = []
    tmp = pathlib.Path(str(path) + ".tmp")
    with open(tmp, "wb") as f:
        def flush(buf):
            nonlocal n
            enc = tok(buf, add_special_tokens=False)["input_ids"]
            arr = np.fromiter((t for ids in enc for t in (ids + [EOT])), dtype=np.uint16)
            if n + len(arr) > limit:
                arr = arr[:limit - n]
            f.write(arr.tobytes()); n += len(arr)
        for t in texts:
            buf.append(t)
            if len(buf) >= batch:
                flush(buf); buf = []
                if n >= limit:
                    break
                if log and n // 50_000_000 != (n - 1) // 50_000_000:
                    print(f"  {log}: {n / 1e6:.0f}M tokens ({n / (time.time() - t0) / 1e6:.2f}M/s)", flush=True)
        if buf and n < limit:
            flush(buf)
    os.replace(tmp, path)
    return n


def main():
    ap = argparse.ArgumentParser(description="E15 tokenised data")
    ap.add_argument("--train-tokens", type=int, default=850_000_000)
    ap.add_argument("--val-tokens", type=int, default=4_000_000)
    ap.add_argument("--shift-tokens", type=int, default=4_000_000)
    ap.add_argument("--out", default="data/e15")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    if args.smoke:
        args.train_tokens, args.val_tokens, args.shift_tokens, args.out = 20000, 5000, 5000, "data/e15_smoke"
    out = ROOT / args.out; out.mkdir(parents=True, exist_ok=True)
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2"); tok.model_max_length = 10 ** 9
    meta = {"tokenizer": "gpt2", "eot": EOT, "dtype": "uint16", "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    src, it, errs = web_stream()
    meta["web_source"] = src; meta["web_source_errors"] = errs
    print(f"web source {src}", flush=True)
    # val first (held-out documents), then train continues the same stream: no document is in both
    meta["val_tokens"] = write_tokens(it, tok, out / "val.bin", args.val_tokens, log="val")
    meta["train_tokens"] = write_tokens(it, tok, out / "train.bin", args.train_tokens, log="train")
    for name, fn, ds, cfg, split, key in (("tinystories", "tinystories.bin", "roneneldan/TinyStories", None, "validation", "text"),
                                          ("python", "python.bin", "codeparrot/codeparrot-clean-valid", None, "train", "content")):
        try:
            meta[f"{name}_tokens"] = write_tokens(stream(ds, cfg, split, key), tok, out / fn, args.shift_tokens, log=name)
            meta[f"{name}_source"] = f"{ds} ({split})"
        except Exception as e:   # noqa: BLE001
            meta[f"{name}_error"] = f"{type(e).__name__}: {e}"[:400]
            print(f"  {name}: FAILED {meta[f'{name}_error']}", flush=True)
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(meta, open(out / "meta.json", "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
