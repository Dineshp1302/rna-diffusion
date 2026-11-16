"""Benchmark generation quality against sampling cost.

Produces the project's headline result: a quality-vs-speed curve for the
diffusion model (swept over number of sampling steps T) with the non-neural
baselines as fixed reference points.

All timings are wall-clock on whatever hardware this runs on, so the node type
must be reported alongside the numbers. Relative comparisons hold across
hardware; absolute throughput does not.

Usage:
    python src/benchmark.py --n 5000
    python src/benchmark.py --n 2000 --steps 10 25 50 --quick
"""
import argparse
import json
import pathlib
import platform
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from baselines import all_baselines  # noqa: E402
from diffusion import sample  # noqa: E402
from make_data import MOTIFS  # noqa: E402
from metrics import evaluate, kmer_js  # noqa: E402
from model import SeqDiffusionTransformer  # noqa: E402
from tokenizer import decode, load  # noqa: E402


def load_model(path="results/model.pt", device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    a = ckpt["args"]
    m = SeqDiffusionTransformer(
        seq_len=a["seq_len"], d_model=a["d_model"],
        n_layers=a["n_layers"], n_heads=a["n_heads"],
    ).to(device)
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    return m, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--steps", type=int, nargs="+",
                    default=[10, 25, 50, 100, 250])
    ap.add_argument("--batch", type=int, default=500,
                    help="sampling batch size; memory scales with this")
    ap.add_argument("--model", type=str, default="results/model.pt")
    ap.add_argument("--mfe-limit", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", type=str, default="results/benchmark.csv")
    ap.add_argument("--order", type=str, default="random",
                    choices=["random", "confidence"])
    ap.add_argument("--gumbel", type=float, default=0.0)
    ap.add_argument("--quick", action="store_true",
                    help="skip MFE folding (the slowest metric)")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mfe_limit = 0 if args.quick else args.mfe_limit

    train = load("data/train.txt", 50)
    test = load("data/test.txt", 50)
    half = len(test) // 2
    rows = []

    def record(name, gen, seconds):
        r = evaluate(gen, test, train=train, motifs=MOTIFS,
                     mfe_limit=mfe_limit, seconds=seconds)
        r["method"] = name
        r["kmer_js_k5"] = kmer_js(gen, test, k=5)
        rows.append(r)
        print(f"{name:16s} k3={r['kmer_js_k3']:.2e} k5={r['kmer_js_k5']:.2e} "
              f"posL1={r['positional_l1']:.4f} "
              f"nov={r.get('novelty', float('nan')):.3f} "
              f"{r['seqs_per_sec']:9.1f} seq/s")

    print(f"device={device}  n={args.n}  node={platform.node()}\n")

    # Ceiling: held-out against itself. No generation, so no timing.
    r = evaluate(test[:half], test[half:], train=train, motifs=MOTIFS,
                 mfe_limit=mfe_limit, seconds=float("nan"))
    r["method"] = "CEILING (real)"
    r["kmer_js_k5"] = kmer_js(test[:half], test[half:], k=5)
    rows.append(r)
    print(f"{'CEILING (real)':16s} k3={r['kmer_js_k3']:.2e} "
          f"k5={r['kmer_js_k5']:.2e} posL1={r['positional_l1']:.4f}\n")

    # Non-neural baselines.
    for b in all_baselines(3):
        b.fit(train)
        t0 = time.perf_counter()
        gen = b.sample(args.n, seed=1)
        record(b.name, gen, time.perf_counter() - t0)

    print()

    # Diffusion model swept over sampling steps.
    model, margs = load_model(args.model, device)
    print(f"model params={model.n_params():,}\n")
    for T in args.steps:
        t0 = time.perf_counter()
        chunks = []
        for i in range(0, args.n, args.batch):
            k = min(args.batch, args.n - i)
            x = sample(model, k, seq_len=margs["seq_len"], n_steps=T,
                       order=args.order, gumbel=args.gumbel,
                       device=device, seed=1000 + i)
            chunks.append(x)
        seconds = time.perf_counter() - t0
        gen = decode(torch.cat(chunks).numpy())
        record(f"diff-{args.order} T={T}", gen, seconds)

    df = pd.DataFrame(rows)
    front = ["method", "kmer_js_k3", "kmer_js_k5", "positional_l1",
             "gc_ks", "novelty", "distinctness", "seqs_per_sec", "seconds"]
    cols = [c for c in front if c in df.columns]
    df = df[cols + [c for c in df.columns if c not in cols]]

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    meta = {
        "node": platform.node(),
        "device": device,
        "threads": args.threads,
        "n_generated": args.n,
        "steps_swept": args.steps,
        "order": args.order,
        "gumbel": args.gumbel,
        "model_params": model.n_params(),
    }
    pathlib.Path(args.out).with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2))

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
