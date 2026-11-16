"""Figures for the README.

  fig1_quality_speed.png  - k-mer divergence vs sampling throughput, with the
                            diffusion model as a curve over T and the
                            non-neural baselines as fixed points. Both axes
                            log-scaled; the held-out ceiling is a horizontal
                            reference line. This is the headline figure.
  fig2_positional.png     - positional GC profile, generated vs real.
  fig3_motif.png          - motif occurrence rates by method.

Usage:
    python src/plots.py
"""
import argparse
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from diffusion import sample  # noqa: E402
from make_data import MOTIFS  # noqa: E402
from metrics import positional_frequencies  # noqa: E402
from model import SeqDiffusionTransformer  # noqa: E402
from tokenizer import decode, load  # noqa: E402

PALETTE = {
    "uniform": "#9e9e9e",
    "positional": "#4c72b0",
    "markov3": "#dd8452",
    "diffusion": "#55a868",
    "ceiling": "#c44e52",
}


def fig_quality_speed(df, out):
    fig, ax = plt.subplots(figsize=(7, 5))

    ceil = df[df.method.str.contains("CEILING")]
    if len(ceil):
        ax.axhline(ceil.kmer_js_k5.iloc[0], ls="--", lw=1.2,
                   color=PALETTE["ceiling"],
                   label=f"held-out ceiling ({ceil.kmer_js_k5.iloc[0]:.1e})")

    for name in ("uniform", "positional", "markov3"):
        r = df[df.method == name]
        if not len(r):
            continue
        ax.scatter(r.seqs_per_sec, r.kmer_js_k5, s=90, zorder=3,
                   color=PALETTE[name], label=name)

    d = df[df.method.str.startswith("diff")].copy()
    if len(d):
        d["T"] = d.method.str.extract(r"T=(\d+)").astype(int)
        d = d.sort_values("T")
        ax.plot(d.seqs_per_sec, d.kmer_js_k5, "-o", color=PALETTE["diffusion"],
                lw=2, ms=7, zorder=4, label="diffusion (sweep over T)")
        for _, r in d.iterrows():
            ax.annotate(f"T={r.T}", (r.seqs_per_sec, r.kmer_js_k5),
                        textcoords="offset points", xytext=(8, 6), fontsize=8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sampling throughput (sequences/sec, 1 CPU node, 8 threads)")
    ax.set_ylabel("5-mer Jensen-Shannon divergence vs held-out  (lower = better)")
    ax.set_title("Generation quality vs sampling cost")
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


def fig_positional(model, out, n=2000):
    real = load("data/test.txt", 50)
    gen = decode(sample(model, n, seq_len=50, n_steps=50,
                        order="random", seed=0).numpy())
    pr, pg = positional_frequencies(real), positional_frequencies(gen)
    gc_r, gc_g = pr[:, 1] + pr[:, 2], pg[:, 1] + pg[:, 2]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(gc_r, lw=2, color="#333333", label="held-out")
    ax.plot(gc_g, lw=2, ls="--", color=PALETTE["diffusion"],
            label="diffusion (T=50)")
    ax.set_xlabel("position")
    ax.set_ylabel("GC fraction")
    ax.set_title(f"Positional GC profile  "
                 f"(mean |diff| = {np.abs(gc_g - gc_r).mean():.4f}, "
                 f"r = {np.corrcoef(gc_g, gc_r)[0, 1]:.4f})")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


def fig_motifs(df, out):
    methods = ["CEILING (real)", "uniform", "positional", "markov3"]
    diff = [m for m in df.method if m.startswith("diff")]
    if diff:
        methods.append(diff[len(diff) // 2])
    rows = [m for m in methods if m in set(df.method)]

    x = np.arange(len(MOTIFS))
    width = 0.8 / max(len(rows), 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, m in enumerate(rows):
        r = df[df.method == m].iloc[0]
        vals = [r.get(f"motif_{mo}_gen", np.nan) for mo in MOTIFS]
        key = ("ceiling" if "CEILING" in m else
               "diffusion" if m.startswith("diff") else m)
        ax.bar(x + i * width, vals, width,
               label=m, color=PALETTE.get(key, "#777777"))

    ax.set_xticks(x + width * (len(rows) - 1) / 2)
    ax.set_xticklabels(MOTIFS)
    ax.set_ylabel("fraction of sequences containing motif")
    ax.set_title("Motif recovery by method")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/benchmark.csv")
    ap.add_argument("--model", default="results/model_v4.pt")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)

    fig_quality_speed(df, outdir / "fig1_quality_speed.png")
    fig_motifs(df, outdir / "fig3_motif.png")

    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    a = ck["args"]
    m = SeqDiffusionTransformer(seq_len=a["seq_len"], d_model=a["d_model"],
                                n_layers=a["n_layers"], n_heads=a["n_heads"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    fig_positional(m, outdir / "fig2_positional.png")


if __name__ == "__main__":
    main()
