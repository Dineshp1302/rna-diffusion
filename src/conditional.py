"""Conditional partial generation: hold a motif fixed, generate the rest.

The sampler clamps chosen positions to supplied tokens and never re-masks them,
so the model fills surrounding context conditioned on a constraint the caller
specifies. This is the generation mode that matters for design work: you rarely
want an unconstrained sample, you want a sequence that contains a required
element.

Two things are measured:

  retention  - does the constraint survive sampling? It must be 1.000 by
               construction; anything less is a bug in the clamping logic, so
               this is a correctness check rather than a result.

  context    - is the generated remainder still realistic? Measured by k-mer
               divergence and positional L1 computed on the NON-fixed
               positions only. Including the fixed positions would flatter the
               result, since those were supplied rather than generated.

Usage:
    python src/conditional.py --model results/model_v4.pt
"""
import argparse
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from diffusion import sample  # noqa: E402
from make_data import MOTIFS  # noqa: E402
from metrics import kmer_js, positional_frequencies  # noqa: E402
from model import SeqDiffusionTransformer  # noqa: E402
from tokenizer import STOI, decode, load  # noqa: E402


def build_constraint(motif, start, seq_len):
    """Return (tokens, mask) clamping `motif` at `start`."""
    tokens = torch.zeros(seq_len, dtype=torch.long)
    mask = torch.zeros(seq_len, dtype=torch.bool)
    for i, c in enumerate(motif):
        tokens[start + i] = STOI[c]
        mask[start + i] = True
    return tokens, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="results/model_v4.pt")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", type=str, default="results/conditional.csv")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    a = ck["args"]
    seq_len = a["seq_len"]
    model = SeqDiffusionTransformer(
        seq_len=seq_len, d_model=a["d_model"],
        n_layers=a["n_layers"], n_heads=a["n_heads"],
    )
    model.load_state_dict(ck["state_dict"])
    model.eval()

    real = load("data/test.txt", seq_len)
    pr = positional_frequencies(real)

    rows = []
    print(f"n={args.n}  T={args.steps}  model={args.model}\n")
    print(f"{'motif':8s} {'pos':>4s} {'retention':>10s} "
          f"{'k3_free':>9s} {'posL1_free':>11s}")

    for motif in MOTIFS:
        for start in (5, 22, 40):
            if start + len(motif) > seq_len:
                continue
            tok, msk = build_constraint(motif, start, seq_len)
            x = sample(model, args.n, seq_len=seq_len, n_steps=args.steps,
                       order="random", fixed_tokens=tok, fixed_mask=msk,
                       seed=7)
            gen = decode(x.numpy())

            # Correctness: the constraint must be present at the exact site.
            retention = float(np.mean(
                [s[start:start + len(motif)] == motif for s in gen]))

            # Quality of the *generated* remainder only.
            free = ~msk.numpy()
            gen_free = ["".join(np.array(list(s))[free]) for s in gen]
            real_free = ["".join(np.array(list(s))[free]) for s in real]
            k3 = kmer_js(gen_free, real_free, k=3)

            pg = positional_frequencies(gen)
            l1 = float(np.abs(pg[free] - pr[free]).sum(axis=1).mean())

            rows.append({"motif": motif, "start": start,
                         "retention": retention, "kmer_js_k3_free": k3,
                         "positional_l1_free": l1})
            print(f"{motif:8s} {start:4d} {retention:10.3f} "
                  f"{k3:9.2e} {l1:11.4f}")

    import pandas as pd
    df = pd.DataFrame(rows)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"\nmean retention      : {df.retention.mean():.4f}")
    print(f"mean k3 (free pos.) : {df.kmer_js_k3_free.mean():.2e}")
    print(f"wrote {args.out}")

    tok, msk = build_constraint("GCCACC", 22, seq_len)
    x = sample(model, 3, seq_len=seq_len, n_steps=args.steps, order="random",
               fixed_tokens=tok, fixed_mask=msk, seed=0)
    print("\nexample (GCCACC clamped at position 22):")
    for s in decode(x.numpy()):
        print(f"  {s[:22]}[{s[22:28]}]{s[28:]}")


if __name__ == "__main__":
    main()
