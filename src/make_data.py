"""Generate a structured synthetic RNA corpus as a development/fallback dataset.

Real 5' UTRs are not uniform random sequence: base composition varies along the
sequence and short functional motifs recur. A uniform random corpus would make
the generative task trivial, so this builds in two learnable signals:

  1. A positional GC gradient (GC-rich 5' end, AU-rich 3' end).
  2. Short motifs implanted at random positions in a subset of sequences.

A positional-frequency baseline can capture (1) but not (2). That gap is what
the diffusion model should close, which is what makes the benchmark informative.
"""
import argparse
import pathlib

import numpy as np

SEQ_LEN = 50
ALPHABET = "ACGU"
MOTIFS = ["GCCACC", "AAUAAA", "UUUUU", "GGAGG"]
GC_START = 0.70
GC_END = 0.30
MOTIF_RATE = 1.0
N_MOTIFS_PER_SEQ = 3


def make_corpus(n=60000, seq_len=SEQ_LEN, seed=0):
    """Return a list of `n` synthetic RNA sequences of length `seq_len`."""
    rng = np.random.default_rng(seed)
    gc_gradient = np.linspace(GC_START, GC_END, seq_len)

    probs = np.empty((seq_len, 4), dtype=np.float64)
    probs[:, 0] = (1.0 - gc_gradient) / 2.0  # A
    probs[:, 1] = gc_gradient / 2.0          # C
    probs[:, 2] = gc_gradient / 2.0          # G
    probs[:, 3] = (1.0 - gc_gradient) / 2.0  # U

    cdf = probs.cumsum(axis=1)
    draws = rng.random((n, seq_len, 1))
    idx = (draws > cdf[None, :, :]).sum(axis=2)
    arr = np.array(list(ALPHABET), dtype="<U1")[idx]

    for i in range(n):
        if rng.random() >= MOTIF_RATE:
            continue
        slots = rng.choice(np.arange(0, seq_len - 8, 8), size=N_MOTIFS_PER_SEQ,
                           replace=False)
        for st in slots:
            motif = MOTIFS[rng.integers(len(MOTIFS))]
            arr[i, st:st + len(motif)] = list(motif)

    return ["".join(row) for row in arr]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--test-size", type=int, default=10000)
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="data")
    args = ap.parse_args()

    if args.test_size >= args.n:
        raise SystemExit("--test-size must be smaller than --n")

    seqs = make_corpus(n=args.n, seq_len=args.seq_len, seed=args.seed)

    rng = np.random.default_rng(args.seed + 1)
    order = rng.permutation(len(seqs))
    seqs = [seqs[i] for i in order]

    train = seqs[args.test_size:]
    test = seqs[:args.test_size]

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "train.txt").write_text("\n".join(train) + "\n")
    (outdir / "test.txt").write_text("\n".join(test) + "\n")

    n_unique = len(set(seqs))
    print(f"wrote {len(train)} train, {len(test)} test -> {outdir}/")
    print(f"length={args.seq_len}  unique={n_unique}/{len(seqs)} "
          f"({100 * n_unique / len(seqs):.2f}%)")
    print(f"example: {train[0]}")


if __name__ == "__main__":
    main()
