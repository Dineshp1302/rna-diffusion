"""Evaluation metrics for generated RNA sequences.

All metrics compare a set of generated sequences against a held-out reference
set. Running the reference set against itself (split in half) gives the ceiling
any generator could reach; running uniform random sequence gives the floor.
Every reported number should be read between those two bounds.

Metrics:
  kmer_js        - Jensen-Shannon divergence of k-mer frequency distributions.
                   Primary quality metric. 0 = identical, 1 = disjoint.
  gc_ks          - Kolmogorov-Smirnov statistic on per-sequence GC content.
  mfe_stats      - Minimum free energy distribution via ViennaRNA (optional).
  novelty        - Fraction of generated sequences absent from the training set.
  distinctness   - Fraction of generated sequences that are unique.
  motif_recall   - Per-motif occurrence rate, generated vs reference.
"""
import itertools
import time

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

ALPHABET = "ACGU"

try:
    import RNA as _RNA
    HAS_VIENNA = True
except ImportError:
    HAS_VIENNA = False


def kmer_counts(seqs, k=3):
    """Return a normalised frequency vector over all 4^k k-mers."""
    kmers = ["".join(p) for p in itertools.product(ALPHABET, repeat=k)]
    index = {km: i for i, km in enumerate(kmers)}
    counts = np.zeros(len(kmers), dtype=np.float64)
    for s in seqs:
        for i in range(len(s) - k + 1):
            j = index.get(s[i:i + k])
            if j is not None:
                counts[j] += 1
    total = counts.sum()
    return counts / total if total else counts


def kmer_js(gen, ref, k=3):
    """Jensen-Shannon divergence between k-mer distributions.

    scipy returns the JS *distance* (sqrt of divergence); we square it back
    so the number is a divergence in [0, 1], which is the convention most
    sequence-generation papers report.
    """
    p = kmer_counts(gen, k)
    q = kmer_counts(ref, k)
    d = jensenshannon(p, q, base=2)
    return float(d ** 2) if np.isfinite(d) else float("nan")


def gc_content(seqs):
    """Per-sequence GC fraction."""
    return np.array([(s.count("G") + s.count("C")) / len(s) for s in seqs])


def gc_ks(gen, ref):
    """KS statistic and p-value comparing GC-content distributions.

    A small statistic means the distributions are close. The p-value is
    reported for completeness but is near-zero for any large sample, so the
    statistic is the number to compare across methods.
    """
    res = ks_2samp(gc_content(gen), gc_content(ref))
    return float(res.statistic), float(res.pvalue)


def positional_frequencies(seqs):
    """(seq_len, 4) matrix of base frequency at each position."""
    arr = np.array([list(s) for s in seqs])
    out = np.zeros((arr.shape[1], 4))
    for i, base in enumerate(ALPHABET):
        out[:, i] = (arr == base).mean(axis=0)
    return out


def positional_l1(gen, ref):
    """Mean L1 distance between positional base-frequency profiles.

    Catches a failure k-mer divergence can miss: a model that gets global
    composition right but puts it in the wrong places.
    """
    return float(np.abs(positional_frequencies(gen) -
                        positional_frequencies(ref)).sum(axis=1).mean())


def mfe_values(seqs, limit=2000):
    """Minimum free energies via ViennaRNA. Folding is the slowest metric
    here, so this subsamples by default."""
    if not HAS_VIENNA or limit <= 0:
        return None
    sub = seqs[:limit]
    return np.array([_RNA.fold(s)[1] for s in sub])


def mfe_stats(gen, ref, limit=2000):
    """Mean/std MFE for both sets plus a KS statistic between them."""
    g = mfe_values(gen, limit)
    r = mfe_values(ref, limit)
    if g is None:
        return None
    res = ks_2samp(g, r)
    return {
        "mfe_mean_gen": float(g.mean()),
        "mfe_std_gen": float(g.std()),
        "mfe_mean_ref": float(r.mean()),
        "mfe_std_ref": float(r.std()),
        "mfe_ks": float(res.statistic),
    }


def novelty(gen, train):
    """Fraction of generated sequences not present verbatim in training data.

    A generator that scores perfectly on every distributional metric while
    scoring low here has memorised rather than learned.
    """
    seen = set(train)
    return float(np.mean([s not in seen for s in gen]))


def distinctness(gen):
    """Fraction of generated sequences that are unique within the sample."""
    return len(set(gen)) / len(gen) if gen else 0.0


def motif_recall(gen, ref, motifs):
    """Per-motif fraction of sequences containing it, generated vs reference."""
    return {
        m: {
            "gen": float(np.mean([m in s for s in gen])),
            "ref": float(np.mean([m in s for s in ref])),
        }
        for m in motifs
    }


class Timer:
    """Context manager measuring wall-clock seconds and throughput.

    Sampling speed is hardware-dependent, so always report the node type
    alongside these numbers.
    """

    def __init__(self, n=None):
        self.n = n

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0
        self.per_second = self.n / self.seconds if self.n else None


def evaluate(gen, ref, train=None, motifs=None, k=3, mfe_limit=2000,
             seconds=None):
    """Run the full metric suite. Returns a flat dict ready for a DataFrame."""
    ks_stat, ks_p = gc_ks(gen, ref)
    out = {
        "n_generated": len(gen),
        f"kmer_js_k{k}": kmer_js(gen, ref, k=k),
        "gc_ks": ks_stat,
        "gc_ks_p": ks_p,
        "gc_mean_gen": float(gc_content(gen).mean()),
        "gc_mean_ref": float(gc_content(ref).mean()),
        "positional_l1": positional_l1(gen, ref),
        "distinctness": distinctness(gen),
    }
    if train is not None:
        out["novelty"] = novelty(gen, train)
    if motifs:
        for m, v in motif_recall(gen, ref, motifs).items():
            out[f"motif_{m}_gen"] = v["gen"]
            out[f"motif_{m}_ref"] = v["ref"]
    stats = mfe_stats(gen, ref, limit=mfe_limit)
    if stats:
        out.update(stats)
    if seconds is not None:
        out["seconds"] = seconds
        out["seqs_per_sec"] = len(gen) / seconds if seconds else float("nan")
    return out
