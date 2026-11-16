"""Non-neural generative baselines.

These exist so the diffusion model's numbers mean something. Each captures
strictly more structure than the last:

  uniform    - no structure at all; the floor.
  positional - per-position base frequencies; captures composition gradients
               but nothing about which bases co-occur.
  markov     - order-n conditional frequencies; captures local dependencies
               (so it can reproduce motifs) but has no positional awareness
               and no global coordination.

If the diffusion model cannot beat the order-3 Markov chain, that is a result
worth reporting honestly rather than hiding.
"""
import numpy as np

ALPHABET = "ACGU"
IDX = {c: i for i, c in enumerate(ALPHABET)}


def _sample_from_rows(probs, rng, n):
    """Vectorised categorical sampling from a (L, 4) probability matrix."""
    cdf = probs.cumsum(axis=1)
    draws = rng.random((n, probs.shape[0], 1))
    idx = (draws > cdf[None, :, :]).sum(axis=2)
    idx = np.clip(idx, 0, 3)
    return np.array(list(ALPHABET), dtype="<U1")[idx]


class UniformBaseline:
    """Every base equally likely at every position."""

    name = "uniform"

    def fit(self, seqs):
        self.seq_len = len(seqs[0])
        return self

    def sample(self, n, seed=0):
        rng = np.random.default_rng(seed)
        probs = np.full((self.seq_len, 4), 0.25)
        return ["".join(r) for r in _sample_from_rows(probs, rng, n)]


class PositionalBaseline:
    """Independent draw at each position from that position's empirical
    base frequency. Captures composition gradients, not co-occurrence."""

    name = "positional"

    def fit(self, seqs):
        arr = np.array([list(s) for s in seqs])
        self.seq_len = arr.shape[1]
        self.probs = np.zeros((self.seq_len, 4))
        for i, base in enumerate(ALPHABET):
            self.probs[:, i] = (arr == base).mean(axis=0)
        return self

    def sample(self, n, seed=0):
        rng = np.random.default_rng(seed)
        return ["".join(r) for r in _sample_from_rows(self.probs, rng, n)]


class MarkovBaseline:
    """Order-n Markov chain over nucleotides.

    Context counts are Laplace-smoothed so unseen contexts stay samplable.
    The first `order` bases are drawn from the empirical prefix distribution.

    NOTE: evaluating this with k-mer JS at k <= order is degenerate -- the
    model is fit on exactly those statistics. Always report a larger k too.
    """

    def __init__(self, order=3, alpha=1.0):
        self.order = order
        self.alpha = alpha
        self.name = f"markov{order}"

    def fit(self, seqs):
        self.seq_len = len(seqs[0])
        k = self.order
        n_ctx = 4 ** k

        counts = np.full((n_ctx, 4), self.alpha, dtype=np.float64)
        prefix = {}
        for s in seqs:
            prefix[s[:k]] = prefix.get(s[:k], 0) + 1
            ctx = 0
            for i in range(k):
                ctx = ctx * 4 + IDX[s[i]]
            for i in range(k, len(s)):
                nxt = IDX[s[i]]
                counts[ctx, nxt] += 1
                ctx = (ctx * 4 + nxt) % n_ctx

        self.trans = counts / counts.sum(axis=1, keepdims=True)
        self.prefixes = list(prefix.keys())
        self.prefix_p = np.array([prefix[p] for p in self.prefixes],
                                 dtype=np.float64)
        self.prefix_p /= self.prefix_p.sum()
        return self

    def sample(self, n, seed=0):
        rng = np.random.default_rng(seed)
        k = self.order
        n_ctx = 4 ** k
        starts = rng.choice(len(self.prefixes), size=n, p=self.prefix_p)

        out = []
        for si in starts:
            s = self.prefixes[si]
            chars = list(s)
            ctx = 0
            for c in chars:
                ctx = ctx * 4 + IDX[c]
            for _ in range(k, self.seq_len):
                nxt = rng.choice(4, p=self.trans[ctx])
                chars.append(ALPHABET[nxt])
                ctx = (ctx * 4 + nxt) % n_ctx
            out.append("".join(chars))
        return out


def all_baselines(order=3):
    return [UniformBaseline(), PositionalBaseline(), MarkovBaseline(order)]
