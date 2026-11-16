"""Nucleotide tokenization for discrete diffusion over RNA sequences.

Vocabulary is the four RNA bases plus a [MASK] token. MASK is the absorbing
state of the forward diffusion process: noising replaces real nucleotides with
MASK, and the model learns to invert that. It is never present in real data.

Sequences are normalised to RNA on the way in (T -> U), so nothing downstream
has to care whether the source file stored DNA or RNA.
"""
import numpy as np

VOCAB = ["A", "C", "G", "U"]
MASK_TOKEN = 4
VOCAB_SIZE = 5  # 4 bases + MASK

STOI = {c: i for i, c in enumerate(VOCAB)}
ITOS = {i: c for i, c in enumerate(VOCAB)}
_DECODE = {**ITOS, MASK_TOKEN: "-"}


def normalize(seq):
    """Uppercase and convert DNA thymine to RNA uracil."""
    return seq.strip().upper().replace("T", "U")


def encode(seqs):
    """List of equal-length sequence strings -> int64 array (n_seqs, seq_len)."""
    seqs = [normalize(s) for s in seqs]
    if not seqs:
        return np.empty((0, 0), dtype=np.int64)
    lengths = {len(s) for s in seqs}
    if len(lengths) != 1:
        raise ValueError(
            f"encode() expects uniform length; saw {sorted(lengths)[:5]}. "
            "Run filter_valid() first."
        )
    try:
        return np.array([[STOI[c] for c in s] for s in seqs], dtype=np.int64)
    except KeyError as e:
        raise ValueError(
            f"encode() found non-ACGU character {e.args[0]!r}. "
            "Run filter_valid() first."
        ) from None


def decode(arr):
    """Int array -> list of sequence strings. MASK renders as '-'."""
    return ["".join(_DECODE[int(t)] for t in row) for row in np.atleast_2d(arr)]


def filter_valid(seqs, length):
    """Keep sequences of exactly `length` containing only A/C/G/U."""
    allowed = set(VOCAB)
    out = []
    for s in seqs:
        s = normalize(s)
        if len(s) == length and set(s) <= allowed:
            out.append(s)
    return out


def load(path, length=None):
    """Read one sequence per line; optionally filter to a fixed length."""
    with open(path) as f:
        seqs = [line.strip() for line in f if line.strip()]
    return filter_valid(seqs, length) if length is not None else [
        normalize(s) for s in seqs
    ]
