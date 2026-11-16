"""Compare positional base frequencies of generated vs real sequences.

The sharpest early test of whether the sampler works: the corpus has a strong
GC gradient by construction, so a working model must reproduce it. A flat
profile means position was ignored; an inverted one means a bug.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, torch
from model import SeqDiffusionTransformer
from diffusion import sample
from tokenizer import decode, load
from metrics import positional_frequencies

ckpt = torch.load("results/model.pt", map_location="cpu", weights_only=False)
a = ckpt["args"]
m = SeqDiffusionTransformer(seq_len=a["seq_len"], d_model=a["d_model"],
                            n_layers=a["n_layers"], n_heads=a["n_heads"])
m.load_state_dict(ckpt["state_dict"])

gen = decode(sample(m, 1000, n_steps=50, seed=0).numpy())
real = load("data/test.txt", a["seq_len"])

pg, pr = positional_frequencies(gen), positional_frequencies(real)
gc_g, gc_r = pg[:, 1] + pg[:, 2], pr[:, 1] + pr[:, 2]
print("pos   GC_gen  GC_real")
for i in range(0, a["seq_len"], 5):
    print(f"{i:3d}   {gc_g[i]:.3f}   {gc_r[i]:.3f}")
print(f"\nmean |diff| = {np.abs(gc_g - gc_r).mean():.4f}")
print(f"corr        = {np.corrcoef(gc_g, gc_r)[0,1]:.4f}")
