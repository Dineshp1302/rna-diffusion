"""Isolate model capability from sampler behaviour.

If the model can fill a single masked position inside an otherwise-visible
motif, it has learned the dependency and the sampler is losing it. If it
cannot, the model never learned motif structure and more sampling steps or a
different unmasking order will not help.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, torch
import torch.nn.functional as F
from model import SeqDiffusionTransformer
from tokenizer import encode, load
from make_data import MOTIFS

ck = torch.load("results/model.pt", map_location="cpu", weights_only=False)
a = ck["args"]
m = SeqDiffusionTransformer(seq_len=a["seq_len"], d_model=a["d_model"],
                            n_layers=a["n_layers"], n_heads=a["n_heads"])
m.load_state_dict(ck["state_dict"]); m.eval()

real = load("data/test.txt", 50)
MASK = 4

for motif in MOTIFS:
    hits = [s for s in real if motif in s][:300]
    if not hits:
        continue
    correct = tot = 0
    for s in hits:
        start = s.index(motif)
        x0 = torch.from_numpy(encode([s]))
        for off in range(len(motif)):
            pos = start + off
            x = x0.clone(); x[0, pos] = MASK
            t = torch.full((1,), 1.0 / 50)
            with torch.no_grad():
                p = F.softmax(m(x, t)[0, pos], -1)
            if int(p.argmax()) == int(x0[0, pos]):
                correct += 1
            tot += 1
    print(f"{motif:8s} single-position fill accuracy: {correct/tot:.3f}  (chance 0.25)")
