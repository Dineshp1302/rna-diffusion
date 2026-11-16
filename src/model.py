"""Transformer denoiser for masked discrete diffusion over RNA sequences.

The network sees a partially masked sequence and the diffusion timestep, and
predicts the original nucleotide at every position. Only predictions at masked
positions contribute to the loss.

Output dimension is 4, not 5: MASK is an input state the model must recognise,
never a target it should predict. Giving the head a MASK logit would let the
model trivially lower its loss by predicting MASK back, so it is excluded.
"""
import math

import torch
import torch.nn as nn

VOCAB_SIZE = 5   # A, C, G, U, MASK
N_BASES = 4      # prediction targets


class TimestepEmbedding(nn.Module):
    """Sinusoidal embedding of the continuous noise level, as in standard
    diffusion models. Passed through a small MLP so the network can learn a
    useful reparameterisation of it."""

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, t):
        # t: (B,) float in [0, 1]
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t.float()[:, None] * freqs[None, :] * 1000.0
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.d_model % 2:
            emb = nn.functional.pad(emb, (0, 1))
        return self.mlp(emb)


class SeqDiffusionTransformer(nn.Module):
    """Bidirectional transformer encoder denoiser.

    Bidirectional, not causal: every position may attend to every other. A
    causal mask would make this an autoregressive model, which is a different
    method with different sampling behaviour -- the whole point of the
    diffusion formulation is that positions are filled in parallel and in a
    confidence-driven order rather than strictly left to right.
    """

    def __init__(self, seq_len=50, d_model=128, n_heads=4, n_layers=4,
                 d_ff=512, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        self.tok_emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.time_emb = TimestepEmbedding(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, N_BASES)

        nn.init.normal_(self.pos_emb, std=0.02)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        # Zero-init the timestep projection. With default init the MLP emits
        # activations of order 1 against token/positional embeddings at
        # std=0.02 -- a ~50x imbalance that swamps the token signal, and the
        # model's easiest descent direction becomes ignoring x entirely and
        # fitting positional marginals. Starting the conditioning at zero
        # lets the network introduce it only as it becomes useful.
        nn.init.zeros_(self.time_emb.mlp[-1].weight)
        nn.init.zeros_(self.time_emb.mlp[-1].bias)

    def forward(self, x, t):
        """x: (B, L) int64 token ids. t: (B,) float in [0, 1].
        Returns logits (B, L, 4)."""
        h = self.tok_emb(x) + self.pos_emb[:, : x.size(1)]
        h = h + self.time_emb(t)[:, None, :]
        h = self.encoder(h)
        return self.head(self.norm(h))

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
