"""Absorbing-state (masked) discrete diffusion.

Why not Gaussian diffusion: nucleotides are categorical, and adding Gaussian
noise to a token id is meaningless -- there is no sense in which G is "between"
A and U. Discrete diffusion instead defines corruption as a Markov chain over
the token alphabet. The absorbing-state variant used here sends tokens to a
single [MASK] state and never out of it, which makes the forward process
trivial to sample at any timestep and the reverse process a masked-token
prediction problem.

Forward process
    At noise level t in [0, 1], each position is independently masked with
    probability t. At t=0 the sequence is clean; at t=1 it is fully masked.
    Because masking is absorbing, q(x_t | x_0) is available in closed form,
    so training needs no sequential simulation.

Reverse process
    Start fully masked. At each of T steps, predict all masked positions, then
    commit the most confident fraction and re-mask the rest. The number of
    positions left masked follows a cosine schedule, so early steps commit few
    tokens (when context is scarce and predictions are unreliable) and later
    steps commit many. T trades quality against speed; it is the knob the
    benchmark sweeps.

Conditional generation
    Positions marked in `fixed` are clamped to supplied values and never
    re-masked, so a motif can be held constant while the model fills in the
    surrounding context conditioned on it.
"""
import numpy as np
import torch
import torch.nn.functional as F

MASK_TOKEN = 4
N_BASES = 4


def q_sample(x0, t, generator=None):
    """Mask each position independently with probability t.

    x0: (B, L) int64 clean tokens
    t:  (B,) float in [0, 1]
    Returns (x_t, mask) where mask is True at masked positions.
    """
    probs = t[:, None].expand_as(x0.float())
    noise = torch.rand(x0.shape, device=x0.device, generator=generator)
    mask = noise < probs
    # Guarantee at least one masked position per sequence, otherwise the
    # loss for that example is undefined and contributes nothing.
    empty = ~mask.any(dim=1)
    if empty.any():
        idx = torch.randint(0, x0.size(1), (int(empty.sum()),),
                            device=x0.device, generator=generator)
        mask[empty, idx] = True
    xt = x0.masked_fill(mask, MASK_TOKEN)
    return xt, mask


def loss_fn(model, x0, generator=None):
    """Cross-entropy at masked positions only, reweighted by 1/t.

    The 1/t weighting is the standard masked-diffusion ELBO weighting: a
    sample masked at t=0.05 has few masked positions, each easy; one at
    t=0.95 has many, each hard. Without reweighting the easy regime would
    dominate the gradient.
    """
    B = x0.size(0)
    t = torch.rand(B, device=x0.device, generator=generator)
    t = t.clamp(min=1e-3)
    xt, mask = q_sample(x0, t, generator=generator)

    logits = model(xt, t)                       # (B, L, 4)
    ce = F.cross_entropy(
        logits.reshape(-1, N_BASES),
        x0.reshape(-1),
        reduction="none",
    ).view(B, -1)

    masked_ce = (ce * mask.float()).sum(dim=1)
    n_masked = mask.float().sum(dim=1).clamp(min=1.0)
    # NOTE: the ELBO weighting (dividing by t) was tried first and made
    # training unstable -- t ~ U[0.001, 1] gives weights up to 1000x, so rare
    # small-t draws dominate the gradient. The unweighted "simple" objective
    # is used instead.
    per_seq = masked_ce / n_masked
    return per_seq.mean()


def _cosine_schedule(seq_len, n_steps):
    """Number of positions left masked after each step; ends at zero."""
    sched = [
        int(np.floor(seq_len * np.cos(np.pi / 2 * (i + 1) / n_steps)))
        for i in range(n_steps)
    ]
    sched[-1] = 0
    # Enforce monotonicity in case of floating-point ties.
    for i in range(1, len(sched)):
        sched[i] = min(sched[i], sched[i - 1])
    return sched


@torch.no_grad()
def sample(model, n, seq_len=50, n_steps=50, temperature=1.0,
           order="random", gumbel=0.0,
           fixed_tokens=None, fixed_mask=None, device="cpu", seed=None):
    """Generate `n` sequences as a (n, seq_len) int64 tensor.

    order: which masked positions to commit at each step.
      "random"     - uniform random subset. This is the unbiased reverse
                     process for absorbing-state diffusion: the forward
                     process masks positions independently of their content,
                     so the reverse must unmask them the same way.
      "confidence" - commit the highest-confidence positions (MaskGIT-style).
                     Improves per-sample plausibility but systematically
                     oversamples high-probability tokens, sharpening the
                     output distribution away from the data distribution.
    gumbel: scale of Gumbel noise added to confidence scores, annealed over
      steps. Only used when order="confidence"; partially counteracts the
      sharpening above.

    fixed_tokens: (seq_len,) or (n, seq_len) int64, values to clamp
    fixed_mask:   same shape, bool, True where the value is held fixed
    """
    model.eval()
    g = None
    if seed is not None:
        g = torch.Generator(device=device).manual_seed(seed)

    x = torch.full((n, seq_len), MASK_TOKEN, dtype=torch.long, device=device)

    if fixed_mask is not None:
        fixed_mask = fixed_mask.to(device)
        fixed_tokens = fixed_tokens.to(device)
        if fixed_mask.dim() == 1:
            fixed_mask = fixed_mask[None, :].expand(n, -1)
            fixed_tokens = fixed_tokens[None, :].expand(n, -1)
        x = torch.where(fixed_mask, fixed_tokens, x)

    schedule = _cosine_schedule(seq_len, n_steps)

    for step, n_keep_masked in enumerate(schedule):
        is_masked = x == MASK_TOKEN
        if not is_masked.any():
            break

        t_val = 1.0 - step / len(schedule)
        t = torch.full((n,), t_val, device=device)

        logits = model(x, t) / max(temperature, 1e-6)
        probs = F.softmax(logits, dim=-1)

        # Sample a candidate token at every position.
        flat = probs.view(-1, N_BASES)
        cand = torch.multinomial(flat, 1, generator=g).view(n, seq_len)
        conf = probs.gather(-1, cand[..., None]).squeeze(-1)

        # Scores decide which masked positions get committed this step.
        if order == "random":
            score = torch.rand(n, seq_len, device=device, generator=g)
        elif order == "confidence":
            score = conf
            if gumbel > 0:
                u = torch.rand(n, seq_len, device=device, generator=g)
                anneal = gumbel * (1.0 - step / max(len(schedule) - 1, 1))
                score = score + anneal * (-torch.log(-torch.log(u + 1e-9) + 1e-9))
        else:
            raise ValueError(f"unknown order={order!r}")
        # Only masked positions are eligible.
        score = score.masked_fill(~is_masked, -1e9)

        # Keep the `n_keep_masked` least-confident masked positions masked.
        n_masked_now = int(is_masked.sum(dim=1).max())
        n_commit = max(0, n_masked_now - n_keep_masked)

        new_x = torch.where(is_masked, cand, x)
        if n_commit < n_masked_now:
            # Re-mask the lowest-confidence positions.
            order_idx = score.argsort(dim=1, descending=True)
            rank = order_idx.argsort(dim=1)
            remask = (rank >= n_commit) & is_masked
            new_x = new_x.masked_fill(remask, MASK_TOKEN)

        x = new_x
        if fixed_mask is not None:
            x = torch.where(fixed_mask, fixed_tokens, x)

    # Safety net: fill any position still masked after the final step.
    still = x == MASK_TOKEN
    if still.any():
        t = torch.zeros(n, device=device)
        logits = model(x, t)
        fill = logits.argmax(dim=-1)
        x = torch.where(still, fill, x)

    return x.cpu()
