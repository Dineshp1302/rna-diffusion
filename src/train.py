"""Train the masked diffusion denoiser.

Usage:
    python src/train.py --epochs 12 --batch-size 256
    python src/train.py --smoke        # 60-second correctness check
"""
import argparse
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from diffusion import loss_fn, sample  # noqa: E402
from model import SeqDiffusionTransformer  # noqa: E402
from tokenizer import decode, encode, load  # noqa: E402


def evaluate_loss(model, data, batch_size, device, seed=0):
    """Held-out loss. Fixed seed so the noise draw is identical across
    epochs -- otherwise epoch-to-epoch variation is mostly sampling noise
    in t, not real change in the model."""
    model.eval()
    g = torch.Generator(device=device).manual_seed(seed)
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            xb = data[i:i + batch_size].to(device)
            total += loss_fn(model, xb, generator=g).item() * len(xb)
            n += len(xb)
    model.train()
    return total / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", type=str, default="results/model.pt")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run to verify the loss decreases")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train = load("data/train.txt", args.seq_len)
    test = load("data/test.txt", args.seq_len)
    if args.smoke:
        train, test = train[:2000], test[:500]
        args.epochs = 3

    xtr = torch.from_numpy(encode(train))
    xte = torch.from_numpy(encode(test))

    model = SeqDiffusionTransformer(
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    ).to(device)
    print(f"device={device} threads={args.threads} "
          f"params={model.n_params():,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = max(1, len(xtr) // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch,
        pct_start=0.1,
    )

    # Baseline: a model that has learned nothing predicts uniform over 4
    # bases, giving ln(4) ~= 1.386 nats before the 1/t weighting.
    print(f"uniform-prediction reference: {np.log(4):.4f} nats\n")

    g = torch.Generator(device=device).manual_seed(args.seed)
    best = float("inf")
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        perm = torch.randperm(len(xtr))
        running, nb = 0.0, 0
        for i in range(0, steps_per_epoch * args.batch_size, args.batch_size):
            xb = xtr[perm[i:i + args.batch_size]].to(device)
            if len(xb) == 0:
                continue
            loss = loss_fn(model, xb, generator=g)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()
            nb += 1

        val = evaluate_loss(model, xte, args.batch_size, device)
        dt = time.perf_counter() - t0
        flag = ""
        if val < best:
            best = val
            torch.save({"state_dict": model.state_dict(),
                        "args": vars(args)}, args.out)
            flag = "  <- saved"
        print(f"epoch {epoch + 1:3d}/{args.epochs}  "
              f"train {running / max(nb, 1):.4f}  "
              f"val {val:.4f}  {dt:.1f}s{flag}")

    print(f"\nbest val loss {best:.4f} -> {args.out}")

    x = sample(model, 4, seq_len=args.seq_len, n_steps=50,
               device=device, seed=0)
    print("\nsample generations:")
    for s in decode(x.numpy()):
        print(" ", s)


if __name__ == "__main__":
    main()
