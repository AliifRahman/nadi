"""Train NADI. Output: artifacts/nadi.pt

  python train.py
      one edition (data/claims.parquet), 9 patients in 10; evaluate.py uses the rest
  python train.py --data data/claims_all.parquet --before 2024-01-01 --steps 10000
      several editions, only claims discharged before 2024; validate.py tests on 2024.
      This is the model the app uses
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import nadi

STEPS = 5000
WARMUP = 200
LR = 1e-3
TOKENS_PER_BATCH = 12_000  # fits the GPU of an 8 GB laptop
OUT = Path("artifacts/nadi.pt")


def batches(X: np.ndarray, n_visits: np.ndarray, rng: np.random.Generator):
    """Batches of similar-length sequences, sampled in proportion to their token count."""
    order = np.argsort(n_visits, kind="stable")
    lengths = n_visits[order]
    cdf = np.cumsum(lengths) / lengths.sum()
    while True:
        center = min(np.searchsorted(cdf, rng.random()), len(order) - 1)
        size = max(8, TOKENS_PER_BATCH // (lengths[center] * X.shape[2]))
        start = int(np.clip(center - size // 2, 0, len(order) - size))
        rows = order[start:start + size]
        yield X[rows, :lengths[start + size - 1]]  # drop trailing padding


def loss_on(model, X: np.ndarray) -> torch.Tensor:
    tokens = torch.from_numpy(X).long().flatten(1).to(nadi.DEVICE)
    return F.cross_entropy(model(tokens).transpose(1, 2), tokens, ignore_index=nadi.PAD)


def sequences(df: pd.DataFrame, vocab: dict):
    X, chunk, _ = nadi.to_sequences(nadi.tokenize(df, vocab), df.patient.to_numpy())
    return X, np.bincount(chunk)


def main(data: Path, out: Path, before, steps: int) -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    df = pd.read_parquet(data)
    test = nadi.is_test(df.patient)
    seen = np.ones(len(df), dtype=bool) if before is None else (df.discharge < before).to_numpy()
    fit, val = df[~test & seen], df[test & seen]
    vocab = nadi.build_vocab(nadi.features(part) for _, part in nadi.by_patient_group(fit))
    X_train, len_train = sequences(fit, vocab)
    X_val = sequences(val, vocab)[0][:256]
    del df, fit, val
    model = nadi.Nadi(len(vocab)).to(nadi.DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"vocab {len(vocab):,} | train claims {len_train.sum():,} in {len(X_train):,} sequences | "
          f"params {n_params / 1e6:.1f}M | {nadi.DEVICE}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    start = time.time()
    for step, batch in zip(range(steps + 1), batches(X_train, len_train, rng)):
        lr = LR * min(1, step / WARMUP) * 0.5 * (1 + math.cos(math.pi * step / steps))
        for group in optimizer.param_groups:
            group["lr"] = lr
        loss = loss_on(model, batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 500 == 0:
            model.eval()
            with torch.no_grad():
                val = np.mean([loss_on(model, X_val[i:i + 32]).item() for i in range(0, len(X_val), 32)])
            model.train()
            print(f"step {step:>5} | train loss {loss.item():.3f} | val loss {val:.3f} | {time.time() - start:>4.0f}s", flush=True)

    out.parent.mkdir(exist_ok=True)
    nadi.save(out, model, vocab)
    print(f"saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data/claims.parquet"))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--before", type=pd.Timestamp, help="train only on claims discharged before this date")
    parser.add_argument("--steps", type=int, default=STEPS)
    args = parser.parse_args()
    main(args.data, args.out, args.before, args.steps)
