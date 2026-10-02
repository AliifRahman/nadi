"""Train NADI on data/claims.parquet. Output: artifacts/nadi.pt"""
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
    tokens = torch.from_numpy(X).flatten(1).to(nadi.DEVICE)
    return F.cross_entropy(model(tokens).transpose(1, 2), tokens, ignore_index=nadi.PAD)


def main() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    df = pd.read_parquet("data/claims.parquet")
    feats = nadi.features(df)
    test = nadi.is_test(df.patient)
    vocab = nadi.build_vocab(feats[~test])
    X, chunk, _ = nadi.to_sequences(nadi.encode(feats, vocab), df.patient.to_numpy())
    n_visits = np.bincount(chunk)
    chunk_is_test = np.zeros(len(X), dtype=bool)
    chunk_is_test[chunk] = test

    X_train, len_train = X[~chunk_is_test], n_visits[~chunk_is_test]
    X_val = X[chunk_is_test][:256]
    model = nadi.Nadi(len(vocab)).to(nadi.DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"vocab {len(vocab):,} | train sequences {len(X_train):,} | "
          f"params {n_params / 1e6:.1f}M | {nadi.DEVICE}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    start = time.time()
    for step, batch in zip(range(STEPS + 1), batches(X_train, len_train, rng)):
        lr = LR * min(1, step / WARMUP) * 0.5 * (1 + math.cos(math.pi * step / STEPS))
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
            print(f"step {step:>5} | train loss {loss.item():.3f} | val loss {val:.3f} | {time.time() - start:>4.0f}s")

    OUT.parent.mkdir(exist_ok=True)
    nadi.save(OUT, model, vocab)
    print(f"saved to {OUT}")


if __name__ == "__main__":
    main()
