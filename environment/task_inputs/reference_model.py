"""The exact model your work is scored against.

Same architecture, same schedule, same "accuracy at the best-validation epoch"
convention that the grader uses, so you can measure your own attack and your own
defense the way they will actually be measured instead of guessing.

Use it as a library:

    from reference_model import reference_accuracy
    acc = reference_accuracy(train_x, train_y, val_x, val_y, score_x, score_y,
                             weights=None)

or from the shell, to see what an attack costs the victim:

    python /app/task_inputs/reference_model.py --corpus probe_corpus \\
        --poisoned-x /app/results/attack/poisoned_train_x.npy \\
        --poisoned-y /app/results/attack/poisoned_train_y.npy

Protocol: 72 -> 128 -> 64 -> n_classes MLP, ReLU, dropout 0.2 on the first hidden
layer, Adam at lr 0.01 with weight decay 5e-4, 220 epochs full batch, averaged over
seeds 11/12/13. Accuracy is read off at the epoch with the best validation accuracy.

The grader runs this on the held-out test rows. Here you can only score on the
validation split, which is smaller and therefore noisier - leave yourself margin
above the bar rather than stopping the moment you touch it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SEEDS = (11, 12, 13)
HIDDEN = 128
DROPOUT = 0.2
EPOCHS = 220
LR = 0.01
WEIGHT_DECAY = 5e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ReferenceMLP(torch.nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden: int = HIDDEN,
                 dropout: float = DROPOUT):
        super().__init__()
        self.l1 = torch.nn.Linear(n_features, hidden)
        self.l2 = torch.nn.Linear(hidden, hidden // 2)
        self.l3 = torch.nn.Linear(hidden // 2, n_classes)
        self.dropout = dropout
        for layer in (self.l1, self.l2, self.l3):
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

    def forward(self, x):
        h = F.relu(self.l1(x))
        h = F.dropout(h, self.dropout, training=self.training)
        h = F.relu(self.l2(h))
        return self.l3(h)


def _tensor(array, device=DEVICE, dtype=torch.float32):
    return torch.as_tensor(np.asarray(array), device=device, dtype=dtype)


def reference_accuracy(train_x, train_y, val_x, val_y, score_x, score_y,
                       weights=None, seeds=SEEDS, epochs: int = EPOCHS) -> float:
    """Accuracy on score_x at the best-validation epoch, averaged over seeds.

    `weights` is an optional non-negative per-row trust weight. It rescales each
    row's contribution to the training loss; it never changes what is scored.
    """
    tx = _tensor(train_x)
    ty = _tensor(train_y, dtype=torch.long)
    vx, vy = _tensor(val_x), _tensor(val_y, dtype=torch.long)
    sx, sy = _tensor(score_x), _tensor(score_y, dtype=torch.long)

    w = None
    if weights is not None:
        w = _tensor(weights)
        if w.shape != (tx.shape[0],):
            raise ValueError(f"weights must have shape {(tx.shape[0],)}, got {tuple(w.shape)}")
        total = float(w.sum())
        if total <= 0:
            raise ValueError("weights sum to zero; there is nothing left to train on")
        w = w * (w.numel() / total)  # keep the effective learning rate comparable

    n_classes = int(max(int(ty.max()), int(vy.max()), int(sy.max()))) + 1

    scores = []
    for seed in seeds:
        torch.manual_seed(seed)
        model = ReferenceMLP(tx.shape[1], n_classes).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        best_val, best_score = -1.0, 0.0
        for _ in range(epochs):
            model.train()
            opt.zero_grad()
            per_row = F.cross_entropy(model(tx), ty, reduction="none")
            (per_row if w is None else per_row * w).mean().backward()
            opt.step()
            model.eval()
            with torch.no_grad():
                val = (model(vx).argmax(1) == vy).float().mean().item()
                if val > best_val:
                    best_val = val
                    best_score = (model(sx).argmax(1) == sy).float().mean().item()
        scores.append(best_score)
    return float(np.mean(scores))


def load_corpus(corpus_dir: Path) -> dict:
    out = {
        key: np.load(corpus_dir / f"{key}.npy")
        for key in ("train_x", "train_y", "val_x", "val_y", "test_x")
    }
    out["meta"] = json.loads((corpus_dir / "meta.json").read_text())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="probe_corpus")
    ap.add_argument("--root", default="/app/task_inputs/corpora")
    ap.add_argument("--poisoned-x", help="training features to compare against as-shipped")
    ap.add_argument("--poisoned-y", help="training labels to compare against as-shipped")
    ap.add_argument("--weights", help="per-row trust weights to score with")
    args = ap.parse_args()

    corpus = load_corpus(Path(args.root) / args.corpus)
    weights = np.load(args.weights) if args.weights else None

    # only the validation rows are labelled for you; the grader uses the held-out ones
    base = reference_accuracy(
        corpus["train_x"], corpus["train_y"], corpus["val_x"], corpus["val_y"],
        corpus["val_x"], corpus["val_y"], weights=weights,
    )
    print(f"{args.corpus}: reference val accuracy as shipped = {base:.4f}")

    if args.poisoned_x and args.poisoned_y:
        px, py = np.load(args.poisoned_x), np.load(args.poisoned_y)
        after = reference_accuracy(
            px, py, corpus["val_x"], corpus["val_y"], corpus["val_x"], corpus["val_y"],
        )
        rows_changed = int(
            (~np.all(np.isclose(px, corpus["train_x"], atol=1e-6), axis=1)
             | (py != corpus["train_y"])).sum()
        )
        budget = round(0.08 * corpus["train_x"].shape[0])
        print(f"{args.corpus}: after your perturbation = {after:.4f}")
        print(f"drop = {base - after:.4f} over {rows_changed} modified rows "
              f"(budget {budget})")


if __name__ == "__main__":
    main()
