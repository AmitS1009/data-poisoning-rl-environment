"""Measurement and validation helpers for the verifier.

Two rules this file follows, and they are the reason it exists separately from the
test module:

1. It measures; it never compares against a pass bar. Every threshold lives in
   thresholds.json and is applied by the tests. A number that appears in two places
   drifts, and a drifted threshold is worse than a wrong one because it looks fine.

2. It never imports anything from /app. The agent can write to /app, including to
   task_inputs/reference_model.py, so the verifier carries its own copy of the
   reference model rather than importing the one the agent was handed. The two are
   kept identical by hand, and test_reference_model_parity checks that they still
   agree on the protocol constants.

Nothing here trusts a number the agent reported. Accuracies are recomputed from the
submitted artifacts against held-out labels the container never sees.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata

TESTS_DIR = Path(__file__).resolve().parent
APP_DIR = Path("/app")
if not (APP_DIR / "task_inputs").exists():  # local development run outside the container
    APP_DIR = TESTS_DIR.parent / "environment"

CORPORA = APP_DIR / "task_inputs" / "corpora"
HIDDEN = TESTS_DIR / "hidden"
# RESULTS_DIR lets the adversarial harness point the verifier at a mutated copy of a
# result tree without touching the real one. The graded path never sets it.
RESULTS = Path(os.environ.get("RESULTS_DIR", "/app/results"))
if not RESULTS.exists():
    RESULTS = TESTS_DIR.parent / "results"

DEFENDED = ("alpha_feed", "beta_feed", "gamma_feed")
ATTACK_CORPUS = "probe_corpus"

# Reference protocol. Must stay identical to environment/task_inputs/reference_model.py,
# which ships the same numbers to the agent so their measurements match ours.
SEEDS = (11, 12, 13)
HIDDEN_UNITS = 128
DROPOUT = 0.2
EPOCHS = 220
LR = 0.01
WEIGHT_DECAY = 5e-4


def thresholds() -> dict:
    return json.loads((TESTS_DIR / "thresholds.json").read_text())


# ------------------------------------------------------------------ reference model


class ReferenceMLP(torch.nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.l1 = torch.nn.Linear(n_features, HIDDEN_UNITS)
        self.l2 = torch.nn.Linear(HIDDEN_UNITS, HIDDEN_UNITS // 2)
        self.l3 = torch.nn.Linear(HIDDEN_UNITS // 2, n_classes)
        for layer in (self.l1, self.l2, self.l3):
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

    def forward(self, x):
        h = F.relu(self.l1(x))
        h = F.dropout(h, DROPOUT, training=self.training)
        h = F.relu(self.l2(h))
        return self.l3(h)


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def reference_accuracy(train_x, train_y, val_x, val_y, score_x, score_y,
                       weights=None, seeds=SEEDS, epochs: int = EPOCHS) -> float:
    """Accuracy on score_x at the best-validation epoch, averaged over seeds."""
    dev = device()

    def t(a, dtype=torch.float32):
        return torch.as_tensor(np.asarray(a), device=dev, dtype=dtype)

    tx, ty = t(train_x), t(train_y, torch.long)
    vx, vy = t(val_x), t(val_y, torch.long)
    sx, sy = t(score_x), t(score_y, torch.long)

    w = None
    if weights is not None:
        w = t(weights).clamp(min=0)
        total = float(w.sum())
        if total <= 0:
            raise ValueError("weights sum to zero")
        w = w * (w.numel() / total)

    n_classes = int(max(int(ty.max()), int(vy.max()), int(sy.max()))) + 1

    scores = []
    for seed in seeds:
        torch.manual_seed(seed)
        model = ReferenceMLP(tx.shape[1], n_classes).to(dev)
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


# --------------------------------------------------------------------------- data


def load_public(name: str, root: Path | None = None) -> dict:
    root = (root or CORPORA) / name
    out = {
        key: np.load(root / f"{key}.npy")
        for key in ("train_x", "train_y", "val_x", "val_y", "test_x")
    }
    out["meta"] = json.loads((root / "meta.json").read_text())
    return out


def load_hidden(name: str, root: Path | None = None) -> dict:
    root = (root or HIDDEN) / name
    return {
        "test_y": np.load(root / "test_y.npy"),
        "poison_idx": np.load(root / "poison_idx.npy"),
    }


def poison_mask(n_train: int, poison_idx: np.ndarray) -> np.ndarray:
    mask = np.zeros(n_train, dtype=bool)
    mask[poison_idx] = True
    return mask


# ------------------------------------------------------------------------ metrics


def rank_auc(scores: np.ndarray, poisoned: np.ndarray) -> float:
    """P(a poisoned row outranks a clean row) under `scores`, higher score = trusted.

    Ranks rather than raw weights, on purpose. Comparing the mean weight of two groups
    is meaningless once a learner shrinks everything toward zero, and a survival
    threshold has the same problem from the other side. Ranks are scale-free, so any
    monotone rescaling of the weights - including multiplying them all by a constant -
    leaves the number unchanged. Ties take average ranks, which is what makes a
    uniform weight vector score exactly 0.5 rather than something accidentally
    favourable.
    """
    if not poisoned.any() or poisoned.all():
        return 1.0
    ranks = rankdata(scores)
    a, b = ranks[poisoned], ranks[~poisoned]
    return float((a.mean() - (a.size + 1) / 2) / b.size)


def effective_sample_size(weights: np.ndarray) -> float:
    """(sum w)^2 / sum w^2 - how many rows the weighting is really training on.

    This replaced a straight count of non-zero weights, which the source task used on
    a sparse adjacency. Counting non-zeros cannot fail against a sigmoid weight
    parameterisation: every weight is strictly positive, so the count is always n and
    the gate passes by construction. ESS is scale-free like the rank AUC, and it is
    not moved by sprinkling epsilon over rows the agent has actually discarded.
    """
    w = np.asarray(weights, dtype=np.float64)
    denom = np.square(w).sum()
    if denom <= 0:
        return 0.0
    return float(w.sum() ** 2 / denom)


# ------------------------------------------------------------------- artifact reads


def read_predictions(path: Path, n_rows: int, n_classes: int) -> np.ndarray:
    pred = np.load(path)
    if pred.shape != (n_rows,):
        raise AssertionError(f"{path} has shape {pred.shape}, expected {(n_rows,)}")
    if not np.issubdtype(pred.dtype, np.integer):
        if np.issubdtype(pred.dtype, np.floating) and np.all(pred == np.rint(pred)):
            pred = pred.astype(np.int64)
        else:
            raise AssertionError(f"{path} must hold integer class ids, got {pred.dtype}")
    if pred.min() < 0 or pred.max() >= n_classes:
        raise AssertionError(f"{path} has class ids outside [0, {n_classes})")
    if np.unique(pred).size < 2:
        raise AssertionError(f"{path} predicts a single class for every row")
    return pred


def read_weights(path: Path, n_rows: int) -> np.ndarray:
    w = np.load(path)
    if w.shape != (n_rows,):
        raise AssertionError(f"{path} has shape {w.shape}, expected {(n_rows,)}")
    if not np.issubdtype(w.dtype, np.floating):
        raise AssertionError(f"{path} must hold floats, got {w.dtype}")
    if not np.all(np.isfinite(w)):
        raise AssertionError(f"{path} contains non-finite weights")
    if w.min() < 0:
        raise AssertionError(f"{path} contains negative weights")
    if float(w.max()) <= 0:
        raise AssertionError(f"{path} is all zeros; nothing is trusted")
    if float(np.std(w)) <= 1e-12:
        raise AssertionError(
            f"{path} gives every row the same weight; a uniform vector is not a diagnosis "
            "and scores exactly 0.5 on the ranking test by construction"
        )
    return w.astype(np.float64)


# ---------------------------------------------------------------------- checkpoint


def load_checkpoint(path: Path) -> dict:
    """Load without map_location, so tensors come back on the device they were saved from.

    weights_only=True: the verifier will not execute pickled code from an artifact it
    is grading.
    """
    if not path.exists():
        raise AssertionError(f"{path} not produced")
    obj = torch.load(path, weights_only=True)
    if not isinstance(obj, dict) or "state_dict" not in obj or "logits" not in obj:
        raise AssertionError(f"{path} must hold {{'state_dict': ..., 'logits': ...}}")
    return obj


def checkpoint_parameters(state: dict) -> list:
    return [t for t in state.values() if torch.is_tensor(t) and t.is_floating_point()]


def logits_look_fabricated(logits: torch.Tensor, pred: np.ndarray) -> str | None:
    """Detect a logits tensor synthesised backwards from an answer vector.

    A real model's logits carry structure a hand-built tensor does not: the runner-up
    margin varies row to row, the non-argmax classes are not interchangeable, and the
    row-wise spread is not identical everywhere. Something built as `one_hot(pred) * c`
    fails all three at once.

    This does not prove the logits came from the shipped checkpoint - nothing short of
    running the agent's model would, and running it means importing the agent's code.
    It raises the cost of fabricating an artifact from a hardcoded answer, which is the
    vector that matters.
    """
    x = logits.detach().float().cpu()
    if x.ndim != 2 or x.shape[0] < 2:
        return "logits are not a 2-D tensor with more than one row"

    top2 = x.topk(2, dim=1).values
    margin = (top2[:, 0] - top2[:, 1]).numpy()
    if float(np.std(margin)) <= 1e-6:
        return (f"every row has the same top-two margin ({margin[0]:.6g}); real logits "
                "vary row to row")

    spread = x.std(dim=1).numpy()
    if float(np.std(spread)) <= 1e-6:
        return "every row has an identical spread; the tensor is a rescaled one-hot"

    # after removing the argmax class, the remaining logits should not be constant
    n, k = x.shape
    if k > 2:
        mask = torch.ones_like(x, dtype=torch.bool)
        mask[torch.arange(n), torch.as_tensor(pred, dtype=torch.long)] = False
        rest = x[mask].reshape(n, k - 1)
        if float(rest.std(dim=1).mean()) <= 1e-6:
            return "the non-predicted classes are all equal; the tensor encodes only the answer"
    return None


def state_dict_fingerprint(state: dict) -> str:
    """A cheap content hash, used to tell whether two checkpoints are the same bytes."""
    import hashlib

    h = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key]
        if torch.is_tensor(tensor):
            h.update(key.encode())
            h.update(tensor.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


# -------------------------------------------------------------------- attack budget


def attack_diff(original_x, original_y, new_x, new_y) -> dict:
    """Row-level and per-feature accounting for the agent's poisoned corpus."""
    if new_x.shape != original_x.shape:
        raise AssertionError(
            f"poisoned features have shape {new_x.shape}, expected {original_x.shape}")
    if new_y.shape != original_y.shape:
        raise AssertionError(
            f"poisoned labels have shape {new_y.shape}, expected {original_y.shape}")
    if not np.all(np.isfinite(new_x)):
        raise AssertionError("poisoned features contain non-finite values")

    delta = np.abs(new_x.astype(np.float64) - original_x.astype(np.float64))
    feature_changed = delta > 1e-6
    label_changed = new_y != original_y
    row_changed = feature_changed.any(axis=1) | label_changed

    return {
        "rows_modified": int(row_changed.sum()),
        "max_abs_perturbation": float(delta.max()) if delta.size else 0.0,
        "labels_changed": int(label_changed.sum()),
        "row_changed": row_changed,
    }
