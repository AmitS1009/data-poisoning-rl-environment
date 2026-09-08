"""Reference implementations: the budgeted poisoning attack and the trust-weight learner.

Nothing here is imported by the verifier. This is one way to solve the task, kept
separate from the driver so the driver stays readable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HIDDEN = 128
DROPOUT = 0.2
LR = 0.01
WEIGHT_DECAY = 5e-4


# ------------------------------------------------------------------------- model


class MLP(torch.nn.Module):
    """Same shape as the reference victim, so measurements transfer directly."""

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

    def features(self, x):
        h = F.relu(self.l1(x))
        h = F.dropout(h, self.dropout, training=self.training)
        return F.relu(self.l2(h))

    def forward(self, x):
        return self.l3(self.features(x))


def load_corpus(path: Path, device: str) -> dict:
    out = {}
    for key in ("train_x", "val_x", "test_x"):
        out[key] = torch.as_tensor(np.load(path / f"{key}.npy"), device=device,
                                   dtype=torch.float32)
    for key in ("train_y", "val_y"):
        out[key] = torch.as_tensor(np.load(path / f"{key}.npy"), device=device,
                                   dtype=torch.long)
    out["meta"] = json.loads((path / "meta.json").read_text())
    return out


def train_model(train_x, train_y, val_x, val_y, n_classes, device, weights=None,
                epochs: int = 220, seed: int = 11, score_x=None):
    """Train to the best-validation epoch. Returns (model_state, logits_on_score_x, val_acc).

    `weights` is renormalised to mean 1 so that dropping rows does not quietly shrink
    the effective learning rate and confound the ablation.
    """
    torch.manual_seed(seed)
    model = MLP(train_x.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    w = None
    if weights is not None:
        w = weights.detach().clamp(min=0)
        total = w.sum()
        w = w * (w.numel() / total.clamp(min=1e-9))

    score_x = val_x if score_x is None else score_x
    best = {"val": -1.0, "state": None, "logits": None}
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        per_row = F.cross_entropy(model(train_x), train_y, reduction="none")
        (per_row if w is None else per_row * w).mean().backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val = (model(val_x).argmax(1) == val_y).float().mean().item()
            if val > best["val"]:
                best = {
                    "val": val,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
                    "logits": model(score_x).detach().clone(),
                }
    return best["state"], best["logits"], best["val"]


# ------------------------------------------------------------------------ attack


class BudgetedPoisoner:
    """Greedy meta-gradient label poisoning inside a row budget.

    For a row i with label c, the model's output-layer gradient is
    outer(softmax_i - onehot_c, h_i). Its inner product with the gradient of the
    trusted validation loss says which way training on that row moves validation
    loss, so relabelling i from y to c changes that alignment by
    A[i, y] - A[i, c], where A = h @ g_val.T. Both terms are one matmul for the whole
    corpus, so every (row, label) pair is scored at once instead of by n separate
    backward passes.

    Picking the argmax over c and the top rows by A[i, c*] - A[i, y] is a single-shot
    attack. Doing it in rounds - refit, rescore, spend the next slice of budget - is
    meaningfully stronger, because the first flips change which rows are worth
    flipping next. Eight rounds measured best; past that the greedy signal starts
    chasing its own damage and the attack gets weaker again.

    Feature perturbation is permitted by the budget and is deliberately not used. A
    dense step across standardised features turns a poisoned row into an obvious
    outlier that the victim discounts, and it measured *worse* than leaving the
    features alone. `eps` is kept so the behaviour can be reproduced.
    """

    def __init__(self, rounds: int = 8, inner_epochs: int = 140, eps: float = 0.0,
                 seed: int = 11):
        self.rounds = rounds
        self.inner_epochs = inner_epochs
        self.eps = eps
        self.seed = seed

    @staticmethod
    def _alignment(model, train_x, val_x, val_y):
        """A[i, c] = <gradient of val loss wrt output row c, hidden features of row i>."""
        model.zero_grad(set_to_none=True)
        model.eval()
        F.cross_entropy(model(val_x), val_y).backward()
        g_val = model.l3.weight.grad.detach().clone()
        with torch.no_grad():
            return model.features(train_x) @ g_val.T

    def attack(self, corpus: dict, n_classes: int, budget_rows: int, device: str):
        train_x, train_y = corpus["train_x"], corpus["train_y"]
        val_x, val_y = corpus["val_x"], corpus["val_y"]
        n = train_x.shape[0]

        labels = train_y.clone()
        spent = torch.zeros(n, dtype=torch.bool, device=device)
        per_round = max(1, budget_rows // self.rounds)

        for r in range(self.rounds):
            take = per_round if r < self.rounds - 1 else budget_rows - int(spent.sum())
            if take <= 0:
                break
            state, _, _ = train_model(train_x, labels, val_x, val_y, n_classes, device,
                                      epochs=self.inner_epochs, seed=self.seed)
            model = MLP(train_x.shape[1], n_classes).to(device)
            model.load_state_dict(state)

            align = self._alignment(model, train_x, val_x, val_y)
            current = align.gather(1, labels[:, None]).squeeze(1)
            alternatives = align.clone()
            alternatives.scatter_(1, labels[:, None], float("-inf"))
            damage = alternatives.max(1).values - current
            damage[spent] = float("-inf")

            chosen = torch.topk(damage, take).indices
            labels[chosen] = alternatives[chosen].argmax(1)
            spent[chosen] = True

        features = train_x.clone()
        if self.eps:
            victims = torch.nonzero(spent).squeeze(1)
            adv = train_x[victims].clone().requires_grad_(True)
            F.cross_entropy(model(adv), labels[victims]).backward()
            features[victims] = train_x[victims] + self.eps * adv.grad.sign()

        return features.detach(), labels, torch.nonzero(spent).squeeze(1)


# ----------------------------------------------------------------------- defense


def knn_agreement(train_x: torch.Tensor, train_y: torch.Tensor, k: int = 25,
                  chunk: int = 512) -> torch.Tensor:
    """Fraction of a row's k nearest neighbours that carry the same label.

    Chunked because the full pairwise distance matrix is the one thing in this task
    that would actually run out of memory on a small GPU.
    """
    n = train_x.shape[0]
    out = torch.empty(n, device=train_x.device)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        d = torch.cdist(train_x[start:stop], train_x)
        d[torch.arange(stop - start, device=d.device), torch.arange(start, stop,
                                                                   device=d.device)] = float("inf")
        idx = d.topk(k, largest=False).indices
        out[start:stop] = (train_y[idx] == train_y[start:stop, None]).float().mean(1)
    return out


class TrustLearner:
    """Learn a per-row trust weight alongside the model.

    Three signals, each of which is the one that carries a different corpus:

    `meta`        - alignment between a row's gradient and the gradient of the trusted
                    validation split. A row that pulls the model the same way the
                    trusted data does is earning its place. This is the signal that
                    sees a poison whose rows are individually unremarkable and only
                    wrong in aggregate.
    `consistency` - agreement with the labels of the row's feature-space neighbours,
                    which catches rows sitting in the wrong neighbourhood no matter
                    what the loss says about them.
    `loss`        - the cheap one: rows the model cannot fit. On its own it is the
                    weakest of the three, and the ablation is built to show that.

    `l1` is not a fourth signal but the pressure that turns a ranking into a decision:
    it shifts the cut so a row has to clear a bar rather than merely be above average.
    Dropping any one of the four changes the weights, and changes them differently on
    different corpora, which is what makes the ablation informative rather than
    decorative.

    Weights are soft (a sigmoid, never exactly zero) on purpose. A hard mask throws
    away the distinction between "certainly poisoned" and "mildly suspect", and that
    distinction is most of what the ranking is measured on.
    """

    def __init__(self, n_classes: int, device: str, rounds: int = 3,
                 inner_epochs: int = 60, base_drop: float = 0.08, l1: float = 0.17,
                 temp: float = 0.35, consistency: float = 1.0, meta: float = 1.0,
                 loss_weight: float = 0.6, seed: int = 11):
        self.n_classes = n_classes
        self.device = device
        self.rounds = rounds
        self.inner_epochs = inner_epochs
        self.base_drop = base_drop
        self.l1 = l1
        self.temp = temp
        self.consistency = consistency
        self.meta = meta
        self.loss_weight = loss_weight
        self.seed = seed

    @staticmethod
    def _z(v: torch.Tensor) -> torch.Tensor:
        return (v - v.mean()) / v.std().clamp(min=1e-9)

    def _influence(self, model, train_x, train_y, val_x, val_y):
        """<per-row gradient, validation gradient> on the output layer, in closed form.

        The output-layer gradient of row i is outer(softmax_i - onehot_i, h_i), so its
        inner product with the validation gradient is one matmul instead of n separate
        backward passes. Positive means the row pushes the model the same way the
        trusted split does.
        """
        model.zero_grad(set_to_none=True)
        model.eval()
        F.cross_entropy(model(val_x), val_y).backward()
        g_val = model.l3.weight.grad.detach().clone()

        with torch.no_grad():
            hidden = model.features(train_x)
            resid = model(train_x).softmax(1)
            resid[torch.arange(train_y.shape[0], device=resid.device), train_y] -= 1.0
            return ((resid @ g_val) * hidden).sum(1)

    def _score(self, model, train_x, train_y, val_x, val_y, agree):
        parts = []
        if self.meta:
            parts.append(self.meta * self._z(
                self._influence(model, train_x, train_y, val_x, val_y)))
        if self.loss_weight:
            model.eval()
            with torch.no_grad():
                loss = F.cross_entropy(model(train_x), train_y, reduction="none")
            parts.append(self.loss_weight * self._z(-loss))
        if self.consistency and agree is not None:
            parts.append(self.consistency * self._z(agree))
        if not parts:  # every signal ablated away: fall back to trusting everything
            return torch.zeros(train_x.shape[0], device=train_x.device)
        return torch.stack(parts).sum(0)

    def _weights(self, score: torch.Tensor) -> torch.Tensor:
        drop = min(0.85, max(0.0, self.base_drop + self.l1))
        if score.std() < 1e-8:
            return torch.ones_like(score)
        tau = torch.quantile(score, drop)
        return torch.sigmoid((score - tau) / self.temp)

    def fit(self, corpus: dict, epochs: int = 220):
        train_x, train_y = corpus["train_x"], corpus["train_y"]
        val_x, val_y = corpus["val_x"], corpus["val_y"]

        agree = knn_agreement(train_x, train_y) if self.consistency else None
        weights = torch.ones(train_x.shape[0], device=self.device)

        # Refit a few times: the trust scores are read off a model that was itself
        # trained under the current weights, so down-weighting the worst rows sharpens
        # the estimate for the next pass. It converges quickly; more rounds mostly buy
        # runtime.
        for _ in range(self.rounds):
            state, _, _ = train_model(train_x, train_y, val_x, val_y, self.n_classes,
                                      self.device, weights=weights,
                                      epochs=self.inner_epochs, seed=self.seed)
            model = MLP(train_x.shape[1], self.n_classes).to(self.device)
            model.load_state_dict(state)
            score = self._score(model, train_x, train_y, val_x, val_y, agree)
            weights = self._weights(score)

        # score validation and test in one forward pass, then split: the reported
        # validation accuracy has to come from the same model as the test predictions,
        # and the verifier checks exactly that
        both = torch.cat([val_x, corpus["test_x"]], dim=0)
        state, logits, val_acc = train_model(
            train_x, train_y, val_x, val_y, self.n_classes, self.device,
            weights=weights, epochs=epochs, seed=self.seed, score_x=both,
        )
        n_val = val_x.shape[0]
        return weights.detach(), state, logits[n_val:], logits[:n_val], val_acc
