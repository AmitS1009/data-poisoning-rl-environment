"""How visible is each poison to each cheap detector?

Produces the table the README cites. The point of the three corpora is that they are
not equally visible to the same signal, so a defense built on one heuristic is strong
somewhere and weak somewhere else. This measures that rather than asserting it.

Reported as rank AUC: the chance that a poisoned row outranks a clean one under the
detector's trust score. 0.5 is blind, lower is better.

    python scripts/detectability.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "solution"))

import verifier_lib as V  # noqa: E402
from defense_lib import MLP, knn_agreement, train_model  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def detector_scores(corpus: dict, n_classes: int) -> dict[str, np.ndarray]:
    """Three trust scores for the same trained model. Higher means more trusted."""
    train_x = torch.as_tensor(corpus["train_x"], device=DEVICE)
    train_y = torch.as_tensor(corpus["train_y"], device=DEVICE, dtype=torch.long)
    val_x = torch.as_tensor(corpus["val_x"], device=DEVICE)
    val_y = torch.as_tensor(corpus["val_y"], device=DEVICE, dtype=torch.long)

    state, _, _ = train_model(train_x, train_y, val_x, val_y, n_classes, DEVICE,
                              epochs=220, seed=11)
    model = MLP(train_x.shape[1], n_classes).to(DEVICE)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        loss = F.cross_entropy(model(train_x), train_y, reduction="none")

    agree = knn_agreement(train_x, train_y)

    model.zero_grad(set_to_none=True)
    F.cross_entropy(model(val_x), val_y).backward()
    g_val = model.l3.weight.grad.detach()
    with torch.no_grad():
        hidden = model.features(train_x)
        resid = model(train_x).softmax(1)
        resid[torch.arange(train_y.shape[0], device=DEVICE), train_y] -= 1.0
        influence = ((resid @ g_val) * hidden).sum(1)

    return {
        "training_loss": (-loss).cpu().numpy(),
        "knn_label_agreement": agree.cpu().numpy(),
        "validation_gradient_alignment": influence.cpu().numpy(),
    }


def main() -> None:
    table = {}
    for name in V.DEFENDED:
        pub, hid = V.load_public(name), V.load_hidden(name)
        mask = V.poison_mask(pub["meta"]["n_train"], hid["poison_idx"])
        scores = detector_scores(pub, pub["meta"]["n_classes"])
        row = {k: round(V.rank_auc(v, mask), 4) for k, v in scores.items()}
        combined = sum(
            (v - v.mean()) / (v.std() + 1e-9) for v in scores.values()
        )
        row["all_three_combined"] = round(V.rank_auc(combined, mask), 4)
        row["n_poisoned"] = int(mask.sum())
        row["poison_rate"] = round(float(mask.mean()), 4)
        table[name] = row

    out = REPO / "data" / "detectability.json"
    out.write_text(json.dumps(table, indent=2) + "\n")

    cols = ["training_loss", "knn_label_agreement", "validation_gradient_alignment",
            "all_three_combined"]
    header = f"{'corpus':12s} {'rate':>6s} " + " ".join(f"{c[:22]:>22s}" for c in cols)
    print(header)
    print("-" * len(header))
    for name, row in table.items():
        print(f"{name:12s} {row['poison_rate']:6.0%} "
              + " ".join(f"{row[c]:22.3f}" for c in cols))
    print(f"\nwrote {out}")
    print("rank AUC: chance a poisoned row outranks a clean one. 0.5 is blind.")


if __name__ == "__main__":
    main()
