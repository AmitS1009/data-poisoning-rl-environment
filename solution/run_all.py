"""Reference solution driver.

1. poisons probe_corpus with a greedy meta-gradient label attack inside the budget
2. runs the trust-weight learner on each of the three tampered corpora
3. runs the leave-one-term-out ablation on alpha_feed
4. writes weights, predictions, checkpoints, metrics.json and report.md

Requires a GPU. There is no CPU fallback here on purpose: the verifier audits that the
checkpoints were saved from a CUDA device, so a CPU run could not pass anyway and a
silent fallback would only turn a clear failure into a confusing one.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from defense_lib import BudgetedPoisoner, TrustLearner, load_corpus, train_model  # noqa: E402

APP_DIR = Path(os.environ.get("APP_DIR", "/app"))
RESULTS = APP_DIR / "results"
INPUTS = APP_DIR / "task_inputs" / "corpora"
if not INPUTS.exists():  # local development run outside the container
    INPUTS = Path(__file__).resolve().parent.parent / "environment" / "task_inputs" / "corpora"

ATTACK_CORPUS = "probe_corpus"
DEFENDED = ("alpha_feed", "beta_feed", "gamma_feed")
ABLATION_CORPUS = "alpha_feed"
ATTACK_ROW_BUDGET = 0.08
SEEDS = (11, 12, 13)

# Chosen on the validation split during development, never on the held-out rows.
DEFENSE_CFG = {"l1": 0.17, "temp": 0.35, "base_drop": 0.08}

# Each variant removes exactly one term. `full` must exist; the grader looks for it.
ABLATION = {
    "full": {},
    "no_meta": {"meta": 0.0},
    "no_consistency": {"consistency": 0.0},
    "no_l1": {"l1": 0.0},
}


def fit_ensemble(corpus: dict, n_classes: int, device: str, **overrides):
    """Average the learner over several seeds.

    One run picks its best-validation epoch off 800 rows, and which epoch that lands
    on moves both the accuracy and the surviving weights by a couple of points between
    machines. Averaging the weights and the logits takes that variance out without
    changing the method. The checkpoint kept is the single best seed's, since a mean
    of three state_dicts is not a model that ever existed.
    """
    weights, logits, val_logits, vals, states = [], [], [], [], []
    for seed in SEEDS:
        cfg = dict(DEFENSE_CFG, **overrides)
        learner = TrustLearner(n_classes, device, seed=seed, **cfg)
        w, state, lg, vlg, val = learner.fit(corpus)
        weights.append(w)
        logits.append(lg)
        val_logits.append(vlg)
        vals.append(val)
        states.append(state)

    mean_w = torch.stack(weights).mean(0)
    mean_logits = torch.stack(logits).mean(0)
    mean_val_logits = torch.stack(val_logits).mean(0)
    best = int(np.argmax(vals))
    # the reported accuracy has to be the ensemble's, not the mean of the members':
    # the verifier recomputes it from the val predictions we are about to ship
    val_pred = mean_val_logits.argmax(1)
    val_acc = float((val_pred == corpus["val_y"]).float().mean().item())
    return mean_w, mean_logits, mean_val_logits, val_acc, states[best]


def run_attack(device: str, metrics: dict) -> None:
    print(f"== attack: poisoning {ATTACK_CORPUS} ==", flush=True)
    corpus = load_corpus(INPUTS / ATTACK_CORPUS, device)
    n_classes = corpus["meta"]["n_classes"]
    n_rows = corpus["train_x"].shape[0]
    budget = round(ATTACK_ROW_BUDGET * n_rows)

    _, clean_logits, clean_val = train_model(
        corpus["train_x"], corpus["train_y"], corpus["val_x"], corpus["val_y"],
        n_classes, device, epochs=220, seed=SEEDS[0],
    )

    t0 = time.time()
    poisoner = BudgetedPoisoner(seed=SEEDS[0])
    new_x, new_y, victims = poisoner.attack(corpus, n_classes, budget, device)
    print(f"  {len(victims)} rows relabelled in {time.time() - t0:.0f}s", flush=True)

    _, _, poisoned_val = train_model(
        new_x, new_y, corpus["val_x"], corpus["val_y"], n_classes, device,
        epochs=220, seed=SEEDS[0],
    )
    print(f"  victim val acc {clean_val:.4f} -> {poisoned_val:.4f}", flush=True)

    out = RESULTS / "attack"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "poisoned_train_x.npy", new_x.cpu().numpy().astype(np.float32))
    np.save(out / "poisoned_train_y.npy", new_y.cpu().numpy().astype(np.int64))

    changed = (~np.all(np.isclose(new_x.cpu().numpy(), corpus["train_x"].cpu().numpy(),
                                  atol=1e-6), axis=1)
               | (new_y.cpu().numpy() != corpus["train_y"].cpu().numpy()))
    metrics["attack"] = {
        "victim_val_acc_clean": clean_val,
        "victim_val_acc_poisoned": poisoned_val,
        "rows_modified": int(changed.sum()),
        "row_budget": budget,
    }


def run_defense(device: str, metrics: dict) -> None:
    metrics["defense"] = {}
    for name in DEFENDED:
        print(f"== defense: {name} ==", flush=True)
        corpus = load_corpus(INPUTS / name, device)
        n_classes = corpus["meta"]["n_classes"]

        _, _, baseline_val = train_model(
            corpus["train_x"], corpus["train_y"], corpus["val_x"], corpus["val_y"],
            n_classes, device, epochs=220, seed=SEEDS[0],
        )
        print(f"  unweighted baseline val {baseline_val:.4f}", flush=True)

        t0 = time.time()
        weights, logits, val_logits, val_acc, state = fit_ensemble(corpus, n_classes, device)
        w_np = weights.cpu().numpy()
        ess = float(w_np.sum() ** 2 / np.square(w_np).sum())
        print(f"  trust learner val {val_acc:.4f} in {time.time() - t0:.0f}s "
              f"(effective sample size {ess:.0f} of {w_np.size})", flush=True)

        out = RESULTS / "defense" / name
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "weights.npy", w_np.astype(np.float32))
        np.save(out / "pred.npy", logits.argmax(1).cpu().numpy().astype(np.int64))
        np.save(out / "val_pred.npy", val_logits.argmax(1).cpu().numpy().astype(np.int64))
        # saved straight off the GPU: the verifier audits the tensors' device
        torch.save({"state_dict": state, "logits": logits}, out / "model.pt")

        metrics["defense"][name] = {
            "defense_val_acc": val_acc,
            "baseline_val_acc": baseline_val,
            "effective_sample_size": ess,
        }


def run_ablation(device: str, metrics: dict) -> None:
    print(f"== ablation on {ABLATION_CORPUS} ==", flush=True)
    corpus = load_corpus(INPUTS / ABLATION_CORPUS, device)
    n_classes = corpus["meta"]["n_classes"]
    metrics["ablation"] = {}

    for tag, override in ABLATION.items():
        _, logits, val_logits, val_acc, _ = fit_ensemble(corpus, n_classes, device, **override)
        out = RESULTS / "ablation" / tag
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "pred.npy", logits.argmax(1).cpu().numpy().astype(np.int64))
        np.save(out / "val_pred.npy", val_logits.argmax(1).cpu().numpy().astype(np.int64))
        metrics["ablation"][tag] = val_acc
        print(f"  {tag:16s} val={val_acc:.4f}", flush=True)


def write_report(metrics: dict) -> None:
    attack, defense, ablation = metrics["attack"], metrics["defense"], metrics["ablation"]
    lines = [
        "# Poisoned corpus defense",
        "",
        "## Attack on probe_corpus",
        (f"A greedy meta-gradient label attack relabels {attack['rows_modified']} training "
         f"rows, {attack['rows_modified'] / attack['row_budget']:.0%} of the "
         f"{attack['row_budget']}-row budget, and drops the reference model's validation "
         f"accuracy from {attack['victim_val_acc_clean']:.3f} to "
         f"{attack['victim_val_acc_poisoned']:.3f}. Each candidate relabelling is scored by "
         "how far it moves the alignment between the row's output-layer gradient and the "
         "gradient of the trusted validation loss, which scores every row/label pair with "
         "two matmuls. Spending the budget over eight rounds, refitting in between, beats "
         "spending it all at once, because the early flips change which rows are worth "
         "flipping next. No feature perturbation is used: a dense step across standardised "
         "features makes a poisoned row an obvious outlier and measured strictly worse than "
         "leaving the features alone."),
        "",
        "## Defense",
        "| corpus | unweighted baseline | trust-weighted | effective sample size |",
        "| --- | --- | --- | --- |",
    ]
    for name, m in defense.items():
        lines.append(
            f"| {name} | {m['baseline_val_acc']:.3f} | {m['defense_val_acc']:.3f} | "
            f"{m['effective_sample_size']:.0f} |"
        )
    lines += [
        "",
        ("Validation accuracy, same split for every method. The learner scores each training "
         "row by three signals - the alignment of its gradient with the validation gradient, "
         "its agreement with the labels of its feature-space neighbours, and its training "
         "loss - then turns the combined score into a soft weight by thresholding at a "
         "quantile, retrains, and repeats. Weights are a sigmoid rather than a hard mask, so "
         "the ranking keeps the difference between a certainly-bad row and a mildly suspect "
         "one."),
        "",
        ("A fair second baseline is dropping the highest-loss rows outright and retraining. "
         "That works on alpha_feed, where the tampering leaves high-loss rows behind, and it "
         "is close to useless on gamma_feed, whose rows are individually well fitted and only "
         "wrong together - which is the case for keeping the gradient-alignment term."),
        "",
        "## Regularizer ablation (alpha_feed)",
        "| variant | val acc |",
        "| --- | --- |",
    ]
    for tag, val in ablation.items():
        lines.append(f"| {tag} | {val:.3f} |")
    lines += [
        "",
        "## What the weights say each feed was doing",
        ("The three corpora do not respond to the same treatment, and the weights show why. "
         "On alpha_feed the down-weighted rows are ones the model fits badly and whose "
         "neighbours disagree with them: labels were changed. On beta_feed the down-weighted "
         "rows sit in a region of feature space dominated by a different class while carrying "
         "their original labels, which is a features-were-moved signature rather than a "
         "labels-were-changed one. On gamma_feed the loss signal is nearly blind and almost "
         "all of the separation comes from gradient alignment, which is what a poison looks "
         "like when each row is individually plausible and only the correlation between them "
         "is a lie."),
    ]
    (RESULTS / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    # --allow-cpu exists for authoring only: it lets the artifacts be produced on a
    # laptop so the verifier can be debugged without burning GPU time. A CPU run
    # cannot pass, because the checkpoint provenance checks require CUDA tensors.
    # solve.sh never passes it, so the graded path has no CPU fallback.
    allow_cpu = "--allow-cpu" in sys.argv
    if not torch.cuda.is_available():
        if not allow_cpu:
            raise SystemExit("this task requires a CUDA GPU; there is no CPU fallback")
        print("WARNING: running on CPU for authoring; this output cannot pass the "
              "checkpoint provenance checks", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEEDS[0])
    label = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    print(f"device={device} ({label})", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    metrics: dict = {}
    t0 = time.time()
    run_attack(device, metrics)
    run_defense(device, metrics)
    run_ablation(device, metrics)

    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    write_report(metrics)
    print(f"total {time.time() - t0:.0f}s", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
