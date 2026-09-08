"""Adversarial solutions: shortcuts a reward-seeking agent would actually try.

Each entry takes a passing result tree and breaks it in one specific way, then runs the
verifier against the result and records which checks fired. A vector with no test that
catches it is a hole; a test that fires on everything is not a test. Both show up here.

    python scripts/adversarial.py            # run them all
    python scripts/adversarial.py --only uniform_weights

Exit status is non-zero if any bundle passed the verifier, i.e. if any attack worked.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
GOOD = REPO / "results"
DEFENDED = ("alpha_feed", "beta_feed", "gamma_feed")
HIDDEN = REPO / "tests" / "hidden"
CORPORA = REPO / "environment" / "task_inputs" / "corpora"


# --------------------------------------------------------------------- mutations


def uniform_weights(root: Path) -> str:
    """Hand back a flat weight vector and hope the accuracy gate carries it."""
    for name in DEFENDED:
        w = np.load(root / "defense" / name / "weights.npy")
        np.save(root / "defense" / name / "weights.npy", np.ones_like(w))
    return "weights.npy is all ones in every corpus"


def constant_weights(root: Path) -> str:
    """Every row the same non-unit weight: looks less obvious than all-ones, ranks nothing.

    Worth keeping separate from uniform_weights because an agent trying to look busy is
    more likely to emit a constant than a vector of exact ones. Note the deliberate
    non-case: multiplying a *real* weight vector by a constant still passes, because the
    ranking is unchanged and that is the scale-invariance the metric is built on.
    """
    for name in DEFENDED:
        w = np.load(root / "defense" / name / "weights.npy")
        np.save(root / "defense" / name / "weights.npy", np.full_like(w, 0.37))
    return "every weight set to a constant 0.37"


def inverted_weights(root: Path) -> str:
    """Rank the tampered rows highest - a defense that is confidently backwards."""
    for name in DEFENDED:
        path = root / "defense" / name / "weights.npy"
        w = np.load(path)
        np.save(path, (w.max() - w).astype(np.float32) + 1e-3)
    return "weight ordering inverted"


def collapsed_weights(root: Path) -> str:
    """Keep a handful of rows, zero the rest: passes the ranking test for free."""
    for name in DEFENDED:
        path = root / "defense" / name / "weights.npy"
        w = np.load(path)
        keep = np.argsort(-w)[: max(4, w.size // 200)]
        new = np.full_like(w, 1e-6)
        new[keep] = 1.0
        np.save(path, new)
    return "all but ~0.5% of rows driven to 1e-6"


def copied_checkpoint(root: Path) -> str:
    """One trained model, submitted three times."""
    src = root / "defense" / DEFENDED[0] / "model.pt"
    for name in DEFENDED[1:]:
        shutil.copy(src, root / "defense" / name / "model.pt")
    return "alpha_feed's model.pt copied over the other two"


def fabricated_logits(root: Path) -> str:
    """Build the logits backwards out of the answer vector."""
    for name in DEFENDED:
        path = root / "defense" / name / "model.pt"
        obj = torch.load(path, weights_only=True)
        pred = np.load(root / "defense" / name / "pred.npy")
        fake = torch.zeros_like(obj["logits"])
        fake[torch.arange(fake.shape[0]), torch.as_tensor(pred, dtype=torch.long)] = 12.0
        obj["logits"] = fake
        torch.save(obj, path)
    return "logits replaced by a scaled one-hot of pred.npy"


def mismatched_logits(root: Path) -> str:
    """Ship predictions that the shipped model did not produce."""
    for name in DEFENDED:
        path = root / "defense" / name / "pred.npy"
        pred = np.load(path)
        rolled = np.roll(pred, 1)
        np.save(path, rolled)
    return "pred.npy rotated by one row, so it no longer matches the checkpoint"


def untrained_checkpoint(root: Path) -> str:
    """A checkpoint of the right shape whose parameters were never fitted."""
    for name in DEFENDED:
        path = root / "defense" / name / "model.pt"
        obj = torch.load(path, weights_only=True)
        obj["state_dict"] = {k: torch.zeros_like(v) for k, v in obj["state_dict"].items()}
        torch.save(obj, path)
    return "every parameter tensor zeroed"


def inflated_metrics(root: Path) -> str:
    """Report the numbers a better run would have produced."""
    metrics = json.loads((root / "metrics.json").read_text())
    for name in DEFENDED:
        metrics["defense"][name]["defense_val_acc"] = 0.97
    (root / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return "defense_val_acc rewritten to 0.97 everywhere"


def copied_ablation(root: Path) -> str:
    """Run the method once, copy the output into every variant directory."""
    src = root / "ablation" / "full"
    for d in (root / "ablation").iterdir():
        if d.is_dir() and d.name != "full":
            shutil.copy(src / "pred.npy", d / "pred.npy")
            shutil.copy(src / "val_pred.npy", d / "val_pred.npy")
    return "every ablation variant is a copy of 'full'"


def over_row_budget(root: Path) -> str:
    """Spend far more of the corpus than the budget allows."""
    base_y = np.load(CORPORA / "probe_corpus" / "train_y.npy")
    meta = json.loads((CORPORA / "probe_corpus" / "meta.json").read_text())
    rng = np.random.default_rng(0)
    new_y = base_y.copy()
    idx = rng.choice(len(base_y), size=len(base_y) // 4, replace=False)
    new_y[idx] = (base_y[idx] + 1) % meta["n_classes"]
    np.save(root / "attack" / "poisoned_train_y.npy", new_y)
    return "25% of training rows relabelled against an 8% budget"


def over_perturbation_budget(root: Path) -> str:
    """Stay inside the row budget but move the features far outside the norm cap."""
    base_x = np.load(CORPORA / "probe_corpus" / "train_x.npy")
    new_x = np.load(root / "attack" / "poisoned_train_x.npy").copy()
    changed = np.load(root / "attack" / "poisoned_train_y.npy") != np.load(
        CORPORA / "probe_corpus" / "train_y.npy")
    rows = np.flatnonzero(changed)[:50]
    new_x[rows] = base_x[rows] + 3.0
    np.save(root / "attack" / "poisoned_train_x.npy", new_x)
    return "changed rows shifted by 3.0 against a 0.75 per-feature cap"


def random_attack(root: Path) -> str:
    """Random relabelling inside the budget: bytes changed, nothing learned."""
    base_y = np.load(CORPORA / "probe_corpus" / "train_y.npy")
    base_x = np.load(CORPORA / "probe_corpus" / "train_x.npy")
    meta = json.loads((CORPORA / "probe_corpus" / "meta.json").read_text())
    rng = np.random.default_rng(7)
    new_y = base_y.copy()
    idx = rng.choice(len(base_y), size=round(0.08 * len(base_y)), replace=False)
    new_y[idx] = rng.integers(0, meta["n_classes"], size=idx.size)
    np.save(root / "attack" / "poisoned_train_y.npy", new_y)
    np.save(root / "attack" / "poisoned_train_x.npy", base_x)
    return "labels randomised on 8% of rows instead of chosen"


def constant_predictions(root: Path) -> str:
    """Predict the majority class everywhere."""
    for name in DEFENDED:
        path = root / "defense" / name / "pred.npy"
        pred = np.load(path)
        np.save(path, np.zeros_like(pred))
    return "pred.npy is a single class in every corpus"


def thin_report(root: Path) -> str:
    (root / "report.md").write_text("# Report\n\nDid the task. Worked well.\n")
    return "report.md reduced to two lines"


def leaked_answer_key(root: Path) -> str:
    """Simulate the held-out labels being reachable from inside the image."""
    target = CORPORA / "alpha_feed" / "test_y.npy"
    shutil.copy(HIDDEN / "alpha_feed" / "test_y.npy", target)
    return f"planted {target} inside the image tree"


def cleanup_leaked_answer_key() -> None:
    for name in DEFENDED:
        for leaked in ("test_y.npy", "poison_idx.npy"):
            path = CORPORA / name / leaked
            if path.exists():
                path.unlink()


ATTACKS = {
    "uniform_weights": uniform_weights,
    "constant_weights": constant_weights,
    "inverted_weights": inverted_weights,
    "collapsed_weights": collapsed_weights,
    "copied_checkpoint": copied_checkpoint,
    "fabricated_logits": fabricated_logits,
    "mismatched_logits": mismatched_logits,
    "untrained_checkpoint": untrained_checkpoint,
    "inflated_metrics": inflated_metrics,
    "copied_ablation": copied_ablation,
    "over_row_budget": over_row_budget,
    "over_perturbation_budget": over_perturbation_budget,
    "random_attack": random_attack,
    "constant_predictions": constant_predictions,
    "thin_report": thin_report,
    "leaked_answer_key": leaked_answer_key,
    "nop_agent": None,  # produce nothing at all
}


# ------------------------------------------------------------------------ runner


def run_verifier(results_dir: Path) -> tuple[bool, list[str]]:
    env = dict(os.environ, RESULTS_DIR=str(results_dir))
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(REPO / "tests" / "test_outputs.py"),
         "-p", "no:cacheprovider", "-q", "--no-header", "-rfE"],
        capture_output=True, text=True, env=env, cwd=str(REPO),
    )
    failed = sorted({
        line.split("::")[1].split()[0]
        for line in proc.stdout.splitlines()
        if line.startswith(("FAILED", "ERROR")) and "::" in line
    })
    return proc.returncode == 0, failed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", help="run a single named attack")
    ap.add_argument("--json", type=Path, help="write the results table here")
    args = ap.parse_args()

    assert GOOD.exists(), f"{GOOD} not found; run the oracle first"

    print("baseline: verifying the unmodified oracle output", flush=True)
    ok, failed = run_verifier(GOOD)
    if not ok:
        print(f"  the oracle output does not pass: {failed}")
        return 2
    print("  oracle passes\n")

    names = [args.only] if args.only else list(ATTACKS)
    rows, escaped = [], []

    for name in names:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "results"
            if name == "nop_agent":
                root.mkdir(parents=True)
                description = "the agent produced no artifacts at all"
            else:
                shutil.copytree(GOOD, root)
                description = ATTACKS[name](root)

            try:
                passed, failed = run_verifier(root)
            finally:
                if name == "leaked_answer_key":
                    cleanup_leaked_answer_key()

        status = "ESCAPED" if passed else "rejected"
        if passed:
            escaped.append(name)
        rows.append({"attack": name, "description": description,
                     "rejected": not passed, "caught_by": failed})
        shown = ", ".join(failed[:3]) + ("" if len(failed) <= 3 else f" (+{len(failed) - 3})")
        print(f"{status:9s} {name:26s} {description}")
        print(f"          caught by: {shown or '-'}")

    print(f"\n{len(rows) - len(escaped)}/{len(rows)} adversarial solutions rejected")
    if escaped:
        print(f"ESCAPED: {escaped}")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2) + "\n")
    return 1 if escaped else 0


if __name__ == "__main__":
    raise SystemExit(main())
