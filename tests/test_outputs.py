"""The verifier.

Every check reads an artifact the agent produced and compares it against ground truth
the container never had access to, or against a quantity this file recomputes itself.
No number the agent reported is taken on trust; `metrics.json` is checked *against* the
predictions rather than believed.

Two markers, and the distinction between them matters. `gpu` means the check cannot
run without CUDA at all - only the checkpoint provenance checks are in that class.
`heavy` means the check trains models and takes minutes rather than milliseconds; those
run fine on a CPU, just slowly. Marking a merely-slow check as `gpu` would quietly
delete it from every CPU run, which is how a gate stops being a gate.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch
import verifier_lib as V

T = V.thresholds()


def _threshold(key: str):
    value = T[key]
    if value is None:
        raise AssertionError(
            f"threshold {key!r} is null in thresholds.json. Gates are calibrated from "
            "repeated oracle runs; shipping an uncalibrated one is a bug, not a default."
        )
    return value


# --------------------------------------------------------------------- fixtures


@pytest.fixture(scope="session")
def public():
    return {name: V.load_public(name)
            for name in (*V.DEFENDED, V.ATTACK_CORPUS)}


@pytest.fixture(scope="session")
def hidden():
    return {name: V.load_hidden(name)
            for name in (*V.DEFENDED, V.ATTACK_CORPUS)}


@pytest.fixture(scope="session")
def predictions(public):
    """pred.npy per defended corpus, validated on the way in."""
    out = {}
    for name in V.DEFENDED:
        meta = public[name]["meta"]
        out[name] = V.read_predictions(
            V.RESULTS / "defense" / name / "pred.npy",
            meta["n_test"], meta["n_classes"],
        )
    return out


@pytest.fixture(scope="session")
def val_predictions(public):
    """Validation predictions, which is what makes the reported numbers checkable."""
    out = {}
    for name in V.DEFENDED:
        meta = public[name]["meta"]
        out[name] = V.read_predictions(
            V.RESULTS / "defense" / name / "val_pred.npy",
            meta["n_val"], meta["n_classes"],
        )
    return out


@pytest.fixture(scope="session")
def ablation_val_predictions(public):
    root = V.RESULTS / "ablation"
    assert root.is_dir(), f"{root} not produced"
    meta = public["alpha_feed"]["meta"]
    out = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        out[d.name] = V.read_predictions(d / "val_pred.npy", meta["n_val"], meta["n_classes"])
    return out


@pytest.fixture(scope="session")
def weights(public):
    return {
        name: V.read_weights(V.RESULTS / "defense" / name / "weights.npy",
                             public[name]["meta"]["n_train"])
        for name in V.DEFENDED
    }


@pytest.fixture(scope="session")
def baselines(public, hidden):
    """Unweighted reference accuracy per corpus, trained here rather than reported.

    This is the number the agent has to beat, so it is the last number that could be
    taken from the agent's own metrics.json.
    """
    out = {}
    for name in V.DEFENDED:
        c, h = public[name], hidden[name]
        out[name] = V.reference_accuracy(
            c["train_x"], c["train_y"], c["val_x"], c["val_y"],
            c["test_x"], h["test_y"],
        )
    return out


@pytest.fixture(scope="session")
def reported():
    path = V.RESULTS / "metrics.json"
    assert path.exists(), f"{path} not produced"
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise AssertionError(f"metrics.json is not valid JSON: {exc}") from exc


@pytest.fixture(scope="session")
def ablation_predictions(public):
    root = V.RESULTS / "ablation"
    assert root.is_dir(), f"{root} not produced"
    meta = public["alpha_feed"]["meta"]
    out = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        out[d.name] = V.read_predictions(d / "pred.npy", meta["n_test"], meta["n_classes"])
    return out


# ----------------------------------------------------------------- data integrity


def test_public_corpora_match_manifest():
    """The corpora in the image are the ones the held-out labels were generated with.

    Cheap, and it turns the one silent failure this task could have - public and
    hidden halves drifting apart - into a loud one. If it ever fires, every accuracy
    number in the run is meaningless.
    """
    import hashlib

    manifest = json.loads((V.TESTS_DIR.parent / "data" / "manifest.json").read_text())
    checked = 0
    for key, expected in manifest.items():
        if not key.startswith("corpora/"):
            continue
        path = V.CORPORA / Path(key).relative_to("corpora")
        assert path.exists(), f"{key} missing from the image"
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == expected, (
            f"{key} does not match the shipped manifest; the public corpora and the "
            "held-out labels are out of sync"
        )
        checked += 1
    assert checked >= 20, f"only {checked} public files checked against the manifest"


def test_held_out_labels_are_not_in_the_container():
    """The answers must not be reachable from where the agent ran."""
    for name in V.DEFENDED:
        corpus_dir = V.CORPORA / name
        assert not (corpus_dir / "test_y.npy").exists(), (
            f"{corpus_dir}/test_y.npy exists inside the image; the held-out labels leaked"
        )
        assert not (corpus_dir / "poison_idx.npy").exists(), (
            f"{corpus_dir}/poison_idx.npy exists inside the image; the answer key leaked"
        )


# ------------------------------------------------------------------------ attack


@pytest.fixture(scope="session")
def attack_artifacts(public):
    base = public[V.ATTACK_CORPUS]
    out_dir = V.RESULTS / "attack"
    x_path, y_path = out_dir / "poisoned_train_x.npy", out_dir / "poisoned_train_y.npy"
    assert x_path.exists(), f"{x_path} not produced"
    assert y_path.exists(), f"{y_path} not produced"
    new_x, new_y = np.load(x_path), np.load(y_path)
    diff = V.attack_diff(base["train_x"], base["train_y"], new_x, new_y)
    return {"new_x": new_x, "new_y": new_y, "diff": diff, "base": base}


def test_attack_respects_row_budget(attack_artifacts):
    base = attack_artifacts["base"]
    budget = round(_threshold("attack_row_budget_fraction") * base["meta"]["n_train"])
    modified = attack_artifacts["diff"]["rows_modified"]
    assert modified <= budget, (
        f"{modified} training rows were modified; the budget is {budget}"
    )
    assert modified > 0, "no rows were modified; there is no attack to score"


def test_attack_respects_perturbation_budget(attack_artifacts):
    """A row-count budget alone would let an agent rewrite eight percent of the corpus
    into anything at all. The per-feature cap is what makes it a perturbation."""
    limit = _threshold("attack_max_abs_perturbation")
    worst = attack_artifacts["diff"]["max_abs_perturbation"]
    assert worst <= limit + 1e-6, (
        f"largest feature change is {worst:.4f}, over the {limit} cap"
    )


def test_attack_labels_are_valid_classes(attack_artifacts):
    n_classes = attack_artifacts["base"]["meta"]["n_classes"]
    new_y = attack_artifacts["new_y"]
    assert np.issubdtype(new_y.dtype, np.integer), (
        f"poisoned labels must be integers, got {new_y.dtype}")
    assert new_y.min() >= 0 and new_y.max() < n_classes, (
        f"poisoned labels fall outside [0, {n_classes})")


@pytest.mark.heavy
def test_attack_costs_the_reference_model_accuracy(attack_artifacts, hidden):
    """Train the victim on the agent's poisoned corpus and measure what it lost.

    Random relabelling inside the same budget moves this a fraction of a point, so the
    bar separates an attack from noise rather than merely detecting that bytes changed.
    """
    base = attack_artifacts["base"]
    truth = hidden[V.ATTACK_CORPUS]["test_y"]

    clean = V.reference_accuracy(base["train_x"], base["train_y"], base["val_x"],
                                 base["val_y"], base["test_x"], truth)
    poisoned = V.reference_accuracy(attack_artifacts["new_x"], attack_artifacts["new_y"],
                                    base["val_x"], base["val_y"], base["test_x"], truth)
    drop = clean - poisoned
    minimum = _threshold("attack_min_accuracy_drop")
    assert drop >= minimum, (
        f"the attack cost the reference model {drop:.4f} accuracy "
        f"({clean:.4f} -> {poisoned:.4f}); at least {minimum} is required"
    )


# --------------------------------------------------------------- defense artifacts


def test_weights_are_well_formed(weights):
    """read_weights does the work; this pins the failure to a named check."""
    for name, w in weights.items():
        assert w.shape[0] > 0, f"{name}: empty weight vector"


def test_weights_retain_a_usable_training_set(weights, public):
    """Effective sample size, not a count of non-zeros.

    A sigmoid weight is never exactly zero, so counting non-zeros would pass by
    construction for any agent whose weights are a smooth function of a score. ESS
    measures how many rows the weighting is really training on and is unmoved by
    sprinkling epsilon over rows that have actually been discarded.
    """
    fraction = _threshold("min_effective_sample_size_fraction")
    for name, w in weights.items():
        n = public[name]["meta"]["n_train"]
        ess = V.effective_sample_size(w)
        assert ess >= fraction * n, (
            f"{name}: effective sample size {ess:.0f} of {n} rows is below "
            f"{fraction:.0%}; that is a discarded corpus, not a repaired one"
        )


def test_weights_retain_the_untampered_rows(weights, public, hidden):
    """Zeroing everything indiscriminately would satisfy the ranking test for free."""
    fraction = _threshold("min_clean_effective_sample_size_fraction")
    for name, w in weights.items():
        mask = V.poison_mask(public[name]["meta"]["n_train"], hidden[name]["poison_idx"])
        clean_w = w[~mask]
        ess = V.effective_sample_size(clean_w)
        assert ess >= fraction * clean_w.size, (
            f"{name}: only {ess:.0f} of {clean_w.size} untampered rows survive in "
            f"effective terms, below {fraction:.0%}"
        )


def test_weights_rank_tampered_rows_below_clean_ones(weights, public, hidden):
    """The central check: did the agent actually find the tampering?

    Every corpus has to clear it. Averaging would let one well-cleaned corpus carry two
    that were handed back untouched, and the instruction promises a per-corpus bar.
    """
    limit = _threshold("max_poison_rank_auc")
    measured = {}
    for name, w in weights.items():
        mask = V.poison_mask(public[name]["meta"]["n_train"], hidden[name]["poison_idx"])
        measured[name] = V.rank_auc(w, mask)

    over = {k: round(v, 4) for k, v in measured.items() if v > limit}
    assert not over, (
        f"tampered rows are not ranked below clean ones in {over} "
        f"(limit {limit}; uniform weights score 0.5). All corpora: "
        f"{ {k: round(v, 4) for k, v in measured.items()} }"
    )


def test_predictions_are_well_formed(predictions):
    for name, pred in predictions.items():
        assert pred.shape[0] > 0, f"{name}: empty prediction vector"


# ------------------------------------------------------------------ defense scores


def test_defense_reaches_required_accuracy(predictions, hidden):
    minimum = _threshold("defense_min_mean_accuracy")
    per_corpus = {
        name: float((pred == hidden[name]["test_y"]).mean())
        for name, pred in predictions.items()
    }
    mean_acc = float(np.mean(list(per_corpus.values())))
    assert mean_acc >= minimum, (
        f"mean held-out accuracy {mean_acc:.4f} is below {minimum}; "
        f"per corpus { {k: round(v, 4) for k, v in per_corpus.items()} }"
    )


@pytest.mark.heavy
def test_defense_beats_the_unweighted_baseline(predictions, hidden, baselines):
    """Accuracy alone could be cleared by a corpus that was never very damaged."""
    minimum = _threshold("defense_min_gain_over_baseline")
    accs = {name: float((pred == hidden[name]["test_y"]).mean())
            for name, pred in predictions.items()}
    mean_acc = float(np.mean(list(accs.values())))
    mean_base = float(np.mean(list(baselines.values())))
    gain = mean_acc - mean_base
    assert gain >= minimum, (
        f"the defense beats the unweighted baseline by only {gain:+.4f} "
        f"({mean_acc:.4f} vs {mean_base:.4f}); at least {minimum} is required"
    )


# ---------------------------------------------------------------------- checkpoint


@pytest.fixture(scope="session")
def checkpoints():
    return {name: V.load_checkpoint(V.RESULTS / "defense" / name / "model.pt")
            for name in V.DEFENDED}


def test_checkpoint_holds_a_real_trained_model(checkpoints):
    """Structural evidence only. The verifier never imports or instantiates the agent's
    model class - the task is meant to leave the choice of method open, and a gate that
    dictates the architecture would close it."""
    minimum = _threshold("min_checkpoint_parameters")
    for name, obj in checkpoints.items():
        params = V.checkpoint_parameters(obj["state_dict"])
        assert params, f"{name}: state_dict holds no floating point parameters"

        total = sum(t.numel() for t in params)
        assert total >= minimum, (
            f"{name}: state_dict holds {total} parameters, below {minimum}; "
            "that is not a trained classifier"
        )
        assert all(torch.isfinite(t).all() for t in params), (
            f"{name}: state_dict contains non-finite parameters")

        spread = max(float(t.float().std()) for t in params if t.numel() > 1)
        assert spread > 1e-8, (
            f"{name}: every parameter tensor is constant; the checkpoint was never trained")


def test_checkpoint_logits_match_submitted_predictions(checkpoints, predictions, public):
    """Ties the graded predictions to the shipped artifact."""
    for name, obj in checkpoints.items():
        meta = public[name]["meta"]
        logits = obj["logits"]
        assert torch.is_tensor(logits), f"{name}: logits are not a tensor"
        assert tuple(logits.shape) == (meta["n_test"], meta["n_classes"]), (
            f"{name}: logits have shape {tuple(logits.shape)}, expected "
            f"{(meta['n_test'], meta['n_classes'])}"
        )
        assert torch.isfinite(logits).all(), f"{name}: logits contain non-finite values"

        agreement = float(
            (logits.argmax(1).cpu().numpy() == predictions[name]).mean())
        assert agreement >= 0.999, (
            f"{name}: checkpoint logits reproduce only {agreement:.3f} of the submitted "
            "predictions; the artifact is not the model that was scored"
        )


def test_checkpoint_logits_are_not_fabricated(checkpoints, predictions):
    """Reject a logits tensor built backwards from an answer vector."""
    for name, obj in checkpoints.items():
        problem = V.logits_look_fabricated(obj["logits"], predictions[name])
        assert problem is None, f"{name}: {problem}"


def test_checkpoints_differ_between_corpora(checkpoints):
    """One model copied into three directories is not three trained models."""
    prints = {name: V.state_dict_fingerprint(obj["state_dict"])
              for name, obj in checkpoints.items()}
    assert len(set(prints.values())) == len(prints), (
        f"the same state_dict was submitted for more than one corpus: {prints}")


@pytest.mark.gpu
def test_checkpoint_was_saved_from_a_cuda_device(checkpoints):
    """Loaded without map_location, so tensors arrive on the device they were saved from.

    This proves the checkpoint lived on a GPU, not that gradient descent physically ran
    there - a CPU tensor moved to CUDA before saving would pass. See the README.
    """
    for name, obj in checkpoints.items():
        logits = obj["logits"]
        assert logits.is_cuda, (
            f"{name}: logits were saved from {logits.device}, expected a CUDA device")

        params = V.checkpoint_parameters(obj["state_dict"])
        off = [str(t.device) for t in params if not t.is_cuda]
        assert not off, (
            f"{name}: {len(off)} parameter tensors were saved from {off[0]}, "
            "expected a CUDA device")


@pytest.mark.gpu
def test_checkpoint_parameters_support_a_cuda_backward(checkpoints):
    for name, obj in checkpoints.items():
        params = V.checkpoint_parameters(obj["state_dict"])[:8]
        leaves = [t.detach().clone().requires_grad_(True) for t in params]
        torch.stack([p.float().pow(2).sum() for p in leaves]).sum().backward()
        for p in leaves:
            assert p.grad is not None and p.grad.is_cuda and torch.isfinite(p.grad).all(), (
                f"{name}: checkpoint parameters do not support a finite CUDA backward pass")


# ------------------------------------------------------------------------ ablation


def test_ablation_has_enough_variants(ablation_predictions):
    minimum = _threshold("min_ablation_variants")
    assert len(ablation_predictions) >= minimum, (
        f"found {len(ablation_predictions)} usable ablation variants, need >= {minimum}")
    assert "full" in ablation_predictions, (
        f"no variant named 'full' among {sorted(ablation_predictions)}")


def test_ablation_variants_were_actually_retrained(ablation_predictions):
    """Copying one result into four directories is the cheapest way to fake this."""
    minimum = _threshold("min_distinct_ablation_predictions")
    distinct = len({p.tobytes() for p in ablation_predictions.values()})
    assert distinct >= minimum, (
        f"only {distinct} distinct prediction vectors across "
        f"{len(ablation_predictions)} variants; the variants were not retrained"
    )


# ------------------------------------------------------------------ reported numbers


def test_metrics_json_has_the_required_shape(reported, ablation_predictions):
    assert isinstance(reported, dict), "metrics.json top level must be an object"

    attack = reported.get("attack")
    assert isinstance(attack, dict), "'attack' must be an object"
    for key in ("victim_val_acc_clean", "victim_val_acc_poisoned", "rows_modified"):
        assert key in attack, f"attack.{key} is missing"

    defense = reported.get("defense")
    assert isinstance(defense, dict), "'defense' must be an object"
    for name in V.DEFENDED:
        assert name in defense, f"defense.{name} is missing"
        for key in ("defense_val_acc", "baseline_val_acc"):
            assert key in defense[name], f"defense.{name}.{key} is missing"

    ablation = reported.get("ablation")
    assert isinstance(ablation, dict), "'ablation' must be an object"
    assert set(ablation) == set(ablation_predictions), (
        f"ablation keys {sorted(ablation)} do not match the variant directories "
        f"{sorted(ablation_predictions)}"
    )


def test_reported_accuracies_match_the_submitted_artifacts(
    reported, public, val_predictions, ablation_val_predictions
):
    """Recompute every reported accuracy from the artifact that claims to describe it.

    Validation labels are public, so this needs nothing held out - which is the point.
    An agent can write any number into metrics.json, but it has to survive being
    rebuilt from the predictions they shipped alongside it. Reporting the number a
    better run would have produced, and shipping the predictions of the run that
    actually happened, fails here.
    """
    tolerance = _threshold("max_reported_accuracy_delta")
    deltas: dict[str, float] = {}

    for name in V.DEFENDED:
        claimed = reported["defense"][name]["defense_val_acc"]
        assert isinstance(claimed, (int, float)), (
            f"defense.{name}.defense_val_acc is not a number")
        seen = float((val_predictions[name] == public[name]["val_y"]).mean())
        deltas[f"defense.{name}"] = abs(float(claimed) - seen)

    alpha_val_y = public["alpha_feed"]["val_y"]
    for tag, pred in ablation_val_predictions.items():
        claimed = reported["ablation"].get(tag)
        if isinstance(claimed, dict):
            claimed = claimed.get("val_accuracy")
        assert isinstance(claimed, (int, float)), (
            f"ablation.{tag} is not a number and has no val_accuracy field")
        seen = float((pred == alpha_val_y).mean())
        deltas[f"ablation.{tag}"] = abs(float(claimed) - seen)

    worst = max(deltas, key=deltas.get)
    assert deltas[worst] <= tolerance, (
        f"{worst} is off by {deltas[worst]:.4f} from the value recomputed from the "
        f"submitted predictions (tolerance {tolerance}); all deltas "
        f"{ {k: round(v, 4) for k, v in deltas.items()} }"
    )


def test_reported_row_count_matches_the_attack_artifact(reported, public):
    """The one attack number the verifier can rebuild exactly rather than approximately."""
    base = public[V.ATTACK_CORPUS]
    new_x = np.load(V.RESULTS / "attack" / "poisoned_train_x.npy")
    new_y = np.load(V.RESULTS / "attack" / "poisoned_train_y.npy")
    measured = V.attack_diff(base["train_x"], base["train_y"], new_x, new_y)["rows_modified"]
    claimed = int(reported["attack"]["rows_modified"])
    assert claimed == measured, (
        f"metrics.json reports {claimed} modified rows; the artifact has {measured}")


def test_report_is_a_real_write_up(reported):
    path = V.RESULTS / "report.md"
    assert path.exists(), f"{path} not produced"
    text = path.read_text().strip()
    lowered = text.lower()

    minimum = _threshold("min_report_chars")
    assert len(text) >= minimum, (
        f"report.md is {len(text)} characters, expected at least {minimum}")

    missing = [name for name in V.DEFENDED if name not in lowered]
    assert not missing, f"report.md never mentions {missing}"

    for topic, wording in (
        ("the attack", ("attack", "poison")),
        ("a baseline comparison", ("baseline", "unweighted")),
        ("the ablation", ("ablation", "regulariz")),
        ("what the weights found", ("weight", "down-weight", "trust", "tamper")),
    ):
        assert any(w in lowered for w in wording), f"report.md does not discuss {topic}"

    figures = re.findall(r"\d+\.\d+", text)
    minimum_figures = _threshold("min_report_numeric_figures")
    assert len(figures) >= minimum_figures, (
        f"report.md quotes only {len(figures)} numeric results, expected at least "
        f"{minimum_figures}; a write-up of real results cites them"
    )
