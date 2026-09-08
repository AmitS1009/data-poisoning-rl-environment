"""Deterministic generator for the vibration-telemetry corpora.

The corpora are generated once by this script and the resulting arrays are committed
alongside it. The generator itself never enters the container image: if it did, an
agent could simply re-run it and read the held-out test labels straight out of the
seed. What ships in the image is the public half; the held-out half travels with the
verifier and arrives only at grading time.

Committing the arrays rather than generating them at build time also removes a real
correctness risk. Poison selection ranks rows by a float64 surrogate fit, and BLAS
reduction order is not guaranteed identical across CPU microarchitectures, so a
build-time regeneration could in principle disagree with the grader about which rows
were poisoned. Generating once and shipping the bytes makes that impossible. The
`--verify` mode re-derives everything from the seed and diffs it against the shipped
manifest, which is what CI runs on amd64 to check the claim rather than assume it.

    python generate_corpora.py --out ../fixtures            # generate + manifest
    python generate_corpora.py --verify ../fixtures         # regenerate and diff
    python generate_corpora.py --out /tmp/tiny --tiny       # CPU smoke fixtures

The physical story: each row is a 72-bin power spectrum from an accelerometer on a
rotating machine. A class is a fault mode, expressed as a set of resonance peaks at
characteristic frequencies plus their harmonics, buried in 1/f noise and observed
through a fixed sensor mixing matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- config

MASTER_SEED = 20260908
N_BINS = 72
N_CLASSES = 5

CLASS_NAMES = ("healthy", "bearing_wear", "imbalance", "misalignment", "looseness")

# One corpus spec per directory. `poison` names the mechanism used to tamper with it;
# `probe_corpus` ships clean because the agent is the one who attacks it.
CORPORA = {
    "probe_corpus": {"seed_offset": 11, "n_train": 3000, "poison": None,
                     "rate": 0.0, "strength": 0.0},
    "alpha_feed": {"seed_offset": 23, "n_train": 3200, "poison": "label_rotation",
                   "rate": 0.16, "strength": 1.0},
    "beta_feed": {"seed_offset": 37, "n_train": 3200, "poison": "clean_label_drift",
                  "rate": 0.30, "strength": 0.92},
    "gamma_feed": {"seed_offset": 53, "n_train": 3000, "poison": "trigger_collision",
                   "rate": 0.18, "strength": 1.5},
}

N_VAL = 800
N_TEST = 1500

# Difficulty knobs. These are set so a well-trained model on clean data lands in the
# mid-80s rather than at ceiling: a task that saturates leaves a defense nothing to
# recover, and every accuracy gate would then be measuring noise.
FREQ_LO, FREQ_HI = 0.10, 0.24   # class fundamentals, as a fraction of the band
NOISE_SCALE = 2.35              # 1/f noise amplitude relative to the resonance peaks
SHELF_LO, SHELF_HI = 0.04, 0.09 # per-class broadband offset, kept small on purpose
JITTER = 0.020                  # per-class frequency jitter

TINY = {  # the CPU smoke path: same code, same invariants, ~40x smaller
    "n_bins": 24,
    "n_classes": 3,
    "n_train": 300,
    "n_val": 120,
    "n_test": 200,
}


# ---------------------------------------------------------------------- spectra


def _class_templates(rng: np.random.Generator, n_bins: int, n_classes: int) -> np.ndarray:
    """One resonance template per class: a few peaks plus their harmonics.

    Peaks are placed on a jittered grid so no class sits exactly on another's
    harmonic, which would make two fault modes genuinely indistinguishable rather
    than merely hard.
    """
    templates = np.zeros((n_classes, n_bins), dtype=np.float64)
    grid = np.linspace(FREQ_LO, FREQ_HI, n_classes)
    axis = np.arange(n_bins, dtype=np.float64)

    for k in range(n_classes):
        fundamental = (grid[k] + rng.uniform(-JITTER, JITTER)) * n_bins
        n_harmonics = int(rng.integers(2, 4))
        for h in range(1, n_harmonics + 1):
            centre = fundamental * h
            if centre >= n_bins - 1:
                break
            width = 1.1 + 0.45 * h
            amplitude = rng.uniform(0.7, 1.25) / h
            templates[k] += amplitude * np.exp(-0.5 * ((axis - centre) / width) ** 2)
        # a broadband shelf that differs per class, so the task is not purely
        # "find the tallest peak"
        shelf = rng.uniform(SHELF_LO, SHELF_HI)
        templates[k] += shelf * np.exp(-axis / rng.uniform(18.0, 45.0))

    return templates


def _pink_noise(rng: np.random.Generator, n: int, n_bins: int) -> np.ndarray:
    """1/f-shaped noise, the dominant nuisance in real accelerometer spectra."""
    white = rng.standard_normal((n, n_bins))
    scale = 1.0 / np.sqrt(np.arange(1, n_bins + 1, dtype=np.float64))
    return white * scale


def _sensor_mixing(rng: np.random.Generator, n_bins: int) -> np.ndarray:
    """A fixed, well-conditioned mixing matrix standing in for sensor placement.

    Without it the informative bins are axis-aligned and a per-feature threshold
    solves the problem, which would make the defense trivially easy for the wrong
    reason.
    """
    base = rng.standard_normal((n_bins, n_bins)) / np.sqrt(n_bins)
    mixing = 0.72 * np.eye(n_bins) + 0.28 * base
    return mixing


def _sample_spectra(
    rng: np.random.Generator,
    templates: np.ndarray,
    mixing: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Draw observed spectra for the given labels."""
    n = labels.shape[0]
    n_bins = templates.shape[1]

    amplitude = rng.uniform(0.75, 1.3, size=(n, 1))
    base = templates[labels] * amplitude

    # per-sample spectral tilt: machines run at slightly different speeds
    tilt = rng.uniform(-0.35, 0.35, size=(n, 1))
    axis = np.linspace(0.0, 1.0, n_bins)[None, :]
    base = base * (1.0 + tilt * axis)

    # per-sample broadband offset: without it the per-class shelf is a clean cue that
    # survives any amount of peak noise, and the classes separate for the wrong reason
    drift = rng.uniform(0.0, 0.55, size=(n, 1))
    base = base + drift * np.exp(-np.arange(n_bins, dtype=np.float64) / 30.0)[None, :]

    observed = base + NOISE_SCALE * _pink_noise(rng, n, n_bins)
    observed = observed @ mixing
    return observed


def _standardise(train: np.ndarray, *others: np.ndarray):
    """Standardise every split with the training split's own statistics."""
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-9, 1.0, std)
    out = [(train - mean) / std]
    out.extend((other - mean) / std for other in others)
    return out


# --------------------------------------------------------------- numpy surrogate


def _fit_surrogate(x: np.ndarray, y: np.ndarray, n_classes: int, steps: int = 320,
                   lr: float = 0.35, l2: float = 1e-3) -> np.ndarray:
    """Multinomial logistic regression by plain full-batch gradient descent.

    Used only to *design* the poison: the attacks below need a notion of "near the
    decision boundary" and "in the direction of another class". Full-batch float64
    GD with a fixed step count is exactly reproducible, which a solver with an
    iteration-count stopping rule would not be.
    """
    n, d = x.shape
    xb = np.concatenate([x, np.ones((n, 1))], axis=1)
    weights = np.zeros((d + 1, n_classes), dtype=np.float64)
    onehot = np.zeros((n, n_classes), dtype=np.float64)
    onehot[np.arange(n), y] = 1.0

    for _ in range(steps):
        logits = xb @ weights
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        grad = xb.T @ (probs - onehot) / n + l2 * weights
        weights -= lr * grad

    return weights


def _surrogate_probs(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
    xb = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    logits = xb @ weights
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    return probs / probs.sum(axis=1, keepdims=True)


# ------------------------------------------------------------------- poisoning
#
# Three mechanisms, deliberately unalike. A defense tuned to any single one of them
# fails the other two, which is the point: the cheap heuristic for each is different,
# so only a method that actually estimates per-row influence clears all three.


def _poison_label_rotation(rng, x, y, n_poison, n_classes, strength):
    """Apply a fixed cyclic relabelling to the rows the surrogate is surest about.

    Systematic beats random: a coherent class -> class+1 rotation teaches the model a
    consistent lie, where uniformly random flips mostly average out. Confident rows
    are chosen because they carry the largest gradient, so the corrupted signal is
    the one that moves the decision boundary furthest.

    This is the one a loss-based filter can find, and it is in the task on purpose as
    the corpus where the obvious heuristic works.
    """
    weights = _fit_surrogate(x, y, n_classes)
    probs = _surrogate_probs(weights, x)
    confidence = probs[np.arange(len(y)), y]

    eligible = np.flatnonzero(probs.argmax(1) == y)
    order = np.argsort(-confidence[eligible], kind="stable")
    idx = eligible[order][:n_poison]

    shift = max(1, int(round(strength)))
    y_out = y.copy()
    y_out[idx] = (y[idx] + shift) % n_classes
    return x.copy(), y_out, np.sort(idx)


def _poison_clean_label_drift(rng, x, y, n_poison, n_classes, strength):
    """Move rows into another class's region, leaving their labels alone.

    Interpolating toward the target centroid rather than along a logistic direction
    matters: a large step along a weight-difference vector just lands the row in
    empty space, where the model writes it off as noise and test accuracy barely
    moves. Landing the row *inside* the target's cloud is what forces the model to
    carve the target's region out for the source label.

    Every label is the one the row was born with, so nothing here is a mislabelling
    a per-row loss test could flag. One source class and one target class, so the
    damage concentrates on a single boundary.
    """
    centroids = np.stack([x[y == k].mean(axis=0) for k in range(n_classes)])
    spreads = np.stack([x[y == k].std(axis=0) for k in range(n_classes)])

    target = int(rng.integers(0, n_classes))

    # victims are drawn from every other class, not from one source class. Taking them
    # all from a single class would strip that class out of the training set, and then
    # deleting the poisoned rows - the ceiling any defense is measured against - would
    # itself score worse than leaving them in, which makes the corpus untestable.
    pool = np.flatnonzero(y != target)
    victims = np.sort(rng.choice(pool, size=min(n_poison, pool.size), replace=False))

    t = float(np.clip(strength, 0.0, 1.0))
    landing = centroids[target][None, :] + spreads[target][None, :] * rng.standard_normal(
        (victims.size, x.shape[1])
    )
    x_out = x.copy()
    x_out[victims] = (1.0 - t) * x[victims] + t * landing
    return x_out, y.copy(), victims


def _poison_trigger_collision(rng, x, y, n_poison, n_classes, strength):
    """Stamp a sparse fixed direction onto rows and relabel them to one target class.

    The model learns that the direction is evidence for the target class. Test rows
    have their own random projection onto that direction, so the spurious feature
    fires on clean data too and the error shows up everywhere rather than only on
    triggered rows. Each poisoned row looks individually ordinary; the damage is in
    the correlation, which is what a per-row test cannot see.
    """
    d = x.shape[1]
    support = rng.choice(d, size=max(3, d // 8), replace=False)
    trigger = np.zeros(d, dtype=np.float64)
    trigger[support] = rng.standard_normal(support.size)
    trigger /= np.linalg.norm(trigger)

    target = int(rng.integers(0, n_classes))
    pool = np.flatnonzero(y != target)
    idx = rng.choice(pool, size=min(n_poison, pool.size), replace=False)

    row_scale = np.linalg.norm(x[idx], axis=1, keepdims=True) / np.sqrt(d)
    x_out = x.copy()
    y_out = y.copy()
    x_out[idx] += strength * trigger[None, :] * row_scale
    y_out[idx] = target
    return x_out, y_out, np.sort(idx)


POISONS = {
    "label_rotation": _poison_label_rotation,
    "clean_label_drift": _poison_clean_label_drift,
    "trigger_collision": _poison_trigger_collision,
}


def build_corpus(name: str, spec: dict, tiny: bool = False) -> dict:
    """Generate one corpus and, if it has one, apply its poison."""
    cfg = TINY if tiny else {}
    n_bins = cfg.get("n_bins", N_BINS)
    n_classes = cfg.get("n_classes", N_CLASSES)
    n_train = cfg.get("n_train", spec["n_train"])
    n_val = cfg.get("n_val", N_VAL)
    n_test = cfg.get("n_test", N_TEST)

    rng = np.random.default_rng(MASTER_SEED + spec["seed_offset"] + (7919 if tiny else 0))

    templates = _class_templates(rng, n_bins, n_classes)
    mixing = _sensor_mixing(rng, n_bins)

    total = n_train + n_val + n_test
    labels = rng.integers(0, n_classes, size=total)
    spectra = _sample_spectra(rng, templates, mixing, labels)

    train_x, val_x, test_x = np.split(spectra, [n_train, n_train + n_val])
    train_y, val_y, test_y = np.split(labels, [n_train, n_train + n_val])
    train_x, val_x, test_x = _standardise(train_x, val_x, test_x)

    poison_idx = np.zeros(0, dtype=np.int64)
    if spec["poison"] is not None:
        n_poison = int(round(spec["rate"] * n_train))
        train_x, train_y, poison_idx = POISONS[spec["poison"]](
            rng, train_x, train_y, n_poison, n_classes, spec["strength"]
        )

    return {
        "name": name,
        "n_classes": n_classes,
        "n_bins": n_bins,
        "train_x": train_x.astype(np.float32),
        "train_y": train_y.astype(np.int64),
        "val_x": val_x.astype(np.float32),
        "val_y": val_y.astype(np.int64),
        "test_x": test_x.astype(np.float32),
        "test_y": test_y.astype(np.int64),
        "poison_idx": poison_idx.astype(np.int64),
    }


def _write_array(path: Path, array: np.ndarray, key: str, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    manifest[key] = hashlib.sha256(path.read_bytes()).hexdigest()


def write_corpus(corpus: dict, public_root: Path, hidden_root: Path, manifest: dict) -> None:
    """Public half into the image tree, held-out half into the verifier tree.

    The split is the whole security model. `test_y` and `poison_idx` are the two
    things the agent must never see, and they are the two things that go to
    hidden_root, which the Dockerfile does not copy. Manifest keys are logical paths
    so the hashes stay meaningful wherever the two trees physically live.
    """
    name = corpus["name"]
    public = public_root / name
    hidden = hidden_root / name

    for fname, key in (
        ("train_x.npy", "train_x"), ("train_y.npy", "train_y"),
        ("val_x.npy", "val_x"), ("val_y.npy", "val_y"),
        ("test_x.npy", "test_x"),
    ):
        _write_array(public / fname, corpus[key], f"corpora/{name}/{fname}", manifest)

    for fname, key in (("test_y.npy", "test_y"), ("poison_idx.npy", "poison_idx")):
        _write_array(hidden / fname, corpus[key], f"hidden/{name}/{fname}", manifest)

    meta = {
        "name": name,
        "n_train": int(corpus["train_x"].shape[0]),
        "n_val": int(corpus["val_x"].shape[0]),
        "n_test": int(corpus["test_x"].shape[0]),
        "n_bins": int(corpus["n_bins"]),
        "n_classes": int(corpus["n_classes"]),
        "class_names": list(CLASS_NAMES[: corpus["n_classes"]]),
    }
    meta_path = public / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    manifest[f"corpora/{name}/meta.json"] = hashlib.sha256(
        meta_path.read_bytes()
    ).hexdigest()


def generate_all(public_root: Path, hidden_root: Path, tiny: bool = False) -> dict:
    public_root.mkdir(parents=True, exist_ok=True)
    hidden_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}
    for name, spec in CORPORA.items():
        corpus = build_corpus(name, spec, tiny=tiny)
        write_corpus(corpus, public_root, hidden_root, manifest)
        print(
            f"{name:14s} train={corpus['train_x'].shape} "
            f"classes={corpus['n_classes']} poisoned={corpus['poison_idx'].size}",
            flush=True,
        )
    return manifest


REPO = Path(__file__).resolve().parent.parent
DEFAULT_PUBLIC = REPO / "environment" / "task_inputs" / "corpora"
DEFAULT_HIDDEN = REPO / "tests" / "hidden"
DEFAULT_MANIFEST = REPO / "data" / "manifest.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--public-out", type=Path, default=DEFAULT_PUBLIC)
    ap.add_argument("--hidden-out", type=Path, default=DEFAULT_HIDDEN)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--verify", action="store_true",
                    help="regenerate into a temp tree and diff against the manifest")
    ap.add_argument("--tiny", action="store_true", help="small corpora for the CPU smoke path")
    args = ap.parse_args()

    if not args.verify:
        manifest = generate_all(args.public_out, args.hidden_out, tiny=args.tiny)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(f"wrote {len(manifest)} files; manifest at {args.manifest}")
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        fresh = generate_all(tmp_root / "corpora", tmp_root / "hidden", tiny=args.tiny)
    shipped = json.loads(args.manifest.read_text())

    extra = sorted(set(fresh) - set(shipped))
    missing = sorted(set(shipped) - set(fresh))
    differing = sorted(k for k in set(fresh) & set(shipped) if fresh[k] != shipped[k])

    for label, items in (("not in manifest", extra),
                         ("not regenerated", missing),
                         ("hash mismatch", differing)):
        for item in items:
            print(f"{label}: {item}")

    if extra or missing or differing:
        raise SystemExit(
            f"regeneration does not match the shipped manifest "
            f"({len(differing)} differing, {len(extra)} extra, {len(missing)} missing)"
        )
    print(f"all {len(fresh)} generated files match the shipped manifest")


if __name__ == "__main__":
    main()
