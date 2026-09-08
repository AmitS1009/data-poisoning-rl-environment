"""Run the oracle N times on a Modal T4 and report the spread of the graded metrics.

Thresholds are set from this, not from a single run. A gate placed just under one
observation is a gate a correct oracle fails intermittently, which is the worst failure
mode an evaluation environment has: it teaches whoever runs it that the environment is
unreliable, and there is no way to tell a flaky gate from a real regression.

Results are written into a modal.Dict rather than returned through the client, because
long silent calls lose their log stream from some networks and a dropped stream should
not lose a paid GPU run.

    modal run scripts/modal_calibrate.py --trials 10     # launch and wait
    python scripts/modal_calibrate.py --collect          # read whatever has landed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent

# Mirrors environment/Dockerfile. from_registry rather than from_dockerfile so Modal
# pulls the multi-gigabyte CUDA base server-side instead of streaming it through the
# authoring machine's connection.
BASE = (
    "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
    "@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755"
)

# The public corpora are regenerated inside the image from the seed rather than
# uploaded. Two reasons. Practically, uploading the 6 MB corpus tree reliably killed
# this machine's connection to Modal partway through, while a few hundred kilobytes of
# source does not. Usefully, it turns the upload into a test: the container hashes what
# it generated against the manifest committed next to the generator, so a run on amd64
# proves the arm64-authored fixtures reproduce bit-for-bit, or fails loudly saying they
# do not. The held-out half is small and is uploaded as-is, so ground truth is never
# regenerated - only checked against.
image = (
    modal.Image.from_registry(BASE, add_python=None)
    .env({"PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    .add_local_file(str(REPO / "environment" / "requirements.txt"),
                    "/tmp/requirements.txt", copy=True)
    .run_commands("pip install --no-cache-dir --require-hashes -r /tmp/requirements.txt")
    .add_local_file(str(REPO / "data" / "generate_corpora.py"),
                    "/build/generate_corpora.py", copy=True)
    .run_commands(
        "python /build/generate_corpora.py"
        " --public-out /app/task_inputs/corpora"
        " --hidden-out /build/discard"
        " --manifest /build/generated_manifest.json",
        "rm -rf /build/discard",
    )
    .add_local_file(str(REPO / "data" / "manifest.json"), "/app/data/manifest.json",
                    copy=True)
    .add_local_dir(str(REPO / "solution"), "/app/solution", copy=True)
    .add_local_dir(str(REPO / "tests"), "/app/tests", copy=True)
)

app = modal.App("poisoned-corpus-calibrate", image=image)
results = modal.Dict.from_name("poisoned-corpus-calibration", create_if_missing=True)


def _measure(trial: int, tag: str) -> dict:
    import json
    import subprocess
    import sys
    import time
    from pathlib import Path

    import numpy as np
    import torch

    sys.path.insert(0, "/app/tests")

    # Before anything is measured: prove the corpora this container generated are the
    # ones the held-out labels belong to. If this fails, every number below is noise.
    import hashlib
    manifest = json.loads(Path("/app/data/manifest.json").read_text())
    mismatched = []
    for key, expected in manifest.items():
        if not key.startswith("corpora/"):
            continue
        path = Path("/app/task_inputs") / key
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            mismatched.append(key)
    if mismatched:
        payload = {"trial": trial, "tag": tag,
                   "error": f"regenerated corpora differ from the manifest: {mismatched}"}
        results[f"{tag}:{trial}"] = payload
        return payload

    t0 = time.time()
    proc = subprocess.run(["python", "/app/solution/run_all.py"],
                          capture_output=True, text=True, cwd="/app")
    oracle_sec = time.time() - t0
    if proc.returncode != 0:
        payload = {"trial": trial, "tag": tag, "error": proc.stderr[-3000:]}
        results[f"{tag}:{trial}"] = payload
        return payload

    import verifier_lib as V

    t1 = time.time()
    accs, baselines, aucs, ess_frac = {}, {}, {}, {}
    for name in V.DEFENDED:
        pub, hid = V.load_public(name), V.load_hidden(name)
        pred = np.load(V.RESULTS / "defense" / name / "pred.npy")
        accs[name] = float((pred == hid["test_y"]).mean())
        baselines[name] = V.reference_accuracy(
            pub["train_x"], pub["train_y"], pub["val_x"], pub["val_y"],
            pub["test_x"], hid["test_y"],
        )
        w = np.load(V.RESULTS / "defense" / name / "weights.npy").astype(np.float64)
        mask = V.poison_mask(pub["meta"]["n_train"], hid["poison_idx"])
        aucs[name] = V.rank_auc(w, mask)
        ess_frac[name] = V.effective_sample_size(w) / w.size

    probe, probe_hidden = V.load_public("probe_corpus"), V.load_hidden("probe_corpus")
    clean = V.reference_accuracy(
        probe["train_x"], probe["train_y"], probe["val_x"], probe["val_y"],
        probe["test_x"], probe_hidden["test_y"],
    )
    poisoned = V.reference_accuracy(
        np.load(V.RESULTS / "attack" / "poisoned_train_x.npy"),
        np.load(V.RESULTS / "attack" / "poisoned_train_y.npy"),
        probe["val_x"], probe["val_y"], probe["test_x"], probe_hidden["test_y"],
    )

    payload = {
        "trial": trial,
        "tag": tag,
        "gpu": torch.cuda.get_device_name(0),
        "oracle_sec": round(oracle_sec, 1),
        "scoring_sec": round(time.time() - t1, 1),
        "mean_accuracy": float(np.mean(list(accs.values()))),
        "gain_over_baseline": float(np.mean(list(accs.values()))
                                    - np.mean(list(baselines.values()))),
        "attack_drop": float(clean - poisoned),
        "worst_rank_auc": float(max(aucs.values())),
        "min_ess_fraction": float(min(ess_frac.values())),
        "per_corpus_accuracy": accs,
        "per_corpus_rank_auc": aucs,
    }
    results[f"{tag}:{trial}"] = payload
    return payload


# One function per GPU model. The point of running the same deterministic oracle on
# different silicon is to find out whether the graded metrics move when the hardware
# does - the failure the accuracy gate has to be set wide enough to survive.
@app.function(gpu="T4", timeout=3600)
def measure(trial: int) -> dict:
    return _measure(trial, "T4")


@app.function(gpu="L4", timeout=3600)
def measure_l4(trial: int) -> dict:
    return _measure(trial, "L4")


@app.function(gpu="A10G", timeout=3600)
def measure_a10g(trial: int) -> dict:
    return _measure(trial, "A10G")


@app.function(gpu="T4", timeout=3600)
def validate(trial: int, nop: bool = False) -> dict:
    """Full trial: run solve.sh, then the real verifier entrypoint, report the reward.

    This is the end-to-end check, as opposed to `measure`, which computes the graded
    quantities directly. Here nothing is short-circuited: solve.sh runs as the agent
    would run it and tests/test.sh grades the result and writes reward.txt.
    """
    import subprocess
    import time
    from pathlib import Path

    tag = "nop" if nop else "oracle"
    logs = Path("/tmp/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)

    oracle_sec = 0.0
    if not nop:
        t0 = time.time()
        run = subprocess.run(["bash", "/app/solution/solve.sh"],
                             capture_output=True, text=True, cwd="/app")
        oracle_sec = time.time() - t0
        if run.returncode != 0:
            payload = {"trial": trial, "tag": tag, "error": run.stderr[-3000:]}
            results[f"validate-{tag}:{trial}"] = payload
            return payload

    t1 = time.time()
    # Inherit the environment and add to it. Replacing it outright drops the conda
    # prefix this image puts python on, and the verifier then dies with exit 127
    # before running a single check - which looks exactly like a failing verifier.
    import os
    verify = subprocess.run(
        ["bash", "/app/tests/test.sh"],
        capture_output=True, text=True, cwd="/app",
        env={**os.environ, "LOG_DIR": str(logs), "PYTHONUNBUFFERED": "1"},
    )
    verifier_sec = time.time() - t1

    reward_file = logs / "reward.txt"
    reward = reward_file.read_text().strip() if reward_file.exists() else "missing"
    summary = [ln for ln in verify.stdout.splitlines()
               if " passed" in ln or " failed" in ln or " error" in ln]

    payload = {
        "trial": trial,
        "tag": tag,
        "reward": reward,
        "verifier_exit": verify.returncode,
        "oracle_sec": round(oracle_sec, 1),
        "verifier_sec": round(verifier_sec, 1),
        "summary": summary[-2:],
        "stdout_tail": verify.stdout[-1500:],
        "stderr_tail": verify.stderr[-800:],
        "failed": sorted({ln.split("::")[1].split()[0]
                          for ln in verify.stdout.splitlines()
                          if ln.startswith(("FAILED", "ERROR")) and "::" in ln}),
    }
    results[f"validate-{tag}:{trial}"] = payload
    return payload


@app.function(timeout=3600)
def time_on_cpu(trial: int) -> dict:
    """Same image, same class of host, no GPU. The honest denominator.

    Claiming a task needs a GPU is a claim about the workload, and the only way to
    check it is to run the identical code on the identical image without one.
    """
    import subprocess
    import time

    t0 = time.time()
    run = subprocess.run(["python", "/app/solution/run_all.py", "--allow-cpu"],
                         capture_output=True, text=True, cwd="/app")
    payload = {"trial": trial, "tag": "cpu-timing",
               "oracle_sec": round(time.time() - t0, 1),
               "returncode": run.returncode,
               "error": run.stderr[-1500:] if run.returncode else None}
    results[f"cputime:{trial}"] = payload
    return payload


@app.local_entrypoint()
def main(trials: int = 10):
    got = list(measure.map(range(trials)))
    print(json.dumps(got, indent=2))


def collect() -> None:
    """Read whatever has landed in the Dict, from a fresh short-lived connection."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "calibration.json")
    args = ap.parse_args()

    store = modal.Dict.from_name("poisoned-corpus-calibration", create_if_missing=True)
    rows = [v for _, v in sorted(store.items(), key=lambda kv: str(kv[0]))]
    good = [r for r in rows if "error" not in r]
    print(f"{len(rows)} trials recorded, {len(good)} successful")
    if good:
        args.out.write_text(json.dumps(good, indent=2) + "\n")
        print(f"wrote {args.out}")
    for r in rows:
        if "error" in r:
            print(f"trial {r['trial']} FAILED: {r['error'][-400:]}")


if __name__ == "__main__":
    collect()
