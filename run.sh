#!/usr/bin/env bash
# Entry point for everything you can do with this task locally.
#
#   ./run.sh generate      regenerate the corpora from the seed
#   ./run.sh check-data    regenerate into a temp tree and diff against the manifest
#   ./run.sh build         build the task image with Docker
#   ./run.sh oracle        run the oracle (needs CUDA)
#   ./run.sh oracle --cpu  run the oracle on CPU, for authoring only
#   ./run.sh verify        run the verifier against ./results
#   ./run.sh adversarial   run every adversarial solution through the verifier
#   ./run.sh cpu-suite     check-data + oracle --cpu + verify + adversarial
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
cd "${HERE}"

usage() { sed -n '2,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

case "${1:-}" in
  generate)
    "${PY}" data/generate_corpora.py "${@:2}"
    ;;

  check-data)
    echo "== regenerating from the seed and comparing to data/manifest.json =="
    "${PY}" data/generate_corpora.py --verify
    echo "== exercising the tiny configuration =="
    tmp="$(mktemp -d)"
    "${PY}" data/generate_corpora.py --tiny \
        --public-out "${tmp}/corpora" --hidden-out "${tmp}/hidden" \
        --manifest "${tmp}/manifest.json" >/dev/null
    "${PY}" - "${tmp}" <<'PYEOF'
import json, sys
from pathlib import Path
import numpy as np
root = Path(sys.argv[1])
for name in ("probe_corpus", "alpha_feed", "beta_feed", "gamma_feed"):
    x = np.load(root / "corpora" / name / "train_x.npy")
    meta = json.loads((root / "corpora" / name / "meta.json").read_text())
    assert x.shape == (meta["n_train"], meta["n_bins"]), (name, x.shape)
    assert meta["n_bins"] == 24 and meta["n_classes"] == 3, meta
print("tiny configuration generates consistent shapes")
PYEOF
    ;;

  build)
    docker build -t poisoned-corpus-defense:local environment/
    ;;

  oracle)
    APP_DIR="${HERE}" "${PY}" solution/run_all.py "${@:2}"
    ;;

  verify)
    "${PY}" -m pytest tests/test_outputs.py -p no:cacheprovider -q "${@:2}"
    ;;

  adversarial)
    "${PY}" scripts/adversarial.py "${@:2}"
    ;;

  cpu-suite)
    "${HERE}/run.sh" check-data
    "${HERE}/run.sh" oracle --allow-cpu
    "${HERE}/run.sh" verify
    "${HERE}/run.sh" adversarial
    ;;


  ""|-h|--help|help)
    usage
    ;;

  *)
    echo "unknown command: $1" >&2
    usage >&2
    exit 2
    ;;
esac
