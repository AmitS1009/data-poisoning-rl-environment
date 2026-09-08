#!/bin/bash
# Harbor verifier entrypoint. Copied to /tests and run inside the task container.
#
# Everything the suite needs is already in the image (numpy, scipy, torch, pytest are
# installed at build time and the environment has no network), so this installs
# nothing and simply runs the suite.
set -uo pipefail

LOG_DIR="${LOG_DIR:-/logs/verifier}"
mkdir -p "${LOG_DIR}"

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -p no:cacheprovider: the verifier must not write .pytest_cache into a graded tree.
python -m pytest \
  "${TESTS_DIR}/test_outputs.py" \
  -p no:cacheprovider \
  --ctrf "${LOG_DIR}/ctrf.json" \
  -rA -q
status=$?

if [ ${status} -eq 0 ]; then
  echo 1 > "${LOG_DIR}/reward.txt"
else
  echo 0 > "${LOG_DIR}/reward.txt"
fi

exit ${status}
