"""Fail if the Modal image definitions have drifted from environment/Dockerfile.

The graded image is built from the Dockerfile. The Modal authoring scripts rebuild an
equivalent image through Modal's API instead, because streaming a docker build of a
multi-gigabyte CUDA base through the authoring machine's connection does not survive.
Two definitions of one environment is a maintenance hazard: calibrate against one, grade
against the other, and the thresholds quietly stop meaning anything.

This checks the parts that can actually change the answer - the pinned base digest and
the hash-locked requirements install - rather than diffing the files, which would fail
on formatting.

    python scripts/check_image_parity.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOCKERFILE = REPO / "environment" / "Dockerfile"
MODAL_SCRIPTS = (REPO / "scripts" / "modal_calibrate.py",
                 REPO / "scripts" / "modal_oracle.py")

DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


def dockerfile_base() -> tuple[str, str]:
    text = DOCKERFILE.read_text()
    from_line = next(ln for ln in text.splitlines() if ln.startswith("FROM "))
    image = from_line.split(None, 1)[1].strip()
    digest = DIGEST.search(image)
    if not digest:
        raise SystemExit(f"{DOCKERFILE} does not pin its base image by digest: {image}")
    return image.split("@")[0], digest.group(0)


def main() -> int:
    repo_tag, repo_digest = dockerfile_base()
    docker_text = DOCKERFILE.read_text()
    problems: list[str] = []

    requires_hashes = "--require-hashes" in docker_text
    if not requires_hashes:
        problems.append(f"{DOCKERFILE.name} does not install with --require-hashes")

    for script in MODAL_SCRIPTS:
        if not script.exists():
            continue
        text = script.read_text()
        name = script.name

        digests = set(DIGEST.findall(text))
        if not digests:
            problems.append(f"{name} does not pin a base image digest")
        elif digests != {repo_digest}:
            problems.append(
                f"{name} pins {sorted(digests)} but the Dockerfile pins {repo_digest}")

        if repo_tag not in text:
            problems.append(f"{name} does not reference the base image {repo_tag}")

        if "--require-hashes" not in text:
            problems.append(f"{name} installs requirements without --require-hashes")

        if "requirements.txt" not in text:
            problems.append(f"{name} does not install from requirements.txt")

    if problems:
        print("image definitions have drifted:")
        for p in problems:
            print(f"  - {p}")
        return 1

    print(f"base image and dependency install agree across {DOCKERFILE.name} "
          f"and {len([s for s in MODAL_SCRIPTS if s.exists()])} Modal scripts")
    print(f"  base:   {repo_tag}")
    print(f"  digest: {repo_digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
