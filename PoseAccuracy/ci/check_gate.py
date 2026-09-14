#!/usr/bin/env python3
"""Compare a pose-accuracy score against the baseline and enforce the gate.

Exit 0: score within tolerance, or the `pose-accuracy-override` label is set.
Exit 1: regression beyond tolerance (fails the CI check).

The override path is a manual, auditable bypass: a maintainer adds the
`pose-accuracy-override` label to the PR and re-runs the job. Who/when is
resolved by the workflow's override step and passed via --override-info.
"""
from __future__ import annotations
import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True, help="scorer output JSON")
    ap.add_argument("--baseline", required=True, help="PoseAccuracy/baseline.json")
    ap.add_argument("--tolerance", type=float, required=True,
                    help="allowed drop below baseline, in score points")
    ap.add_argument("--override", choices=["true", "false"], default="false")
    ap.add_argument("--override-info", default="",
                    help="human-readable who/when the override label was added")
    a = ap.parse_args()

    with open(a.result) as f:
        score = json.load(f)["score"]
    with open(a.baseline) as f:
        baseline = json.load(f)
    base_score = baseline["score"]

    if a.override == "true":
        print(f"::notice::pose-accuracy gate OVERRIDDEN ({a.override_info}). "
              f"Score {score:.2f} vs baseline {base_score:.2f} — not enforced.")
        return 0

    floor = base_score - a.tolerance
    print(f"pose-accuracy score: {score:.2f} | baseline: {base_score:.2f} "
          f"| tolerance: {a.tolerance:.2f} | floor: {floor:.2f}")
    if score < floor:
        print(f"::error::Pose-accuracy regression: score {score:.2f} is below "
              f"baseline {base_score:.2f} minus tolerance {a.tolerance:.2f}. "
              f"If this regression is accepted, add the `pose-accuracy-override` "
              f"label to the PR and re-run.")
        return 1
    print("pose-accuracy gate: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
