#!/usr/bin/env python3
"""Val-minus-holdout person-leakage gap (turnip-ml#42, TRAINING_DESIGN.md §2.6).

Person leakage is a measured risk: the export carries no account
identifier, so one person's videos can land in both train and val, and
val is what the promotion gate reads. The holdout is built from footage
of people outside the contributor pool, so a model that learned
individuals rather than tricks shows up as a gap between val and
holdout. Every run reports that gap; a gap above 0.05 absolute on three
consecutive promoted runs flags the split-grouping question.

Inputs
------
--metrics      metrics.json from Training/train.py (val_accuracy,
               val_segments, split_sha256).
--report       holdout_report.json from Training/evaluate.py (f1,
               name_accuracy_given_detection, iou_threshold,
               inputs.split_sha256).
--split        split.json both were computed on. Checked against both
               SHAs, like name_metrics.py, so the gap always describes
               one run.
--samples-dir  sample JSON dir (val ground truth).
--registry     decision log (JSON array; default Training/registry.json).
               Gap records are appended; the log is what the nightly
               trigger reads, so the streak survives fresh checkouts
               via R2.
--promotion    this run's promotion decision: promote | archive |
               bootstrap | not-run.

What it computes
----------------
- promotion_metric gap: val_accuracy − holdout name_accuracy_given_detection.
- segment_f1 gap: val segment F1 − holdout f1. Val F1 is computed on the
  same footing as holdout F1: metrics.json's val_segments decoded
  exactly as predict.py decodes them (name.split("+") into names), scored
  by evaluate.compute_metrics against val ground truth at the report's
  IoU threshold.
- streak: consecutive promoted runs with either gap above 0.05 absolute.
  A promoted run at or under threshold, or a non-promoted run (archive,
  not-run), resets the streak to 0. A bootstrap run leaves it unchanged:
  it passed no gate, so it is neither a promoted run nor a break in the
  pattern. Runs where this script never ran leave no record and do not
  move the streak either way.
- flagged: streak >= 3.

Outputs
-------
- The report gains "val_minus_holdout": both gaps, the four inputs, the
  streak, and the flag (same format evaluate.py writes: indent 2, sorted
  keys, trailing newline).
- A {"kind": "leakage_gap", ...} record is appended to the registry
  (atomic write, like promote.py). Other record kinds in the log are
  ignored when replaying the streak.
- Stdout carries aggregates only (gaps, streak, flag). Per-sample
  outputs stay in data/run/; no sample ids, no account-shaped data.

Exit codes: 0 = written, 2 = input error (fail closed): unreadable or
malformed inputs, split mismatch, a val_segments entry outside the val
split, or a malformed existing registry.

Determinism: same inputs -> byte-identical report and registry payload
(sorted keys, floats rounded to 6 decimals), except the registry
record's wall-clock `decided_at` (a registry is a log, not a
reproducible artifact).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import evaluate

GAP_THRESHOLD = 0.05
FLAG_STREAK = 3
PROMOTIONS = ("promote", "archive", "bootstrap", "not-run")
EXIT_OK = 0
EXIT_ERROR = 2


class GapError(Exception):
    """Input error: the script refuses to compute rather than guess."""


def _fail(msg: str) -> Any:
    raise GapError(msg)


def _load_json(path: Path) -> Any:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
        _fail(f"cannot read JSON from {path}: {e}")


def _unit(value: Any, where: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        _fail(f"{where} must be a number in [0, 1]")
    return float(value)


def _iou_threshold(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 < value <= 1.0
    ):
        _fail(f"iou_threshold must be a number in (0, 1]")
    return float(value)


def _val_predictions(
    val_segments: Any, val_ids: list[str]
) -> dict[str, list[tuple[int, int, frozenset[str]]]]:
    """metrics.json val_segments as evaluate.py predictions.

    Decoded exactly as predict.py decodes them: a segment's `name` splits
    on "+" into names (a combo label is its names, not one name).
    """
    if not isinstance(val_segments, list):
        _fail("metrics val_segments must be a list")
    val_set = set(val_ids)
    pred: dict[str, list[tuple[int, int, frozenset[str]]]] = {
        sid: [] for sid in val_ids
    }
    for i, seg in enumerate(val_segments):
        where = f"metrics val_segments[{i}]"
        if not isinstance(seg, dict):
            _fail(f"{where} must be an object")
        sid = seg.get("sample_id")
        if sid not in val_set:
            _fail(f"{where} sample_id {sid!r} is not in the val split")
        start, end = seg.get("start_frame"), seg.get("end_frame")
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or isinstance(start, bool)
            or isinstance(end, bool)
            or not 0 <= start < end
        ):
            _fail(f"{where} needs 0 <= start_frame < end_frame")
        name = seg.get("name")
        if not isinstance(name, str) or not name:
            _fail(f"{where} needs a non-empty name")
        pred[sid].append((start, end, frozenset(name.split("+"))))
    return pred


def compute_gaps(
    metrics: Any, report: Any, split: Any, samples_dir: Path
) -> dict[str, float]:
    """Both val-minus-holdout gaps plus their inputs, rounded to 6."""
    if not isinstance(metrics, dict):
        _fail("metrics must be a JSON object")
    if not isinstance(report, dict):
        _fail("report must be a JSON object")
    if not isinstance(split, dict):
        _fail("split must be a JSON object of sample_id -> split")
    val_accuracy = _unit(metrics.get("val_accuracy"), "metrics val_accuracy")
    holdout_acc = _unit(
        report.get("name_accuracy_given_detection"),
        "report name_accuracy_given_detection",
    )
    holdout_f1 = _unit(report.get("f1"), "report f1")
    iou = _iou_threshold(report.get("iou_threshold"))

    val_ids = sorted(sid for sid, part in split.items() if part == "val")
    gt = evaluate.load_ground_truth(samples_dir, val_ids)
    pred = _val_predictions(metrics.get("val_segments"), val_ids)
    val_f1 = evaluate.compute_metrics(gt, pred, iou)["f1"]
    val_f1 = _unit(val_f1, "computed val f1")

    return {
        "promotion_metric": evaluate.r6(val_accuracy - holdout_acc),
        "segment_f1": evaluate.r6(val_f1 - holdout_f1),
        "val_accuracy": evaluate.r6(val_accuracy),
        "val_f1": evaluate.r6(val_f1),
        "holdout_name_accuracy": evaluate.r6(holdout_acc),
        "holdout_f1": evaluate.r6(holdout_f1),
    }


def _over_threshold(gaps: dict[str, float]) -> bool:
    return (
        abs(gaps["promotion_metric"]) > GAP_THRESHOLD
        or abs(gaps["segment_f1"]) > GAP_THRESHOLD
    )


def replay_streak(records: list[Any]) -> int:
    """Streak over prior leakage_gap records, in log order."""
    streak = 0
    for rec in records:
        if not isinstance(rec, dict) or rec.get("kind") != "leakage_gap":
            continue
        promotion = rec.get("promotion")
        gaps = rec.get("val_minus_holdout")
        if promotion == "bootstrap":
            continue
        if promotion == "promote" and isinstance(gaps, dict):
            over = abs(gaps.get("promotion_metric", 0)) > GAP_THRESHOLD or (
                abs(gaps.get("segment_f1", 0)) > GAP_THRESHOLD
            )
            streak = streak + 1 if over else 0
        else:
            streak = 0
    return streak


def run(
    metrics_path: Path,
    report_path: Path,
    split_path: Path,
    samples_dir: Path,
    registry_path: Path,
    promotion: str,
) -> dict[str, Any]:
    metrics = _load_json(metrics_path)
    report = _load_json(report_path)
    split = _load_json(split_path)
    if not isinstance(split, dict):
        _fail("split must be a JSON object of sample_id -> split")
    for part in split.values():
        if part not in evaluate.SPLITS:
            _fail(f"split has unknown part {part!r}")
    split_sha = evaluate.sha256_file(split_path)
    if not isinstance(metrics, dict) or metrics.get("split_sha256") != split_sha:
        _fail("metrics split_sha256 does not match split.json")
    inputs = report.get("inputs") if isinstance(report, dict) else None
    if not isinstance(inputs, dict) or inputs.get("split_sha256") != split_sha:
        _fail("report inputs.split_sha256 does not match split.json")

    gaps = compute_gaps(metrics, report, split, samples_dir)

    registry = load_registry(registry_path)
    streak = replay_streak(registry)
    if promotion == "promote":
        streak = streak + 1 if _over_threshold(gaps) else 0
    elif promotion == "bootstrap":
        pass  # neither a promoted run nor a break
    else:
        streak = 0
    flagged = streak >= FLAG_STREAK

    report["val_minus_holdout"] = {
        **gaps,
        "streak_promoted_runs_over_threshold": streak,
        "flagged": flagged,
    }
    tmp = report_path.with_suffix(report_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    tmp.replace(report_path)

    record: dict[str, Any] = {
        "kind": "leakage_gap",
        "decided_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "promotion": promotion,
        "val_minus_holdout": gaps,
        "streak_promoted_runs_over_threshold": streak,
        "flagged": flagged,
    }
    registry.append(record)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = registry_path.with_suffix(registry_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(registry, fh, indent=2, sort_keys=True)
        fh.write("\n")
    tmp.replace(registry_path)
    return record


def load_registry(path: Path) -> list[Any]:
    """promote.py's registry read, without importing promote (no numpy)."""
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
        _fail(f"cannot read registry file {path}: {e}")
    if not isinstance(data, list):
        _fail(f"registry file {path} must contain a JSON array")
    return data


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Val-minus-holdout leakage gap.")
    ap.add_argument("--metrics", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--split", required=True, type=Path)
    ap.add_argument("--samples-dir", required=True, type=Path)
    ap.add_argument("--registry", required=True, type=Path)
    ap.add_argument("--promotion", required=True, choices=PROMOTIONS)
    args = ap.parse_args(argv)
    try:
        record = run(
            args.metrics,
            args.report,
            args.split,
            args.samples_dir,
            args.registry,
            args.promotion,
        )
    except (GapError, ValueError) as e:
        print(f"leakage_gap.py: error: {e}", file=sys.stderr)
        return EXIT_ERROR
    gaps = record["val_minus_holdout"]
    print(
        f"val-minus-holdout: promotion {gaps['promotion_metric']:+.4f}, "
        f"segment_f1 {gaps['segment_f1']:+.4f}, "
        f"streak {record['streak_promoted_runs_over_threshold']}, "
        f"flagged {record['flagged']}"
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
