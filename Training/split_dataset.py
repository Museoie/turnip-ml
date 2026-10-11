#!/usr/bin/env python3
"""Deterministic stratified train/val/holdout split by user_id.

Splits a training-sample manifest into train / validation / holdout sets for
the trick-detection training pipeline (see Training/README.md).

Contract:
  - Target ratios 80/10/10 (train/val/holdout), configurable.
  - Stratified by ``user_id``: every sample from one user lands in a single
    split, so no user's clips ever leak across splits.
  - ``holdout_ids`` pins the admin-curated holdout set (poisoning defense):
    pinned samples are never trained on — and neither are their siblings,
    because the whole user goes to holdout with them.
  - Deterministic and seeded: the same manifest + seed + Python interpreter
    version always produces the byte-identical mapping, so training runs are
    reproducible (CPython does not guarantee ``random.Random`` shuffle order
    stable across interpreter versions).

User integrity is the hard constraint; the ratios are targets. Users are
indivisible, so the realized ratios only approximate 80/10/10 (e.g. a
single-user manifest lands 100% in train).

Determinism: all ordering derives from sorted ids plus a seeded
``random.Random``; no timestamps, no unseeded RNG, no dict-iteration order
dependence. Input record order does not affect the result. Byte-identical
reproduction additionally requires the same Python interpreter version:
CPython does not guarantee ``random.Random`` shuffle order across versions.

Usage:
  split_dataset.py --in manifest.jsonl --out split.json [--seed 0]
                   [--train-ratio 0.8 --val-ratio 0.1 --holdout-ratio 0.1]
                   [--holdout-ids holdout.txt]
Input JSONL: one object per line, each with "sample_id" and "user_id"
(extra fields are ignored). ``--holdout-ids`` is a text file with one
sample id per line.
Output JSON: {"<sample_id>": "train"|"val"|"holdout"}, keys sorted.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from typing import Iterable, Mapping

SPLITS = ("train", "val", "holdout")
DEFAULT_SEED = 0
DEFAULT_RATIOS = (0.8, 0.1, 0.1)


def _check_ratios(train_ratio: float, val_ratio: float, holdout_ratio: float) -> None:
    ratios = (train_ratio, val_ratio, holdout_ratio)
    if any(r < 0 for r in ratios):
        raise ValueError(f"split ratios must be non-negative, got {ratios}")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"split ratios must sum to 1.0, got {ratios}")


def _normalize(records: Iterable[Mapping]) -> list[tuple[str, str]]:
    """Validate records into (sample_id, user_id) pairs; fail closed."""
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for i, rec in enumerate(records):
        if not isinstance(rec, Mapping):
            raise ValueError(f"record {i} is not an object: {rec!r}")
        sid = rec.get("sample_id")
        uid = rec.get("user_id")
        if sid is None or uid is None or str(sid) == "" or str(uid) == "":
            raise ValueError(f"record {i} needs non-empty sample_id and user_id: {rec!r}")
        sid, uid = str(sid), str(uid)
        if sid in seen:
            raise ValueError(f"duplicate sample_id: {sid!r}")
        seen.add(sid)
        pairs.append((sid, uid))
    return pairs


def split_records(
    records: Iterable[Mapping],
    *,
    seed: int = DEFAULT_SEED,
    train_ratio: float = DEFAULT_RATIOS[0],
    val_ratio: float = DEFAULT_RATIOS[1],
    holdout_ratio: float = DEFAULT_RATIOS[2],
    holdout_ids: Iterable[str] = (),
) -> dict[str, str]:
    """Assign each sample to train/val/holdout, stratified by user_id.

    Returns {sample_id: split}. See the module docstring for the contract.
    """
    _check_ratios(train_ratio, val_ratio, holdout_ratio)
    pairs = _normalize(records)

    # Validate holdout ids even for an empty manifest: a stale holdout-ids
    # file against an empty/wrong manifest must fail closed, not silently
    # write an empty split.json.
    pinned = {str(h) for h in holdout_ids}
    known = {sid for sid, _ in pairs}
    unknown = sorted(pinned - known)
    if unknown:
        raise ValueError(f"holdout_ids not present in manifest: {unknown}")

    if not pairs:
        return {}

    # Group samples per user; canonical (sorted) order everywhere for determinism.
    by_user: dict[str, list[str]] = {}
    for sid, uid in pairs:
        by_user.setdefault(uid, []).append(sid)
    for uid in by_user:
        by_user[uid].sort()

    assignment: dict[str, str] = {}
    counts = {s: 0 for s in SPLITS}

    # Admin-curated holdout: a pinned sample pulls its whole user into
    # holdout, so no user straddles the holdout boundary (poisoning defense).
    pinned_users = sorted({uid for sid, uid in pairs if sid in pinned})
    for uid in pinned_users:
        for sid in by_user[uid]:
            assignment[sid] = "holdout"
        counts["holdout"] += len(by_user[uid])

    # Remaining users: seeded shuffle, then greedy deficit-filling against the
    # ratio targets. The seed decides *which* users land where; the greedy
    # pass keeps the realized ratios near the targets.
    total = len(pairs)
    targets = {
        "train": train_ratio * total,
        "val": val_ratio * total,
        "holdout": holdout_ratio * total,
    }
    rng = random.Random(seed)
    pinned_set = set(pinned_users)
    rest = sorted(u for u in by_user if u not in pinned_set)
    rng.shuffle(rest)
    for uid in rest:
        # Split with the largest remaining deficit; ties break toward train.
        best = max(SPLITS, key=lambda s: (targets[s] - counts[s], -SPLITS.index(s)))
        for sid in by_user[uid]:
            assignment[sid] = best
        counts[best] += len(by_user[uid])

    return assignment


def dump_split(assignment: Mapping[str, str]) -> str:
    """Serialize a split mapping deterministically (sorted keys)."""
    return json.dumps(dict(assignment), indent=2, sort_keys=True) + "\n"


def _read_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {e}") from e
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in", dest="inp", required=True,
                        help="input manifest as JSONL (sample_id, user_id per line)")
    parser.add_argument("--out", dest="out", required=True,
                        help="output JSON path {sample_id: split}")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="RNG seed; same manifest + seed + interpreter = byte-identical output")
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_RATIOS[0])
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_RATIOS[1])
    parser.add_argument("--holdout-ratio", type=float, default=DEFAULT_RATIOS[2])
    parser.add_argument("--holdout-ids", default=None,
                        help="text file with admin-curated holdout sample ids, one per line")
    args = parser.parse_args(argv)

    holdout_ids: list[str] = []
    if args.holdout_ids:
        with open(args.holdout_ids, "r", encoding="utf-8") as fh:
            holdout_ids = [ln.strip() for ln in fh if ln.strip()]

    records = _read_jsonl(args.inp)
    assignment = split_records(
        records,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        holdout_ratio=args.holdout_ratio,
        holdout_ids=holdout_ids,
    )
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(dump_split(assignment))

    n = len(assignment)
    by_split = {s: sum(1 for v in assignment.values() if v == s) for s in SPLITS}
    print(f"split {n} samples (seed={args.seed}): " +
          ", ".join(f"{s}={by_split[s]}" for s in SPLITS), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
