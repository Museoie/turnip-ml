#!/usr/bin/env python3
"""Deterministic train/val/holdout split grouped by source_id.

Splits a training-sample manifest into train / validation / holdout sets for
the trick-detection training pipeline (see Training/README.md).

Contract:
  - Holdout is exactly the samples the farm flags ``"holdout": true``:
    the maintainer-curated footage (TRAINING_DESIGN.md §2.6), never
    trained on (poisoning defense). A missing flag means false. The flag
    belongs to the source: samples of one source that disagree are an
    error, so a source never straddles the holdout boundary.
  - The rest is split train/val, target ratios 90/10, configurable.
  - Grouped by ``source_id``: every sample from one source lands in a
    single split, so no source's clips ever leak across splits. Training
    data carries no account identifier, so one person's different sources
    can land in different splits; the curated holdout is the check on
    that.
  - ``--require-holdout`` refuses a manifest with no flagged sample, so
    the pipeline stops before training rather than at the holdout
    evaluation.
  - Deterministic and seeded: the same manifest + seed + Python interpreter
    version always produces the byte-identical mapping, so training runs are
    reproducible (CPython does not guarantee ``random.Random`` shuffle order
    stable across interpreter versions).

Source integrity is the hard constraint; the ratios are targets. Sources
are indivisible, so the realized ratios only approximate 90/10 (e.g. a
single unflagged source lands 100% in train).

Determinism: all ordering derives from sorted ids plus a seeded
``random.Random``; no timestamps, no unseeded RNG, no dict-iteration order
dependence. Input record order does not affect the result. Byte-identical
reproduction additionally requires the same Python interpreter version:
CPython does not guarantee ``random.Random`` shuffle order across versions.

Usage:
  split_dataset.py --in manifest.jsonl --out split.json [--seed 0]
                   [--train-ratio 0.9 --val-ratio 0.1] [--require-holdout]
Input JSONL: one object per line, each with "sample_id" and "source_id",
and optionally "holdout" (a boolean; extra fields are ignored).
Output JSON: {"<sample_id>": "train"|"val"|"holdout"}, keys sorted.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Iterable, Mapping

SPLITS = ("train", "val", "holdout")
DEFAULT_SEED = 0
DEFAULT_RATIOS = (0.9, 0.1)


def _check_ratios(train_ratio: float, val_ratio: float) -> None:
    ratios = (train_ratio, val_ratio)
    if any(r < 0 for r in ratios):
        raise ValueError(f"split ratios must be non-negative, got {ratios}")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"split ratios must sum to 1.0, got {ratios}")


def _normalize(records: Iterable[Mapping]) -> list[tuple[str, str, bool]]:
    """Validate records into (sample_id, source_id, holdout); fail closed."""
    pairs: list[tuple[str, str, bool]] = []
    seen: set[str] = set()
    for i, rec in enumerate(records):
        if not isinstance(rec, Mapping):
            # ValueError is this module's single fail-closed error type.
            raise ValueError(f"record {i} is not an object: {rec!r}")  # noqa: TRY004
        sid = rec.get("sample_id")
        src = rec.get("source_id")
        if sid is None or src is None or str(sid) == "" or str(src) == "":
            raise ValueError(
                f"record {i} needs non-empty sample_id and source_id: {rec!r}"
            )
        holdout = rec.get("holdout", False)
        if not isinstance(holdout, bool):
            raise ValueError(f"record {i}: holdout must be a boolean")  # noqa: TRY004
        sid, src = str(sid), str(src)
        if sid in seen:
            raise ValueError(f"duplicate sample_id: {sid!r}")
        seen.add(sid)
        pairs.append((sid, src, holdout))
    return pairs


def split_records(
    records: Iterable[Mapping],
    *,
    seed: int = DEFAULT_SEED,
    train_ratio: float = DEFAULT_RATIOS[0],
    val_ratio: float = DEFAULT_RATIOS[1],
    require_holdout: bool = False,
) -> dict[str, str]:
    """Assign each sample to train/val/holdout, grouped by source_id.

    Returns {sample_id: split}. See the module docstring for the contract.
    """
    _check_ratios(train_ratio, val_ratio)
    pairs = _normalize(records)

    # Group samples per source; canonical (sorted) order everywhere for
    # determinism.
    by_source: dict[str, list[str]] = {}
    flags: dict[str, bool] = {}
    for sid, src, holdout in pairs:
        by_source.setdefault(src, []).append(sid)
        if flags.setdefault(src, holdout) != holdout:
            raise ValueError("samples of one source disagree on holdout")
    for sids in by_source.values():
        sids.sort()

    holdout_sources = sorted(src for src, flagged in flags.items() if flagged)
    if require_holdout and not holdout_sources:
        raise ValueError("no sample is flagged holdout: nothing to evaluate on")
    if not pairs:
        return {}

    assignment: dict[str, str] = {}
    for src in holdout_sources:
        for sid in by_source[src]:
            assignment[sid] = "holdout"

    # Remaining sources: seeded shuffle, then greedy deficit-filling against
    # the ratio targets. The seed decides *which* sources land where; the greedy
    # pass keeps the realized ratios near the targets.
    rest = sorted(src for src, flagged in flags.items() if not flagged)
    total = sum(len(by_source[src]) for src in rest)
    targets = {"train": train_ratio * total, "val": val_ratio * total}
    counts = {"train": 0, "val": 0}
    rng = random.Random(seed)
    rng.shuffle(rest)
    order = ("train", "val")
    for src in rest:
        # Split with the largest remaining deficit; ties break toward train.
        best = max(order, key=lambda s: (targets[s] - counts[s], -order.index(s)))
        for sid in by_source[src]:
            assignment[sid] = best
        counts[best] += len(by_source[src])

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
    parser.add_argument(
        "--in",
        dest="inp",
        required=True,
        help="input manifest as JSONL (sample_id, source_id, holdout? per line)",
    )
    parser.add_argument(
        "--out", dest="out", required=True, help="output JSON path {sample_id: split}"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="RNG seed; same manifest + seed + interpreter = byte-identical output",
    )
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_RATIOS[0])
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_RATIOS[1])
    parser.add_argument(
        "--require-holdout",
        action="store_true",
        help="refuse a manifest with no sample flagged holdout",
    )
    args = parser.parse_args(argv)

    records = _read_jsonl(args.inp)
    assignment = split_records(
        records,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        require_holdout=args.require_holdout,
    )
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(dump_split(assignment))

    n = len(assignment)
    by_split = {s: sum(1 for v in assignment.values() if v == s) for s in SPLITS}
    print(
        f"split {n} samples (seed={args.seed}): "
        + ", ".join(f"{s}={by_split[s]}" for s in SPLITS),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
