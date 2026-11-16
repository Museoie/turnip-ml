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
  - Every other source goes to val or train by a stable hash of its
    ``source_id`` alone (``source_bucket``): val iff the first 8 bytes of
    SHA-256("turnip-split-v2:" + source_id), read as a big-endian
    integer, are below ``val_ratio`` * 2**64. A source's split therefore
    never depends on which other sources are in the manifest, their
    order, or the Python version, and it is the same on every run: a new
    source never moves an old one. That matters because the nightly run
    fine-tunes the champion: a source it trained on must never come back
    as validation, where it would inflate the val metric and pose as a
    non-member in the membership-inference check.
  - Grouped by ``source_id``: every sample from one source lands in a
    single split, so no source's clips ever leak across splits. Training
    data carries no account identifier, so one person's different sources
    can land in different splits; the curated holdout is the check on
    that.
  - ``--require-holdout`` refuses a manifest with no flagged sample, so
    the pipeline stops before training rather than at the holdout
    evaluation.
``val_ratio`` (default 0.1) is each source's probability of landing in
val, not a target the split steers toward: the realized share of sources
in val is only near it, and the share of samples also depends on how
many samples each source holds (a small manifest can put every source on
one side). There is no train ratio and no seed: train is the complement,
and a seed would only be a second salt that, changed, would reshuffle
every source.

Determinism: the mapping is a pure function of the manifest and
val_ratio, byte-identical on any Python version; no RNG, timestamps or
input-order dependence.

Usage:
  split_dataset.py --in manifest.jsonl --out split.json
                   [--val-ratio 0.1] [--require-holdout]
Input JSONL: one object per line, each with "sample_id" and "source_id",
and optionally "holdout" (a boolean; extra fields are ignored).
Output JSON: {"<sample_id>": "train"|"val"|"holdout"}, keys sorted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Iterable, Mapping

SPLITS = ("train", "val", "holdout")
DEFAULT_VAL_RATIO = 0.1
# Versions the hash (v1 was the seeded shuffle it replaces): changing it
# reassigns every source, so it changes only with a deliberate
# from-scratch retrain.
SPLIT_SALT = "turnip-split-v2:"
BUCKET_BITS = 64


def source_bucket(source_id: str) -> int:
    """The source's fixed position in [0, 2**64): the first 8 bytes of
    SHA-256(SPLIT_SALT + source_id), big-endian."""
    digest = hashlib.sha256((SPLIT_SALT + source_id).encode("utf-8")).digest()
    return int.from_bytes(digest[: BUCKET_BITS // 8], "big")


def is_val_source(source_id: str, val_ratio: float) -> bool:
    """Val iff the source's bucket is below val_ratio of the range. The
    product is exact: multiplying a float by a power of two only shifts
    its exponent."""
    return source_bucket(source_id) < int(val_ratio * 2**BUCKET_BITS)


def _check_val_ratio(val_ratio: float) -> None:
    if not (math.isfinite(val_ratio) and 0 < val_ratio < 1):
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")


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
    val_ratio: float = DEFAULT_VAL_RATIO,
    require_holdout: bool = False,
) -> dict[str, str]:
    """Assign each sample to train/val/holdout, grouped by source_id.

    Returns {sample_id: split}. See the module docstring for the contract.
    """
    _check_val_ratio(val_ratio)
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

    # Every other source by its own hash alone, so a source keeps its split
    # whatever else the manifest holds.
    for src, flagged in flags.items():
        if flagged:
            continue
        part = "val" if is_val_source(src, val_ratio) else "train"
        for sid in by_source[src]:
            assignment[sid] = part

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
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help="each unflagged source's probability of landing in val (0 < r < 1)",
    )
    parser.add_argument(
        "--require-holdout",
        action="store_true",
        help="refuse a manifest with no sample flagged holdout",
    )
    args = parser.parse_args(argv)

    records = _read_jsonl(args.inp)
    assignment = split_records(
        records,
        val_ratio=args.val_ratio,
        require_holdout=args.require_holdout,
    )
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(dump_split(assignment))

    n = len(assignment)
    by_split = {s: sum(1 for v in assignment.values() if v == s) for s in SPLITS}
    print(
        f"split {n} samples (val_ratio={args.val_ratio}): "
        + ", ".join(f"{s}={by_split[s]}" for s in SPLITS),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
