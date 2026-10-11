# Training

Trick-detection training pipeline for Turnip: pose keypoint sequences in,
trick segments (start/end frame) + trick names out. The iOS app consumes
what this repo validates; see the root README for repo-wide conventions.

## Layout

- `split_dataset.py` — deterministic stratified train/val/holdout splitter
  (turnip-ml#12). See its docstring for the full contract.
- `tests/` — unit tests (`python -m unittest discover -s Training/tests -t .`).

## Determinism contract

Every randomized step in this directory is seeded, and the seed is part of
the artifact: the same manifest + seed + Python interpreter version always
yields the byte-identical output. No timestamps, no unseeded RNG, no
dependence on input order. A training run that cannot be reproduced
byte-for-byte on the same interpreter is a bug.

## Splitting a manifest

```sh
python Training/split_dataset.py \
  --in manifest.jsonl --out split.json --seed 0 \
  --holdout-ids admin_holdout.txt
```

`manifest.jsonl` carries one JSON object per line with `sample_id` and
`user_id` (extra fields are ignored). `admin_holdout.txt` lists the
admin-curated holdout sample ids, one per line — those samples, and every
sibling sample from the same users, are pinned to holdout and never trained
on (poisoning defense). `split.json` maps each sample id to
`train` / `val` / `holdout` with sorted keys.
