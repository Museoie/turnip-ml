# Smoke fixture

Tiny deterministic synthetic dataset for the `train.py` smoke test
(turnip-ml#13). Eight 32-frame @10Hz samples (MoveNet-17 keypoints, 0-1
coords) with trick segments. Motion signatures are deliberately separable —
the smoke run must reach 100% val accuracy; anything less is a
training-pipeline bug, not data noise.

- `manifest.jsonl` — one `{"sample_id", "user_id"}` per line (splitter input)
- `split.json` — `{sample_id: train|val|holdout}`, produced by the real
  `split_dataset.py` (turnip-ml#12), `--seed 0`
- `samples/*.json` — one sample per file (see `train.py` docstring for format)

Regenerate byte-identically with (`split_dataset.py` lands with the
still-open splitter PR, turnip-ml#17 — needed for the second step until it merges):

```sh
python Training/make_smoke_fixture.py --out Training/fixtures/smoke --seed 0
python Training/split_dataset.py \
  --in Training/fixtures/smoke/manifest.jsonl \
  --out Training/fixtures/smoke/split.json --seed 0
```

This fixture is synthetic test data (a few KB of JSON), not a training
dataset — no real footage, no PII, safe to commit.
