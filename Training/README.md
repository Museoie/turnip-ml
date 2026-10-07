# Training

Trick-detection training pipeline for Turnip: pose keypoint sequences in,
trick segments (start/end frame) + trick names out. The iOS app consumes
what this repo validates; see the root README for repo-wide conventions.

## Layout

- `split_dataset.py` — deterministic stratified train/val/holdout splitter.
  See its docstring for the full contract. (Lands with the still-open
  splitter PR, turnip-ml#17 — the splitter steps below need that PR until it merges.)
- `train.py` — champion training / fine-tuning script (turnip-ml#13).
  Hermetic TOML config, deterministic, CI smoke-tested.
- `make_smoke_fixture.py` — generates the tiny synthetic smoke fixture.
- `config/smoke.toml` — smoke-test hyperparameters (used in CI).
- `config/full.toml` — full-run template (droplet / GPU worker).
- `fixtures/smoke/` — tiny synthetic fixture (see its README).
- `requirements.txt` — pinned training dependencies.
- `tests/` — unit tests (`python -m unittest discover -s Training/tests -t .`).

## Determinism contract

Every randomized step in this directory is seeded, and the seed is part of
the artifact: the same manifest + seed always yields the byte-identical
output. No timestamps, no unseeded RNG, no dependence on input order.
A training run that cannot be reproduced byte-for-byte is a bug.

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

## Training the champion

```sh
pip install -r Training/requirements.txt
python Training/train.py --config Training/config/smoke.toml
```

`train.py` implements turnip-ml#13:

- **Hermetic config.** Every hyperparameter lives in the TOML file —
  unknown or missing keys fail closed, so a typo'd hyperparameter can never
  silently fall back to a default. `--out` optionally overrides `out_dir`.
- **Split discipline.** Trains on the `train` split only; `val` is for
  evaluation; `holdout` samples are never loaded for training (the
  poisoning defense from #12 is enforced at load time, not just by
  convention). A val label never seen in train fails closed.
- **Windowing.** Samples are cut into overlapping windows
  (`window_frames` / `window_stride`); each window is labeled by the trick
  covering at least half of it, else `background`. A segment carrying N
  names labels the window `"a+b"` (sorted) — `is_combo` is derived from the
  `"+"`, per the model output contract (master plan): source-frame
  coordinates, one segment with N names.
- **Model (smoke).** Softmax linear classifier over per-joint mean position
  + mean speed, trained with full-batch gradient descent (seeded). The
  architecture is deliberately tiny: the smoke test proves the pipeline
  end-to-end, not the model. `Training/README.md` "Full run" covers the
  real architecture seam.
- **Fine-tuning.** Set `init_checkpoint` to a previous run's `model.npz`
  to continue training the champion instead of starting from scratch;
  class list and window params must match or it fails closed.
- **Artifacts.** The out dir gets `config.toml` (byte-exact snapshot of the
  effective config), `model.npz` (weights, class list, feature stats,
  window params), and `metrics.json` (sorted keys: seed, losses, val
  accuracy overall + per class, decoded val segments, config sha256).
  Re-running the same config + data yields byte-identical `metrics.json`.

The CI smoke test (`.github/workflows/training-smoke.yml`) runs the script
on the synthetic fixture and the unit tests. The fixture is separable by
construction, so the smoke run must reach 100% val accuracy — anything less
fails the run.

## Full run

Real training runs on the droplet during MVP; the GPU worker comes later.
The seam is one variable — `compute` in the config:

```toml
compute = "cpu"   # droplet / smoke
compute = "gpu"   # GPU worker (cupy-backed); fails closed without it
```

`compute` is read in exactly one place (`train.py::_select_backend`).

1. Export labeled samples from turnip-farm (`GET /api/labels/export`),
   one JSON sample per file in `data/` (git-ignored; see `train.py`
   docstring for the sample format — JSON here, TKP1 binary in the real
   pipeline).
2. Copy `Training/config/full.toml` to `Training/config/full.local.toml`
   (git-ignored scratch) and point `split_path` / `samples_dir` at the
   export; choose hyperparams there, not on the CLI.
3. `python Training/split_dataset.py --in data/manifest.jsonl
   --out data/split.json --seed 0`
4. `python Training/train.py --config Training/config/full.local.toml`
5. To fine-tune the current champion instead of training from scratch, set
   `init_checkpoint = "runs/<champion>/model.npz"`.
6. `metrics.json` feeds the promotion gate (turnip-ml#15: champion /
   challenger promotion on >=1% val improvement).

Swapping the smoke model for the real architecture (temporal conv net over
the keypoint sequence) is a `build_dataset`/`train_classifier`-level change
behind the same config + artifact contract; the smoke model stays as the
fast CI path.
