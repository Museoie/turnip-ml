# turnip-ml

Machine learning for [Turnip](https://github.com/hoiekim/turnip-ios): pose-estimation models, evaluation harnesses, and the tooling around them. The iOS app consumes what this repo validates.

## Layout

- `PoseAccuracy/` — pose-accuracy fixture harness and CI gate: pinned MoveNet Thunder baseline over QA'd labeled frames from tricking footage. See `PoseAccuracy/FIXTURES.md` for the rebuild workflow.
- `Training/` — trick-detection training pipeline utilities: deterministic dataset splitting, and (as the pipeline grows) fine-tuning, evaluation, and the champion/challenger promotion gate. See `Training/README.md`.

## Conventions

- Large artifacts are never committed: no model binaries, datasets, or videos in git. Fixture videos live in Cloudflare R2 behind public URLs and are sha256-verified on every download.
- Evaluation baselines are locked JSON (`baseline.json`). A regression fails CI; accepted regressions are overridden explicitly via the `pose-accuracy-override` label, never silently.
- Training pipeline steps are deterministic: the same inputs + seed always yield the byte-identical output (no timestamps, no unseeded RNG, no input-order dependence). A run that cannot be reproduced byte-for-byte is a bug.
