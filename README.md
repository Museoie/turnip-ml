# turnip-ml

Machine learning for [Turnip](https://github.com/hoiekim/turnip-ios): trick-detection models (pose sequence → trick segments + names), training code, and evaluation.

## Conventions

- Large artifacts are never committed: no model binaries, datasets, or videos in git.
- Determinism: training and eval outputs are byte-identical on re-run (fixed seeds, no timestamps, no dict-order dependence).
