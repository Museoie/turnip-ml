# Pose-accuracy gate

Every PR that touches pose estimation (`Turnip/Pose/**`), the harness itself
(`PoseAccuracy/**`), or the gate workflow gets a **pose-accuracy score**: the
app's MoveNet Thunder model is run over three fixed fixture clips (Hoie's
tricking footage) and compared against hand-QA'd reference poses. The
deterministic 0–100 score must stay within `POSE_TOLERANCE` (0.5 pts) of
`baseline.json`, or the check fails.

## Files

| Path | Purpose |
|---|---|
| `baseline.json` | Locked baseline: score **37.9367**, model id + SHA, manifest SHA, per-clip breakdown |
| `fixture-manifest.json` | The three fixture clips (id, label, sha256, duration, fps, dimensions). Videos are **not** committed |
| `reference-poses/reference.json` | Hand-QA'd reference poses (171 frames, COCO-17, normalized xy + confidence) |
| `reference-poses/dropped-frames.json` | Frames excluded from scoring (no-detection / QA failures), per clip |
| `reference-poses/qa-notes.md` | How the labels were made and QA'd |
| `movenet_infer.py` | Candidate extraction: samples frames, runs MoveNet Thunder int8, writes poses |
| `sample_frames.py` | Frame sampling shared by the harness (10 samples/s, every 3rd frame at 30fps) |
| `scorer.py` | Deterministic 0–100 scorer (validated on synthetic perturbations) |
| `requirements.txt` | Pinned Python deps (determinism: dep drift must not move the score) |
| `ci/fetch_fixture.py` | Downloads fixture clips from `POSE_FIXTURE_URLS` (or R2), verifies sha256 |
| `ci/check_gate.py` | Compares a score against the baseline, enforces the tolerance |

## Running locally

```bash
pip install -r PoseAccuracy/requirements.txt
# 1. fetch fixtures (needs POSE_FIXTURE_URLS: one URL per clip, manifest order)
python PoseAccuracy/ci/fetch_fixture.py PoseAccuracy/fixture-manifest.json /tmp/fixture
# 2. extract candidate poses with the app's model
python PoseAccuracy/movenet_infer.py \
  --manifest PoseAccuracy/fixture-manifest.json \
  --frames-dir /tmp/fixture \
  --drop-frames PoseAccuracy/reference-poses/dropped-frames.json \
  --out /tmp/candidate.json
# 3. score and check the gate
python PoseAccuracy/scorer.py \
  --reference PoseAccuracy/reference-poses/reference.json \
  --candidate /tmp/candidate.json \
  --model-id "movenet-thunder-int8/kaggle-singlepose-thunder-tflite-int8-1/sha256:b72fed22707cd6fb94b5a248b9bddb9c062b9f445471b4fa263407cf6d222011" \
  --manifest-sha "$(python -c "import hashlib; print(hashlib.sha256(open('PoseAccuracy/fixture-manifest.json','rb').read()).hexdigest())")" \
  --out /tmp/result.json
python PoseAccuracy/ci/check_gate.py \
  --result /tmp/result.json \
  --baseline PoseAccuracy/baseline.json \
  --tolerance 0.5 --override false
```

`movenet_infer.py` verifies the downloaded model against the expected
sha256/size and refuses to run on a mismatch.

## The metric

For each joint, displacement vs the reference is normalized by the
reference's shoulder-to-hip torso scale; joint quality is
`max(0, 1 - normalized_displacement / 0.6)`. The frame score blends 85%
displacement quality with 15% confidence agreement (both estimators confident
or both not). Scores are averaged over joints, then frames, then clips
(equal weight per clip). Four-decimal, sorted-key JSON, no timestamps —
byte-identical on re-run.

The reference defines the task: **pose of the tricking subject**. If the
candidate tracks a bystander instead, that counts against it — that is
honest model behavior, verified by visual QA, not a fixture bug.

## Override policy

A score regression beyond tolerance is a real failing check. If the
regression is accepted (e.g. a deliberate model change), a maintainer adds
the `pose-accuracy-override` label to the PR and re-runs the job: the gate
turns green and logs who added the label and when. The label never merges
code by itself.

## Fixture storage

The fixture videos are Hoie's personal iPhone footage and are **not**
committed to this repo. They live in his Cloudflare R2 bucket
(`turnip-storage`, prefix `turnip-ios-ci-fixtures/`) behind a public
r2.dev URL. CI downloads each clip over plain HTTPS from
`POSE_FIXTURE_URLS` — a repo **variable** (one URL per clip, in manifest
order) — and sha256-verifies every download against the manifest. Variables,
unlike secrets, are visible to fork-PR workflows, so the gate works on PRs
from forks with no credentials. `ci/fetch_fixture.py` also supports a
private-R2 source (`POSE_FIXTURE_R2_BUCKET` + `R2_ENDPOINT` /
`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` secrets) for setups that
don't want public fixture URLs. If neither source is configured, the job
fails loudly — it never silently passes.

To rebuild the fixture set (add/remove clips), run
`PoseAccuracy/rebuild_fixtures.py` — see `PoseAccuracy/FIXTURES.md`.
