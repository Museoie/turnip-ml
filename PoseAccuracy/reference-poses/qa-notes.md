# Reference-label QA notes

Reference labels generated with MediaPipe PoseLandmarker heavy (IMAGE mode,
10fps sampling) via `PoseAccuracy/rebuild_fixtures.py`, then visually QA'd
frame by frame via skeleton-overlay montages (every kept frame inspected).

## Clips

- `IMG_9229` (double cartfull), IMG_9229.mov — 61 labeled frames, candidate score 39.9938.
  Indoor gym double cartfull.
- `IMG_9420` (hook - scoot - gainer - cartfull combo), IMG_9420.mov — 69 labeled frames,
  candidate score 31.5320. Indoor gym combo: hook, scoot, gainer, cartfull.
- `IMG_9437` (corkscrew), IMG_9437.mov — 41 labeled frames, candidate score 42.2844.
  Indoor gym corkscrew off the trampoline.

## Method

- Auto-dropped: frames where MediaPipe found no pose.
- Auto-flagged for human review: mean joint visibility < 0.5, centroid jumps
  > 0.25 frame-width between adjacent samples (subject-switch detector).
  Flags never auto-drop; every flagged frame was viewed on the montage.
- Montages rendered for every kept frame of every clip; the skeleton tracks
  the tricking subject in all kept frames (no bystander switches observed).

## Drops

- `IMG_9229`: 1 (frame 12, no-detection)
- `IMG_9420`: 9 (frames 0, 6, 12, 18, 24, 30 — athlete not yet in frame at
  the clip start; frames 216, 318, 324 — no-detection)
- `IMG_9437`: 1 (frame 133, no-detection)

## Final counts

171 labeled frames / 182 sampled (11 dropped).

## Baseline

MoveNet Thunder int8
(movenet-thunder-int8/kaggle-singlepose-thunder-tflite-int8-1/sha256:b72fed22707cd6fb94b5a248b9bddb9c062b9f445471b4fa263407cf6d222011)
scores **37.9367** against these labels. Candidate extraction and scoring
are both byte-identical on re-run (verified); the scorer also passes the
synthetic validation.
