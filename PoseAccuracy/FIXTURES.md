# Fixture management

The pose-accuracy gate scores the app's MoveNet model against hand-QA'd
reference poses on a fixed set of fixture clips. The fixture **videos are
Hoie's personal footage and are never committed to git**; everything else
(manifest, reference poses, baseline) is committed under `PoseAccuracy/`.

The whole fixture set is rebuilt with `PoseAccuracy/rebuild_fixtures.py`.
There is no other supported way to change the set: adding or removing a
clip means re-running the rebuild (the baseline score is only meaningful
for the exact clip set + labels it was computed from).

## Prerequisites

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r PoseAccuracy/requirements.txt mediapipe==1.0.1
```

`requirements.txt` pins the deterministic CI deps (ai-edge-litert, numpy,
opencv). `mediapipe` is needed only for reference-label generation --
it never runs in CI, so it is not in `requirements.txt`.

If anything is missing, `rebuild_fixtures.py` says so at startup with the
exact install commands and exits non-zero -- it never dies with a raw
`ImportError` traceback.

The reference model (MediaPipe PoseLandmarker heavy, ~30MB) is downloaded
automatically on first run from the official MediaPipe model storage into
`~/.cache` (or pass `--reference-model PATH`). Its sha256 is recorded in
the QA report, so model drift between rebuilds is visible. The candidate
model (MoveNet Thunder int8) is downloaded from TFHub and
sha256/size-verified exactly like CI does (or pass `--candidate-model`).

## Adding a fixture

1. Get the video bytes. Either put the file on disk, or upload it to the
   R2 bucket (see "R2" below) -- the rebuild pulls byte-identical
   copies and sha256-verifies them.
2. Run phase 1 (reference labels + QA artifacts, no repo files written):

```bash
python PoseAccuracy/rebuild_fixtures.py \
  --local-dir /path/to/videos \
  --clips IMG_9229:"double cartfull" IMG_9420:"hook - scoot - gainer - cartfull (combo)" IMG_9437:corkscrew \
  --sha256 IMG_9229:0556... IMG_9420:8fc5... IMG_9437:fecd... \
  --work-dir /tmp/rebuild-work
```

`--clips` entries are `ID:LABEL[:FILENAME]` (filename required in R2
mode; in local mode it defaults to the single file starting with `ID.`).
`--sha256` is optional but recommended: downloads are verified against
known-good hashes *before* any label is built on them.

3. Review the QA. Phase 1 writes, per clip, a skeleton-overlay montage
   (`montage_<ID>.png` under `--work-dir`) plus `qa-report.json` with
   drop reasons and auto-flags (low visibility, centroid jumps that
   suggest a subject switch). **Look at every tile**: the skeleton must
   track the tricking subject in every frame. Flags never auto-drop --
   they are there to tell you where to look.
4. Re-run with `--qa-approved`, adding `--drop ID:FRAME` for any frames
   the montage review rejects:

```bash
python PoseAccuracy/rebuild_fixtures.py ... --qa-approved --drop IMG_9420:216
```

5. Phase 2 runs the MoveNet candidate, scores it, and writes the final
   files: `fixture-manifest.json`, `reference-poses/reference.json`,
   `reference-poses/dropped-frames.json`, `baseline.json`, and a
   `reference-poses/qa-notes.md` draft. Append your human QA
   observations to the notes, then commit exactly what the script's
   `git add` list prints (never `--work-dir` scratch).

## Removing a fixture

Rebuild the set without that clip: run the same two phases with the clip
omitted from `--clips`. Everything (manifest, labels, baseline) is
regenerated consistently; do not hand-edit the manifest or baseline.

## Naming / label conventions

- Clip id: the filename stem, uppercase (`IMG_9229`). One id per file.
- Label: the trick, short and lowercase (`double cartfull`,
  `hook - scoot - gainer - cartfull (combo)`, `corkscrew`). No colons (the
  `--clips` syntax uses `:` as a separator).
- Keep the original filename and container in the manifest (`file`).
- Manifest clips are sorted by id.

## Determinism requirements

The gate compares scores across runs, so the rebuild must be
deterministic: sorted JSON keys, 4-decimal rounding on every float, no
timestamps in any output. The script enforces this; do not post-process
its outputs.

Reference inference defaults to MediaPipe **IMAGE mode** (one independent
detection per frame). This is deliberate: VIDEO-mode tracking once froze
on a wrong pose for ~180 frames of a clip with no error signal, and only
montage review caught it. IMAGE mode cannot carry tracker state between
frames, so that failure mode is impossible. (`--reference-mode video`
exists but is not recommended.)

Labels are generated once, QA'd, and committed. CI never regenerates
them -- it only re-runs the candidate + scorer, which are verified
byte-identical.

## R2

Fixture videos live in Hoie's Cloudflare R2 bucket, served through a
public r2.dev URL. In R2 mode the
script downloads each clip with the Cloudflare **v4 REST Get Object**
endpoint (`GET /accounts/{account}/r2/buckets/{bucket}/objects/{key}`,
stdlib `urllib`, no extra dependencies) using a token from `--r2-token`
or `CLOUDFLARE_R2_TOKEN`:

```bash
python PoseAccuracy/rebuild_fixtures.py \
  --r2-account <account-id> --r2-bucket turnip-storage \
  --r2-prefix turnip-ios-ci-fixtures/ \
  --clips IMG_9229:"double cartfull":IMG_9229.mov ... \
  --work-dir /tmp/rebuild-work
```

CI does not use R2 mode at all: it downloads the clips over plain HTTPS
from the public r2.dev URLs, configured as the `POSE_FIXTURE_URLS` repo
**variable** (Variables, unlike Secrets, are visible to fork-PR workflows,
so the gate works on PRs from forks with no credentials). Every download is
sha256-verified against the manifest, so the public URLs are safe.

### Permission gotcha (hit 2026-09-14)

Creating the R2 token with **"Workers R2 Storage Bucket Item Read"**
scoped to the bucket is *not enough* for this endpoint: that permission
is S3-API-only (the `https://<account>.r2.cloudflarestorage.com` SigV4
path, e.g. the `aws` CLI in CI). The v4 REST objects API needs
**account-scoped "Workers R2 Storage: Read"**. With the wrong permission
you get a `403` that looks exactly like a bad token. The script's R2
error message calls this out.
