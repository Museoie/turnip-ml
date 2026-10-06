# Model Manifest + On-Device Inference Contract (`turnip-ml` → `turnip-ios`)

**Scope.** This document is the contract between the training pipeline
and the iOS client: the model manifest format, the publishing flow, and
exactly what the client must do to run the trick-detection model
on-device. It implements the "Export + deploy" step of master plan §6
(`https://github.com/hoiekim/turnip-farm/blob/main/docs/MASTER_PLAN.md`)
and the OTA flow of §5.4. Companion docs: the farm's
`DATABASE_DESIGN.md` (`models` table, `POST /api/models`,
`GET /api/models/current`), `TKP1.md` (pose wire format), and
`TRAINING_DESIGN.md` (how champions are produced).

## 1. Manifest format

Served by `GET /api/models/current` and stored on the farm's `models`
row. Exact fields:

```jsonc
{
  "version": "trick-v1.3",      // trick-v<major>.<minor> (TRAINING_DESIGN.md §9)
  "model_type": "trick-detection",
  "taxonomy_version": 7,              // monotonic int; the vocab this model trained on
  "r2_key": "models/trick-v1.3.mlmodelc.zip",
  "sha256": "<hex of the zip bytes>",
  "byte_size": 1843200,
  "input": {
    "format": "TKP1",
    "format_version": 1,
    "sample_rate_hz": 10.0,           // canonical input rate the model expects
    "keypoint_count": 17,
    "features": ["x", "y", "confidence", "gap_indicator"]
  },
  "output": {
    "coordinates": "source-frame",    // absolute indices into the canonical 10 Hz sequence
    "fields": ["start_frame", "end_frame", "score", "trick_names", "is_combo"]
  },
  "vocabulary": [                     // full canonical vocab at taxonomy_version
    {"canonical": "cork", "aliases": ["cork 720", "corkscrew"]},
    {"canonical": "gainer", "aliases": ["moon kick"]},
    "..."
  ],
  "val_metrics": {
    "segment_f1": 0.81,
    "detection_rate_iou05": 0.88,
    "name_accuracy_given_detection": 0.93
  },
  "holdout_report_r2_key": "models/trick-v1.3.holdout_report.json",
  "min_client_version": "0.2.0",      // clients below this skip the update
  "promoted_at": "2026-10-07T07:12:00Z",
  "training_run_id": "nightly-2026-10-07"
}
```

Notes:

- `vocabulary` is the full canonical vocabulary at `taxonomy_version`,
  with aliases for display and search. The client MUST use this embedded
  vocabulary for the active model — never mix it with a cached
  vocabulary from another version (§5).
- `score` is per-segment confidence in [0, 1]; the client thresholds it
  (§3.3).
- `promoted_at` is RFC 3339 UTC. All sizes are bytes; the checksum is
  over the exact zip bytes the client downloads.

## 2. Publishing flow

1. The nightly pipeline promotes a challenger (TRAINING_DESIGN.md §8).
2. Export: `coremltools` → Core ML `.mlmodelc`, zipped; compute
   `sha256` + `byte_size` over the zip.
3. Upload the artifact and the holdout report to R2
   (`models/trick-v<...>.mlmodelc.zip`).
4. `POST /api/models` (admin-scoped): creates the `models` row
   (version, `model_type`, `taxonomy_version`, `r2_key`, `val_metrics`
   JSONB, `promoted_at`) and flips `current` to the new row. Old rows
   are retained for rollback.
5. `GET /api/models/current` serves the manifest above: version, URL,
   checksum, `taxonomy_version`, and the vocabulary list.

Endpoint naming note: an earlier draft of this contract referenced
`GET /api/models/latest`; the merged master plan (§4) names the
endpoint `GET /api/models/current`. This document follows the master
plan.

## 3. On-device inference contract

### 3.1 Input

- The client feeds the model a canonical **10 Hz TKP1 frame sequence**:
  `T × 17 × 3` float32 `(x, y, confidence)`, x/y normalized 0–1 in
  source-frame coordinates, gaps as all-zero-confidence frames.
- If the client's analysis rate differs from 10 Hz, the client resamples
  to 10 Hz first, using the master-plan rules (linear interpolation of
  x/y/confidence; **gaps break interpolation** — never synthesize pose
  across missing data).
- **Judgment call:** the client appends the gap-indicator channel
  (1.0 on gap frames, 0.0 otherwise) to build the `T × 52` model input
  itself, rather than the exported model deriving it. Rationale: the
  "all confidences are zero" test is trivial client-side and keeps the
  Core ML graph purely numeric.

### 3.2 Output

Per inference call, the model returns segments in **source-frame
coordinates**:

```jsonc
{"start_frame": 120, "end_frame": 175, "score": 0.91,
 "trick_names": ["cork"], "is_combo": false}
```

- `start_frame` / `end_frame` are absolute indices into the canonical
  10 Hz sequence the client fed in. The client maps them back to its own
  timeline (video timestamps) via its resampling map.
- `is_combo` is always `trick_names.length > 1`, kept explicit for
  client convenience (display, routing).
- Combo MVP: one segment with N names — no sub-segmentation of
  individual tricks.

Worked example: a 60 s source is 600 canonical frames. A returned
segment `start_frame=120, end_frame=175` spans seconds 12.0–17.5 of the
canonical timeline; the client converts to its own frame/timestamp
space from there.

### 3.3 Client postprocessing (normative)

1. **Threshold:** drop segments with `score < 0.5`. (Judgment call: a
   fixed default for now; a future manifest `client_config` may
   override it per model version.)
2. **NMS:** class-agnostic non-maximum suppression — for overlapping
   segments with temporal IoU > 0.5, keep the higher `score`.
   (Judgment call: single class-agnostic pass; per-name NMS only if
   false-positive analysis demands it.)
3. **Merge:** merge segments with identical `trick_names` separated by
   fewer than 5 frames (0.5 s) into one segment spanning the union.
   (Judgment call: covers flicker at trick boundaries.)
4. The surviving segments feed the clip editor as proposals — windows
   plus names — and the user confirms or edits them per the confirmed
   contribution flow. Proposals are never contributed without explicit
   user confirmation.

### 3.4 Preprocessing checklist (client)

- Resample pose to canonical 10 Hz (gap-safe, §3.1).
- Do **not** re-normalize x/y per sequence — values are already 0–1 in
  source-frame coordinates, and absolute position in frame matters for
  tricks.
- Append the gap-indicator channel → `T × 52` float32.
- Inference is fully on-device; no data leaves the device for
  inference. The manifest/vocabulary download (§4) is the only network
  touch in this contract.

## 4. OTA update flow

- The client polls `GET /api/models/current` on app launch and on
  foreground. (Per the iOS side, this polling is already wired into
  launch/foreground — the contract it speaks is the §1 manifest.)
- Update iff `version` differs from the active model AND the
  client version ≥ `min_client_version`.
- Download the artifact (R2 URL or presigned GET from the manifest)
  streamed to a temp file; verify `sha256` (and `byte_size`) **before**
  touching the active model.
- **Atomic swap:** verify → move into place → activate on the next
  inference call. Never half-replace the active model.
- **Failure fallback:** any failure — network error, checksum mismatch,
  Core ML load error — keeps the previous model active and retries on
  the next poll. The client retains the previous artifact until the new
  one loads successfully; on load failure the temp file is discarded.
  (Judgment call: load-success is the acceptance check; no on-device
  accuracy smoke test for v1.)
- **First run / never downloaded:** fall back to the **bundled model**
  shipped in the app bundle (a frozen champion checked in at release
  time), or — if none is bundled — the heuristic detector. The app must
  never be left without a proposal path.
- Provenance: the client includes the active `version` in
  contribution uploads, so future debugging can tell which model
  proposed the clips the user confirmed.

## 5. Compatibility

- **Taxonomy mismatch:** the client always uses the `vocabulary`
  embedded in the active model's manifest. A new model with a new
  `taxonomy_version` carries its own names, so an old client displays
  them correctly with no app update. Never merge vocabularies across
  versions.
- **Model absent:** heuristic detector fallback (master plan §5.4) —
  proposals are window-only and names come from the user.
- **`min_client_version`:** clients below it skip the update silently
  (logged locally). Breaking input-format changes bump
  `min_client_version` and the model major version together.
- **Rollback (server side):** re-pointing `GET /api/models/current` at
  the previous champion's row; clients pick it up on the next poll via
  the normal version-compare path. No client-side rollback protocol is
  needed beyond the atomic-swap failure fallback (§4).

## 6. Open questions

1. Score threshold (§3.3) — 0.5 default is a starting point; tune
   against precision/recall on the holdout set.
2. NMS/merge policy (§3.3) — class-agnostic NMS + 5-frame merge is the
   MVP; revisit with real model behavior.
3. Gap-indicator placement (§3.1) — client-side vs baked into the
   exported graph.
4. On-device acceptance (§4) — whether a forward-pass smoke test over a
   cached pose snippet is worth the complexity for v1.
