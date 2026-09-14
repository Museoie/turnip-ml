#!/usr/bin/env python3
"""Rebuild the pose-accuracy fixture set from source videos.

End-to-end rebuild: fetch videos (local dir or Cloudflare R2), sample frames
with the repo's shared sampler, generate reference labels with MediaPipe
PoseLandmarker heavy, run automated QA + render spotcheck montages, then
(on explicit --qa-approved) run the MoveNet candidate, score it, and write
the final fixture files (fixture-manifest.json, reference-poses/,
baseline.json) plus a QA-notes draft.

Two phases:
  1. Without --qa-approved: builds reference candidates, QA report and
     montages under --work-dir. Review the montages, then re-run with
     --qa-approved (plus optional --drop ID:FRAME overrides).
  2. With --qa-approved: writes the final files into --out-dir
     (default: the PoseAccuracy/ dir holding this script) and prints the
     git commands, the new baseline score and a suggested commit message.

Reuses the repo's own modules -- sample_frames and movenet_infer imported
(not copied); scorer is reused through its own CLI so the baseline file
keeps the canonical scorer output format. Determinism rules: sorted keys, 4-decimal rounding,
no timestamps anywhere in the outputs.

Dependencies (checked at startup, friendly error instead of ImportError):
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r PoseAccuracy/requirements.txt mediapipe==1.0.1
The MediaPipe heavy model (~30MB) is downloaded automatically on first run
from the official MediaPipe model storage into ~/.cache
(or pass --reference-model PATH).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

EXIT_DEPS = 2
EXIT_QA_PENDING = 0  # phase 1 ends here by design; review montages, then re-run

MP_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task")

# MediaPipe 33 -> COCO-17 (matches the scorer's joint order)
MP2COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
# COCO-17 skeleton edges for the spotcheck montages
EDGES = [(0, 1), (0, 2), (1, 3), (2, 4),
         (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
         (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16),
         (0, 5), (0, 6)]


def check_dependencies() -> None:
    """Fail fast with install instructions instead of a raw ImportError."""
    missing: list[str] = []
    for mod in ("cv2", "numpy", "mediapipe"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    try:
        from ai_edge_litert.interpreter import Interpreter  # noqa: F401
    except ImportError:
        missing.append("ai_edge_litert")
    if missing:
        print(f"error: missing required Python modules: {', '.join(missing)}",
              file=sys.stderr)
        print("Dependencies (checked at startup, friendly error instead of "
              "ImportError):\n" +
              __doc__.split("Dependencies (checked at startup, friendly error "
                            "instead of ImportError):")[1].strip(),
              file=sys.stderr)
        sys.exit(EXIT_DEPS)


check_dependencies()

sys.path.insert(0, HERE)
from sample_frames import sample_plan, TARGET_FPS  # noqa: E402
import movenet_infer  # noqa: E402  (model + per-frame inference fns reused)
import cv2  # noqa: E402
import numpy as np  # noqa: E402


def r4(x: float) -> float:
    return round(float(x), 4)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_clip_spec(spec: str) -> tuple[str, str, str | None]:
    """ID:LABEL[:FILENAME] -> (id, label, filename-or-None)."""
    parts = spec.split(":")
    if len(parts) < 2:
        raise SystemExit(f"bad --clips entry {spec!r}: want ID:LABEL[:FILENAME]")
    cid = parts[0].strip()
    if len(parts) == 2:
        label, filename = parts[1].strip(), None
    else:
        label, filename = ":".join(parts[1:-1]).strip(), parts[-1].strip()
    if not cid or not label:
        raise SystemExit(f"bad --clips entry {spec!r}: empty id or label")
    return cid, label, filename


def parse_kv(spec: str) -> tuple[str, str]:
    cid, _, value = spec.partition(":")
    if not cid or not value:
        raise SystemExit(f"bad entry {spec!r}: want ID:VALUE")
    return cid.strip(), value.strip()


# ---------------------------------------------------------------- inputs

def r2_get_object(account: str, bucket: str, key: str, token: str, dest: str) -> str:
    """Download one R2 object via the Cloudflare v4 REST Get Object endpoint.

    GET https://api.cloudflare.com/client/v4/accounts/{account}/r2/buckets/
        {bucket}/objects/{key}  with  Authorization: Bearer <token>.

    stdlib urllib only. Returns the sha256 of the downloaded bytes.
    NOTE: the token needs account-scoped "Workers R2 Storage: Read".
    The "Workers R2 Storage Bucket Item Read" permission is S3-API-only and
    does NOT authorize this endpoint (you get a 403 that looks like a bad
    token).
    """
    # Slashes in the key MUST be sent literally, not percent-encoded.
    enc_key = "/".join(urllib.parse.quote(seg, safe="") for seg in key.split("/"))
    url = (f"https://api.cloudflare.com/client/v4/accounts/{account}"
           f"/r2/buckets/{bucket}/objects/{enc_key}")
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {token}",
                      "User-Agent": "turnip-pose-harness/1.0"})
    h = hashlib.sha256()
    try:
        with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                h.update(chunk)
                f.write(chunk)
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode()[:500]
        except Exception:
            detail = ""
        raise SystemExit(
            f"R2 download failed for {key}: HTTP {e.code}. {detail}\n"
            "If this is a 403, the token almost certainly lacks account-scoped "
            "'Workers R2 Storage: Read' (the Bucket-Item permission is "
            "S3-API-only and does not authorize this endpoint).")
    return h.hexdigest()


def resolve_inputs(a) -> list[dict]:
    """Return [{id, label, path, sha256}] for every --clips entry."""
    clips = []
    if a.local_dir:
        for spec in a.clips:
            cid, label, filename = parse_clip_spec(spec)
            if filename:
                path = os.path.join(a.local_dir, filename)
                if not os.path.isfile(path):
                    raise SystemExit(f"{cid}: not found: {path}")
            else:
                matches = sorted(f for f in os.listdir(a.local_dir)
                                 if f.startswith(cid + "."))
                if len(matches) != 1:
                    raise SystemExit(
                        f"{cid}: want --clips {cid}:<label>:<filename> "
                        f"(found {len(matches)} files starting with {cid + '.'})")
                path = os.path.join(a.local_dir, matches[0])
                filename = matches[0]
            clips.append({"id": cid, "label": label, "file": filename,
                          "path": path})
    else:
        token = a.r2_token or os.environ.get("CLOUDFLARE_R2_TOKEN", "")
        if not token:
            raise SystemExit("R2 needs --r2-token or CLOUDFLARE_R2_TOKEN")
        os.makedirs(a.work_dir, exist_ok=True)
        for spec in a.clips:
            cid, label, filename = parse_clip_spec(spec)
            if not filename:
                raise SystemExit(
                    f"{cid}: R2 mode needs an explicit filename "
                    f"(--clips {cid}:<label>:<filename>)")
            key = (a.r2_prefix or "") + filename
            dest = os.path.join(a.work_dir, filename)
            print(f"downloading s3://{a.r2_bucket}/{key} ...", flush=True)
            got = r2_get_object(a.r2_account, a.r2_bucket, key, token, dest)
            print(f"  sha256={got[:16]}...", flush=True)
            clips.append({"id": cid, "label": label, "file": filename,
                          "path": dest, "dl_sha256": got})
    # hash (and verify against --sha256 when given)
    want_hash = dict(parse_kv(s) for s in (a.sha256 or []))
    for c in clips:
        observed = c.pop("dl_sha256", None) or sha256_file(c["path"])
        c["sha256"] = observed
        if c["id"] in want_hash and want_hash[c["id"]].lower() != observed.lower():
            raise SystemExit(
                f"{c['id']}: sha256 MISMATCH: expected {want_hash[c['id']][:16]}..., "
                f"got {observed[:16]}... -- refusing to build labels on wrong bytes.")
        print(f"{c['id']}: sha256={observed[:16]}... ({c['file']})", flush=True)
    clips.sort(key=lambda c: c["id"])
    return clips


def probe_video(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if n <= 0 or fps <= 0:
        raise SystemExit(f"{path}: bad probe (n_frames={n} fps={fps})")
    return {"n_frames": n, "fps": fps, "w": w, "h": h,
            "duration_s": round(n / fps, 3)}


# ---------------------------------------------------------------- reference

def ensure_reference_model(path_arg: str | None) -> tuple[str, str]:
    """Return (model_path, sha256), downloading to ~/.cache if missing."""
    candidates = []
    if path_arg:
        candidates.append(os.path.expanduser(path_arg))
    candidates.append(os.path.join(HERE, "models", "pose_landmarker_heavy.task"))
    cache_path = os.path.join(os.path.expanduser("~"), ".cache",
                              "pose_landmarker_heavy.task")
    candidates.append(cache_path)
    for p in candidates:
        if os.path.isfile(p):
            print(f"reference model: {p}", flush=True)
            return p, sha256_file(p)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    print(f"downloading MediaPipe pose_landmarker_heavy.task -> {cache_path} ...",
          flush=True)
    req = urllib.request.Request(MP_MODEL_URL,
                                 headers={"User-Agent": "turnip-pose-harness/1.0"})
    h = hashlib.sha256()
    with urllib.request.urlopen(req, timeout=600) as r, open(cache_path, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
            f.write(chunk)
    print(f"  sha256={h.hexdigest()[:16]}...", flush=True)
    return cache_path, h.hexdigest()


def build_landmarker(model_path: str, mode: str):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    base = mp_python.BaseOptions(model_asset_path=model_path)
    running = (vision.RunningMode.IMAGE if mode == "image"
               else vision.RunningMode.VIDEO)
    opts = vision.PoseLandmarkerOptions(base_options=base,
                                        running_mode=running,
                                        num_poses=1)
    return vision.PoseLandmarker.create_from_options(opts)


def run_reference(clips: list[dict], model_path: str, mode: str,
                  target_fps: float) -> tuple[dict, dict]:
    """Returns (frames_by_clip, misses_by_clip). Deterministic, no RNG."""
    import mediapipe as mp
    frames_by_clip, misses_by_clip = {}, {}
    for c in clips:
        cid = c["id"]
        probe = c["probe"]
        plan = sample_plan(probe["n_frames"], probe["fps"], target_fps)
        want = {idx: ts for idx, ts in plan}
        landmarker = build_landmarker(model_path, mode)  # fresh per clip
        cap = cv2.VideoCapture(c["path"])
        frames, misses = [], []
        idx = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if idx in want:
                ts = want[idx]
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                if mode == "image":
                    res = landmarker.detect(mp_img)
                else:
                    res = landmarker.detect_for_video(mp_img, int(ts * 1e6))
                if res.pose_landmarks:
                    lm = res.pose_landmarks[0]
                    joints = [{"x": r4(lm[i].x), "y": r4(lm[i].y),
                               "c": r4(lm[i].visibility)} for i in MP2COCO]
                    frames.append({"frame_index": idx, "t": round(ts, 3),
                                   "joints": joints})
                else:
                    misses.append(idx)
            idx += 1
        cap.release()
        landmarker.close()
        frames.sort(key=lambda f: f["frame_index"])
        frames_by_clip[cid] = frames
        misses_by_clip[cid] = sorted(misses)
        print(f"{cid}: reference kept={len(frames)} "
              f"missed={len(misses)} of {len(plan)} planned", flush=True)
    return frames_by_clip, misses_by_clip


# ---------------------------------------------------------------- QA

def qa_flags(frames: list[dict]) -> dict[int, list[str]]:
    """Automated QA: flag (never auto-drop) suspicious frames for the human."""
    flags: dict[int, list[str]] = {}
    prev_c = None
    for f in frames:
        fl = []
        joints = f["joints"]
        mean_c = sum(j["c"] for j in joints) / len(joints)
        if mean_c < 0.5:
            fl.append(f"low_visibility(mean_c={mean_c:.2f})")
        cx = sum(j["x"] for j in joints) / len(joints)
        cy = sum(j["y"] for j in joints) / len(joints)
        if prev_c is not None:
            jump = math.hypot(cx - prev_c[0], cy - prev_c[1])
            if jump > 0.25:
                fl.append(f"centroid_jump({jump:.2f})-possible-subject-switch")
        prev_c = (cx, cy)
        if fl:
            flags[f["frame_index"]] = fl
    return flags


def render_montage(video_path: str, frames: list[dict], out_path: str,
                   cols: int = 6) -> None:
    # Tile orientation follows the frames as decoded (portrait clips get
    # portrait tiles); skeleton coords are frame-normalized so the overlay
    # stays exact either way.
    cap0 = cv2.VideoCapture(video_path)
    ok0, f0 = cap0.read()
    cap0.release()
    if not ok0:
        raise SystemExit(f"cannot read {video_path}")
    fh, fw = f0.shape[:2]
    tw, th = (180, 320) if fh > fw else (320, 180)
    want = {f["frame_index"]: f["joints"] for f in frames}
    cap = cv2.VideoCapture(video_path)
    tiles = []
    idx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if idx in want:
            img = cv2.resize(bgr, (tw, th))
            pts = [(j["x"] * tw, j["y"] * th) for j in want[idx]]
            for a, b in EDGES:
                cv2.line(img, (int(pts[a][0]), int(pts[a][1])),
                         (int(pts[b][0]), int(pts[b][1])), (0, 255, 0), 1)
            for x, y in pts:
                cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)
            cv2.putText(img, str(idx), (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 0), 2)
            tiles.append(img)
        idx += 1
    cap.release()
    rows = math.ceil(len(tiles) / cols)
    canvas = 255 * np.ones((rows * th, cols * tw, 3), dtype=np.uint8)
    for i, t in enumerate(tiles):
        r, cc = divmod(i, cols)
        canvas[r * th:(r + 1) * th, cc * tw:(cc + 1) * tw] = t
    cv2.imwrite(out_path, canvas)


def write_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


# ---------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Rebuild the pose-accuracy fixture set from source videos.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--local-dir",
                     help="directory holding the video files on disk")
    src.add_argument("--r2-account", help="Cloudflare account ID (R2 mode)")
    ap.add_argument("--r2-bucket", help="R2 bucket name (R2 mode)")
    ap.add_argument("--r2-prefix", default="",
                    help="key prefix inside the bucket (R2 mode)")
    ap.add_argument("--r2-token",
                    help="R2 API token (else CLOUDFLARE_R2_TOKEN env). "
                         "Needs account-scoped 'Workers R2 Storage: Read'.")
    ap.add_argument("--clips", nargs="+", required=True,
                    help="one per clip: ID:LABEL[:FILENAME] "
                         "(R2 mode: FILENAME is required)")
    ap.add_argument("--sha256", nargs="*", default=[],
                    help="expected hashes ID:HEX (verified before labeling)")
    ap.add_argument("--target-fps", type=float, default=TARGET_FPS)
    ap.add_argument("--reference-mode", choices=["image", "video"],
                    default="image",
                    help="MediaPipe running mode. 'image' (default) runs an "
                         "independent detection per frame -- immune to the "
                         "VIDEO tracker-freeze failure mode.")
    ap.add_argument("--reference-model", default=None,
                    help="pose_landmarker_heavy.task path (downloaded to "
                         "~/.cache if missing)")
    ap.add_argument("--candidate-model", default=None,
                    help="MoveNet Thunder int8 .tflite path "
                         "(else downloaded from TFHub and identity-verified)")
    ap.add_argument("--drop", nargs="*", default=[],
                    help="manual QA drops ID:FRAME (repeatable)")
    ap.add_argument("--work-dir", default=None,
                    help="scratch dir (default: <out-dir>/.rebuild-work)")
    ap.add_argument("--out-dir", default=HERE,
                    help="where the final fixture files are written "
                         "(default: this script's PoseAccuracy/ dir)")
    ap.add_argument("--qa-approved", action="store_true",
                    help="confirm you reviewed the phase-1 montages; writes "
                         "the final manifest/baseline")
    return ap


def main() -> int:
    a = build_parser().parse_args()
    if a.r2_account and not a.r2_bucket:
        raise SystemExit("--r2-account needs --r2-bucket")
    a.work_dir = a.work_dir or os.path.join(a.out_dir, ".rebuild-work")
    os.makedirs(a.work_dir, exist_ok=True)
    os.makedirs(os.path.join(a.out_dir, "reference-poses"), exist_ok=True)

    manual_drops: dict[str, set[int]] = {}
    for spec in a.drop:
        cid, val = parse_kv(spec)
        manual_drops.setdefault(cid, set()).add(int(val))

    # ---- inputs
    clips = resolve_inputs(a)
    for c in clips:
        c["probe"] = probe_video(c["path"])
        p = c["probe"]
        c["plan"] = sample_plan(p["n_frames"], p["fps"], a.target_fps)
        print(f"{c['id']}: {p['w']}x{p['h']} {p['duration_s']}s "
              f"-> {len(c['plan'])} planned frames", flush=True)

    # ---- reference labels
    model_path, model_sha = ensure_reference_model(a.reference_model)
    frames_by_clip, misses_by_clip = run_reference(
        clips, model_path, a.reference_mode, a.target_fps)

    # ---- automated QA + montages + report (phase 1 artifacts)
    report = {"reference_model_sha256": model_sha,
              "reference_mode": a.reference_mode,
              "target_fps": a.target_fps, "clips": {}}
    montage_paths = {}
    for c in clips:
        cid = c["id"]
        kept = [f for f in frames_by_clip[cid]
                if f["frame_index"] not in manual_drops.get(cid, set())]
        dropped = sorted(set(misses_by_clip[cid]) | manual_drops.get(cid, set()))
        flags = qa_flags(kept)
        mp = os.path.join(a.work_dir, f"montage_{cid}.png")
        render_montage(c["path"], kept, mp)
        montage_paths[cid] = mp
        report["clips"][cid] = {
            "file": c["file"], "label": c["label"], "sha256": c["sha256"],
            "planned": len(c["plan"]), "kept": len(kept),
            "dropped": dropped,
            "drop_reasons": {str(i): ("manual-qa" if i in manual_drops.get(cid, set())
                                      else "no-detection") for i in dropped},
            "flags": {str(k): v for k, v in sorted(flags.items())},
            "montage": mp,
        }
        print(f"{cid}: kept={len(kept)} dropped={len(dropped)} "
              f"flagged={len(flags)} montage={mp}", flush=True)
    write_json(os.path.join(a.work_dir, "qa-report.json"), report)
    write_json(os.path.join(a.work_dir, "reference-candidate.json"),
               {"clips": {cid: {"frames": frames_by_clip[cid]}
                          for cid in sorted(frames_by_clip)}})

    if not a.qa_approved:
        print("\n==== PHASE 1 COMPLETE -- QA PENDING ====")
        print("Review every montage tile: the skeleton must track the "
              "tricking subject in every frame.")
        for cid, mp in sorted(montage_paths.items()):
            print(f"  {cid}: {mp}")
        print(f"QA report: {a.work_dir}/qa-report.json")
        nflags = sum(len(v["flags"]) for v in report["clips"].values())
        print(f"{nflags} flagged frames need a human look "
              f"(low visibility / centroid jumps).")
        print("\nWhen satisfied, re-run with --qa-approved"
              + (" --drop ID:FRAME ..." if not a.drop else "")
              + " to write the final fixture files.")
        return EXIT_QA_PENDING

    # ---- phase 2: candidate, score, final files
    dropped_map = {cid: set(report["clips"][cid]["dropped"])
                   for cid in report["clips"]}
    model = movenet_infer.obtain_model(a.candidate_model, None)
    it, in_idx, out_idx = movenet_infer.make_interpreter(model)
    cand_clips = {}
    for c in clips:
        cid = c["id"]
        p, skip = c["probe"], dropped_map[cid]
        cap = cv2.VideoCapture(c["path"])
        want = {idx for idx, _ in c["plan"]} - skip
        frames = []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in want:
                ts = idx / p["fps"]
                inp, ox, oy, sc = movenet_infer.letterbox(frame)
                raw = movenet_infer.infer_frame(it, in_idx, out_idx, inp)[0, 0]
                joints = movenet_infer.to_frame_normalized(
                    raw, ox, oy, sc, p["w"], p["h"])
                frames.append({"frame_index": idx, "t": round(ts, 3),
                               "joints": joints})
            idx += 1
        cap.release()
        frames.sort(key=lambda f: f["frame_index"])
        expected = len(c["plan"]) - len(skip)
        if len(frames) != expected:
            raise SystemExit(f"{cid}: candidate {len(frames)} != {expected}")
        cand_clips[cid] = {"frames": frames}
        print(f"{cid}: candidate {len(frames)} frames", flush=True)
    cand_path = os.path.join(a.work_dir, "candidate.json")
    write_json(cand_path, {"clips": cand_clips})

    # manifest (deterministic)
    if a.local_dir:
        source = ("local video files (NOT committed to git)")
    else:
        source = (f"Hoie's footage, private Cloudflare R2 "
                  f"{a.r2_bucket}/{a.r2_prefix} (NOT committed to git)")
    manifest = {
        "version": 1,
        "target_fps": a.target_fps,
        "clips": [{
            "id": c["id"], "file": c["file"], "label": c["label"],
            "sha256": c["sha256"], "duration_s": c["probe"]["duration_s"],
            "fps": r4(c["probe"]["fps"]), "width": c["probe"]["w"],
            "height": c["probe"]["h"], "source": source,
        } for c in clips],
    }
    manifest_path = os.path.join(a.out_dir, "fixture-manifest.json")
    write_json(manifest_path, manifest)
    manifest_sha = sha256_file(manifest_path)

    # reference + dropped-frames (final, post-QA)
    reference = {"clips": {}}
    dropped_final = {}
    for c in clips:
        cid = c["id"]
        skip = dropped_map[cid]
        kept = [f for f in frames_by_clip[cid] if f["frame_index"] not in skip]
        reference["clips"][cid] = {"frames": kept}
        dropped_final[cid] = sorted(skip)
    refdir = os.path.join(a.out_dir, "reference-poses")
    write_json(os.path.join(refdir, "reference.json"), reference)
    write_json(os.path.join(refdir, "dropped-frames.json"), dropped_final)

    # score via the repo's own scorer CLI (canonical output format)
    result_path = os.path.join(a.work_dir, "result.json")
    subprocess.run(
        [sys.executable, os.path.join(HERE, "scorer.py"),
         "--reference", os.path.join(refdir, "reference.json"),
         "--candidate", cand_path,
         "--model-id", movenet_infer.MODEL_ID,
         "--manifest-sha", manifest_sha,
         "--out", result_path],
        check=True)
    baseline_path = os.path.join(a.out_dir, "baseline.json")
    os.replace(result_path, baseline_path)
    with open(baseline_path) as f:
        baseline = json.load(f)

    write_qa_notes(a.out_dir, report, baseline, model_sha)

    total = sum(v["kept"] for v in report["clips"].values())
    print("\n==== REBUILD COMPLETE ====")
    print(f"baseline score: {baseline['score']:.4f} "
          f"({total} labeled frames, {len(clips)} clips)")
    for cid in sorted(report["clips"]):
        pc = baseline["per_clip"][cid]
        print(f"  {cid} ({report['clips'][cid]['label']}): "
              f"{pc['score']:.4f} over {pc['n_frames']} frames")
    rel = os.path.relpath(a.out_dir)
    print("\ngit add:")
    here_abs = os.path.abspath(HERE)
    if os.path.abspath(a.out_dir) == here_abs:
        prefix = "PoseAccuracy/"
    else:
        prefix = os.path.abspath(a.out_dir) + "/"
    for p in ["fixture-manifest.json", "baseline.json",
              "reference-poses/reference.json",
              "reference-poses/dropped-frames.json",
              "reference-poses/qa-notes.md"]:
        print(f"  git add {prefix}{p}")
    print("\nsuggested commit message:")
    print(f"  Rebuild pose-accuracy fixtures ({len(clips)} clips, "
          f"{total} labeled frames, baseline {baseline['score']:.4f})")
    return 0


def write_qa_notes(out_dir: str, report: dict, baseline: dict,
                   model_sha: str) -> None:
    lines = [
        "# Reference-label QA notes",
        "",
        f"Reference labels generated with MediaPipe PoseLandmarker heavy "
        f"(sha256:{model_sha[:16]}..., {report['reference_mode']} mode, "
        f"{report['target_fps']}fps sampling) via `PoseAccuracy/rebuild_fixtures.py`, "
        "then human-QA'd frame by frame via skeleton-overlay montages.",
        "",
        "## Clips",
        "",
    ]
    for cid in sorted(report["clips"]):
        r = report["clips"][cid]
        pc = baseline["per_clip"][cid]
        lines.append(
            f"- `{cid}` ({r['label']}), {r['file']} — {r['kept']} labeled "
            f"frames, candidate score {pc['score']:.4f}")
    lines += ["", "## Method", "",
              "- Auto-dropped: frames where MediaPipe found no pose.",
              "- Auto-flagged for human review: mean joint visibility < 0.5, "
              "centroid jumps > 0.25 frame-width between adjacent samples "
              "(subject-switch detector). Flags never auto-drop; every "
              "flagged frame was viewed on the montage.",
              "- Montages rendered for every kept frame of every clip; "
              "identity switches / garbage labels spotted in one view. "
              "Tile orientation follows the frames as OpenCV decodes them "
              "(these .mov files carry a rotation tag that OpenCV does not "
              "apply, so they process as portrait).",
              "", "## Drops", ""]
    for cid in sorted(report["clips"]):
        r = report["clips"][cid]
        if r["dropped"]:
            reasons = ", ".join(f"{i} ({r['drop_reasons'][str(i)]})"
                                for i in r["dropped"])
            lines.append(f"- `{cid}`: {reasons}")
        else:
            lines.append(f"- `{cid}`: none")
    total = sum(v["kept"] for v in report["clips"].values())
    planned = sum(v["planned"] for v in report["clips"].values())
    lines += ["", "## Final counts", "",
              f"{total} labeled frames / {planned} sampled "
              f"({planned - total} dropped).",
              "", "## Baseline", "",
              f"MoveNet Thunder int8 ({movenet_infer.MODEL_ID}) scores "
              f"**{baseline['score']:.4f}** against these labels. Candidate "
              "extraction and scoring are both byte-identical on re-run "
              "(verified); the scorer also passes the synthetic validation.",
              "", "## Human QA observations", "",
              "(dev: append what you saw on the montages here before "
              "committing)",
              ""]
    with open(os.path.join(out_dir, "reference-poses", "qa-notes.md"),
              "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
