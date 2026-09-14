#!/usr/bin/env python3
"""Deterministic pose-accuracy scorer (source of truth for the CI gate).

Compares candidate poses (e.g. MoveNet Thunder) against reference poses
(e.g. MediaPipe heavy) frame-by-frame over the fixture clips.

Joint order (COCO-17), normalized [0,1] coords, origin top-left:
  0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
  5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow,
  9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip,
  13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle

Metric (documented, deterministic):
  For each frame f and joint j:
    d(f,j)   = hypot(cx - rx, cy - ry)              # normalized-coord displacement
    s(f)     = max(0.05, dist(mid(ref shoulders), mid(ref hips)))
                                                # person scale from REFERENCE (stable)
    nd(f,j)  = d(f,j) / s(f)                        # scale-normalized displacement
    q_disp   = max(0, 1 - nd / 0.6)                  # 0.6 torso-lengths = total miss
    q_conf   = 1 - |conf_c - conf_r|                # occlusion/confidence agreement
    q(f,j)   = 0.85 * q_disp + 0.15 * q_conf
  frame score = mean_j q(f,j)
  clip score  = mean_f frame score
  total score = mean over clips (equal weight) * 100  -> 0..100

Determinism: no timestamps, no RNG, sorted keys, floats rounded to 4 decimals.
Usage:
  scorer.py --reference REF.json --candidate CAND.json --model-id ID \\
            --manifest-sha SHA --out OUT.json
Input JSON schema:
  {"clips": {"<clip_id>": {"frames": [{"frame_index": int, "t": float,
     "joints": [{"x": float, "y": float, "c": float} x17]}]}}}
"""
from __future__ import annotations
import argparse, json, math, sys

JOINT_NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
               "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
               "left_wrist", "right_wrist", "left_hip", "right_hip",
               "left_knee", "right_knee", "left_ankle", "right_ankle"]
assert len(JOINT_NAMES) == 17

W_DISP, W_CONF = 0.85, 0.15
MISS_ND = 0.6          # normalized displacement that scores zero
MIN_SCALE = 0.05       # clamp for person scale
ROUND = 4

SCORER_VERSION = "1.0.0"


def _mid(a, b):
    return ((a["x"] + b["x"]) / 2.0, (a["y"] + b["y"]) / 2.0)


def _dist(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])


def score_frame(ref_joints, cand_joints):
    """Return (frame_score, per_joint_scores[17])."""
    r = ref_joints
    scale = _dist(_mid(r[5], r[6]), _mid(r[11], r[12]))
    scale = max(MIN_SCALE, scale)
    per_joint = []
    for j in range(17):
        d = math.hypot(cand_joints[j]["x"] - r[j]["x"],
                       cand_joints[j]["y"] - r[j]["y"])
        nd = d / scale
        q_disp = max(0.0, 1.0 - nd / MISS_ND)
        q_conf = 1.0 - abs(cand_joints[j]["c"] - r[j]["c"])
        q_conf = min(1.0, max(0.0, q_conf))
        per_joint.append(W_DISP * q_disp + W_CONF * q_conf)
    return sum(per_joint) / 17.0, per_joint


def r4(x):
    return round(float(x), ROUND)


def score_all(reference, candidate):
    ref_clips = reference["clips"]
    cand_clips = candidate["clips"]
    if sorted(ref_clips) != sorted(cand_clips):
        raise SystemExit(f"clip id mismatch: ref={sorted(ref_clips)} cand={sorted(cand_clips)}")
    per_clip = {}
    joint_accum = {name: [] for name in JOINT_NAMES}
    for clip_id in sorted(ref_clips):
        rf = ref_clips[clip_id]["frames"]
        cf = cand_clips[clip_id]["frames"]
        if len(rf) != len(cf):
            raise SystemExit(f"frame count mismatch in {clip_id}: {len(rf)} vs {len(cf)}")
        frame_scores = []
        clip_joint = {name: [] for name in JOINT_NAMES}
        for rfr, cfr in zip(rf, cf):
            if rfr["frame_index"] != cfr["frame_index"]:
                raise SystemExit(f"frame_index mismatch in {clip_id}")
            fs, pj = score_frame(rfr["joints"], cfr["joints"])
            frame_scores.append(fs)
            for name, q in zip(JOINT_NAMES, pj):
                clip_joint[name].append(q)
                joint_accum[name].append(q)
        clip_score = sum(frame_scores) / len(frame_scores)
        per_clip[clip_id] = {
            "score": r4(clip_score * 100),
            "n_frames": len(frame_scores),
            "per_joint": {n: r4(sum(v) / len(v) * 100) for n, v in sorted(clip_joint.items())},
        }
    total = sum(c["score"] for c in per_clip.values()) / len(per_clip)
    return {
        "score": r4(total),
        "scorer_version": SCORER_VERSION,
        "n_clips": len(per_clip),
        "per_clip": per_clip,
        "per_joint_overall": {n: r4(sum(v) / len(v) * 100)
                              for n, v in sorted(joint_accum.items())},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--manifest-sha", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.reference) as f:
        reference = json.load(f)
    with open(a.candidate) as f:
        candidate = json.load(f)
    result = score_all(reference, candidate)
    result["model_id"] = a.model_id
    result["fixture_manifest_sha256"] = a.manifest_sha
    # stable key order: score, model_id, manifest sha, version, counts, breakdowns
    ordered = {
        "score": result["score"],
        "model_id": result["model_id"],
        "fixture_manifest_sha256": result["fixture_manifest_sha256"],
        "scorer_version": result["scorer_version"],
        "n_clips": result["n_clips"],
        "per_joint_overall": result["per_joint_overall"],
        "per_clip": result["per_clip"],
    }
    with open(a.out, "w") as f:
        json.dump(ordered, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"score={result['score']:.4f} model={a.model_id}")


if __name__ == "__main__":
    main()
