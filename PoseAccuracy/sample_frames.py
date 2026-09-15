"""Shared deterministic frame sampler for the pose-accuracy harness.

Replicates the app's VideoFrameSampler cadence (~10 samples/sec) on this Linux
box: given a video, returns (results, metadata) where results is a list of
(frame_index, timestamp_sec, bgr_frame) for the deterministically chosen
sample indices. Both reference-label generation and
MoveNet candidate extraction MUST use this function so frames stay aligned.
"""
from __future__ import annotations
import cv2

TARGET_FPS = 10.0

def sample_plan(n_frames: int, src_fps: float, target_fps: float = TARGET_FPS) -> list[tuple[int, float]]:
    """Return [(frame_index, timestamp_sec)] — deterministic, no randomness."""
    if src_fps <= 0 or n_frames <= 0:
        return []
    step = src_fps / target_fps
    plan, i = [], 0
    while True:
        idx = int(round(i * step))
        if idx >= n_frames:
            break
        plan.append((idx, idx / src_fps))
        i += 1
    # de-dupe (can happen if src_fps < target_fps)
    seen, out = set(), []
    for idx, ts in plan:
        if idx not in seen:
            seen.add(idx)
            out.append((idx, ts))
    return out

def iter_samples(video_path: str, target_fps: float = TARGET_FPS):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    plan = sample_plan(n, fps, target_fps)
    want = {idx for idx, _ in plan}
    idx = 0
    results = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in want:
            ts = idx / fps
            results.append((idx, ts, frame))
        idx += 1
    cap.release()
    results.sort(key=lambda r: r[0])
    return results, {"n_frames": n, "fps": fps, "w": w, "h": h}
