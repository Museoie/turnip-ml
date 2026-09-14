#!/usr/bin/env python3
"""MoveNet Thunder candidate-pose extraction for the pose-accuracy harness.

Faithful Linux re-implementation of the app's Turnip/Pose pipeline
(verified against hoiekim/turnip-ios @ 6adc2953, see
estimator-interface-v2.md):

- Frame sampling: VideoFrameSampler.targetSamplesPerSecond = 10, fps-aware
  stride max(1, round(fps/10)) -> 30fps video = every 3rd frame. The sampler
  decodes through the track's preferredTransform, so a 720x1280 portrait clip
  yields 720x1280 portrait frames (cv2 gives us the same; the IG clips are
  already upright).
- Preprocessing (FramePreprocessor): uniform scale = min(256/w, 256/h),
  centered offsets, zero-filled (black) pad, INTER_LINEAR resize, RGB uint8
  0-255, no mean/std normalization. Model input [1,256,256,3] uint8.
- Inference: MoveNet Thunder single-pose int8; output float32 [1,1,17,3]
  with (y, x, score) per keypoint in MoveNet y-before-x order; parsed to
  x/y/confidence. COCO-17 order matches the scorer's joint order.
- Inverse letterbox (same file): srcX = (x*256 - offsetX)/scale, divided by
  the source extent -> frame-normalized coords in the ORIGINAL frame's space
  (pad-region keypoints may fall outside [0,1], exactly like the app).

Model provenance: the app bundles the Kaggle artifact
google/movenet singlepose-thunder-tflite-int8 (see Turnip/Models/README.md).
This script downloads from TFHub (no auth) and VERIFIES sha256/size against
the README-recorded identity before running; mismatch -> hard failure.

Output: scorer-schema JSON {"clips": {clip_id: {"frames":
[{"frame_index","t","joints":[{"x","y","c"} x17]}]}}}, 4-decimal rounding,
sorted keys. Frames are sample_frames.sample_plan() indices minus any
indices listed in --drop-frames, so they align frame-for-frame with the
reference labels.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.request

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sample_frames import sample_plan  # noqa: E402

DEFAULT_MODEL_URL = ("https://tfhub.dev/google/lite-model/movenet/singlepose/"
                     "thunder/tflite/int8/4?lite-format=tflite")
# Identity recorded in Turnip/Models/README.md @ 6adc2953 (Kaggle artifact).
EXPECTED_SHA256 = "b72fed22707cd6fb94b5a248b9bddb9c062b9f445471b4fa263407cf6d222011"
EXPECTED_SIZE = 7126768
MODEL_ID = ("movenet-thunder-int8/kaggle-singlepose-thunder-tflite-int8-1/"
            "sha256:b72fed22707cd6fb94b5a248b9bddb9c062b9f445471b4fa263407cf6d222011")

IN_SIZE = 256


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def obtain_model(model_path: str | None, model_url: str) -> str:
    if model_path:
        path = model_path
    else:
        url = model_url or DEFAULT_MODEL_URL
        path = os.path.join(tempfile.gettempdir(), "movenet_thunder_int8.tflite")
        if not os.path.exists(path):
            print(f"downloading model from {url} ...")
            req = urllib.request.Request(url, headers={"User-Agent": "turnip-pose-harness/1.0"})
            with urllib.request.urlopen(req, timeout=300) as r, open(path, "wb") as f:
                for chunk in iter(lambda: r.read(1 << 20), b""):
                    f.write(chunk)
    got_size = os.path.getsize(path)
    got_sha = sha256_file(path)
    if got_sha != EXPECTED_SHA256 or got_size != EXPECTED_SIZE:
        raise SystemExit(
            f"model identity mismatch: expected sha256 {EXPECTED_SHA256[:16]}... "
            f"size {EXPECTED_SIZE}, got {got_sha[:16]}... size {got_size}. Refusing to run.")
    print(f"model ok: sha256={got_sha[:16]}... size={got_size}")
    return path


def letterbox(frame_bgr: np.ndarray):
    """Replicate FramePreprocessor: returns (input_rgb_uint8, offsetX, offsetY, scale)."""
    h, w = frame_bgr.shape[:2]
    scale = min(IN_SIZE / w, IN_SIZE / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    canvas = np.zeros((IN_SIZE, IN_SIZE, 3), dtype=np.uint8)
    offset_x = (IN_SIZE - new_w) // 2
    offset_y = (IN_SIZE - new_h) // 2
    canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = rgb
    return canvas, offset_x, offset_y, scale


def make_interpreter(model_path: str):
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=model_path)
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    out = it.get_output_details()[0]
    assert list(inp["shape"]) == [1, IN_SIZE, IN_SIZE, 3], inp["shape"]
    assert inp["dtype"] == np.uint8, inp["dtype"]
    return it, inp["index"], out["index"]


def infer_frame(it, in_idx: int, out_idx: int, input_rgb: np.ndarray):
    it.set_tensor(in_idx, np.expand_dims(input_rgb, axis=0))
    it.invoke()
    return it.get_tensor(out_idx)  # [1,1,17,3] float32 (y,x,score)


def to_frame_normalized(kpts_17x3, offset_x, offset_y, scale, w, h):
    joints = []
    for k in range(17):
        y, x, score = (float(v) for v in kpts_17x3[k])
        src_x = (x * IN_SIZE - offset_x) / scale / w
        src_y = (y * IN_SIZE - offset_y) / scale / h
        joints.append({"x": round(src_x, 4), "y": round(src_y, 4),
                       "c": round(score, 4)})
    return joints


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--frames-dir", required=True,
                    help="dir containing the fixture mp4s (named per manifest)")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--model-url", default=None)
    ap.add_argument("--drop-frames", default=None,
                    help="dropped_frames.json: sampled indices with no reference "
                         "label are skipped so the candidate aligns frame-for-frame "
                         "with the reference")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    with open(a.manifest) as f:
        manifest = json.load(f)
    dropped = {}
    if a.drop_frames:
        with open(a.drop_frames) as f:
            dropped = {cid: set(idxs) for cid, idxs in json.load(f).items()}
    model = obtain_model(a.model_path, a.model_url)
    it, in_idx, out_idx = make_interpreter(model)

    clips_out = {}
    for clip in manifest["clips"]:
        cid = clip["id"]
        path = os.path.join(a.frames_dir, clip["file"])
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise SystemExit(f"cannot open {path}")
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        plan = sample_plan(n, fps)
        skip = dropped.get(cid, set())
        want = {idx for idx, _ in plan} - skip
        frames = []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in want:
                ts = idx / fps
                inp, ox, oy, sc = letterbox(frame)
                raw = infer_frame(it, in_idx, out_idx, inp)[0, 0]
                joints = to_frame_normalized(raw, ox, oy, sc, w, h)
                frames.append({"frame_index": idx, "t": round(ts, 3), "joints": joints})
            idx += 1
        cap.release()
        frames.sort(key=lambda r: r["frame_index"])
        expected = len(plan) - len(skip)
        if len(frames) != expected:
            raise SystemExit(f"{cid}: sampled {len(frames)} frames, expected {expected}")
        clips_out[cid] = {"frames": frames}
        print(f"{cid}: {len(frames)} frames scored")

    with open(a.out, "w") as f:
        json.dump({"clips": clips_out}, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"wrote {a.out} model_id={MODEL_ID}")


if __name__ == "__main__":
    main()
