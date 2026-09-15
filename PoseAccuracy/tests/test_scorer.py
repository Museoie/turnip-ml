#!/usr/bin/env python3
"""Unit tests for PoseAccuracy/scorer.py.

Fast and hermetic: synthetic landmarks only, no network, no model download.
Run with:  python -m unittest discover -s PoseAccuracy/tests -t .
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scorer  # noqa: E402


def make_joints(noise_level=0.0, frame=0, confidence=0.9):
    """Deterministic synthetic 17-joint pose in normalized [0,1] coords.

    Shoulders/hips are placed so the reference person scale is 0.32
    (well above the scorer's MIN_SCALE clamp). `noise_level` scales a
    fixed deterministic displacement pattern per joint/frame.
    """
    joints = []
    for j in range(17):
        x = 0.50 + 0.02 * ((j * 7) % 5 - 2)
        y = 0.10 + 0.045 * j
        joints.append({"x": x, "y": y, "c": confidence})
    # anatomically sane torso for a stable person scale
    joints[5] = {"x": 0.40, "y": 0.30, "c": confidence}   # left_shoulder
    joints[6] = {"x": 0.60, "y": 0.30, "c": confidence}   # right_shoulder
    joints[11] = {"x": 0.42, "y": 0.62, "c": confidence}  # left_hip
    joints[12] = {"x": 0.58, "y": 0.62, "c": confidence}  # right_hip
    if noise_level:
        noisy = []
        for j, jt in enumerate(joints):
            # deterministic pattern in [-1, 1], scaled by noise_level
            w = (((j * 13 + frame * 7) % 11) - 5) / 5.0
            noisy.append({"x": jt["x"] + noise_level * 0.30 * w,
                          "y": jt["y"] + noise_level * 0.30 * (1 - abs(w)) * 0.5,
                          "c": jt["c"]})
        return noisy
    return joints


def make_doc(clip_frames, noise_level=0.0, confidence=0.9):
    """Build a scorer input doc: {clip_id: [frame_index, ...]} -> frames."""
    clips = {}
    for clip_id, frame_indexes in clip_frames.items():
        frames = []
        for fi in frame_indexes:
            frames.append({"frame_index": fi, "t": fi / 10.0,
                           "joints": make_joints(noise_level, fi, confidence)})
        clips[clip_id] = {"frames": frames}
    return {"clips": clips}


def zero_joints(confidence=0.0):
    return [{"x": 0.0, "y": 0.0, "c": confidence} for _ in range(17)]


class TestScorer(unittest.TestCase):
    def test_identical_input_scores_100(self):
        doc = make_doc({"clip_a": [0, 1, 2], "clip_b": [0, 1]})
        result = scorer.score_all(doc, doc)
        self.assertEqual(result["score"], 100.0)
        for clip in result["per_clip"].values():
            self.assertEqual(clip["score"], 100.0)

    def test_deterministic_across_runs(self):
        ref = make_doc({"clip_a": [0, 1, 2]})
        cand = make_doc({"clip_a": [0, 1, 2]}, noise_level=0.5)
        first = json.dumps(scorer.score_all(ref, cand), sort_keys=True)
        second = json.dumps(scorer.score_all(ref, cand), sort_keys=True)
        self.assertEqual(first, second)

    def test_dropped_frames_excluded_not_scored_as_zero(self):
        # Frame 1 dropped from both sides: absent frames contribute nothing.
        ref_full = make_doc({"clip": [0, 1, 2]})
        ref_dropped = make_doc({"clip": [0, 2]})
        full = scorer.score_all(ref_full, ref_full)
        dropped = scorer.score_all(ref_dropped, ref_dropped)
        self.assertEqual(full["score"], 100.0)
        self.assertEqual(dropped["score"], 100.0)
        self.assertEqual(dropped["per_clip"]["clip"]["n_frames"], 2)
        # A present-but-garbage frame scores near zero and drags the mean
        # down: absence must not behave like that.
        cand_zero = {"clips": {"clip": {"frames": [
            {"frame_index": 0, "t": 0.0, "joints": make_joints()},
            {"frame_index": 1, "t": 0.1, "joints": zero_joints()},
            {"frame_index": 2, "t": 0.2, "joints": make_joints()},
        ]}}}
        zero_scored = scorer.score_all(ref_full, cand_zero)
        self.assertLess(zero_scored["score"], 100.0)
        self.assertGreater(dropped["score"], zero_scored["score"])

    def test_empty_clips_raise_value_error(self):
        with self.assertRaises(ValueError):
            scorer.score_all({"clips": {}}, {"clips": {}})

    def test_empty_frames_raise_value_error(self):
        doc = {"clips": {"clip": {"frames": []}}}
        with self.assertRaises(ValueError):
            scorer.score_all(doc, doc)

    def test_noise_degrades_score_monotonically(self):
        ref = make_doc({"clip_a": [0, 1, 2], "clip_b": [0, 1, 2]})
        scores = []
        for level in (0.0, 0.5, 1.0, 2.0):
            cand = make_doc({"clip_a": [0, 1, 2], "clip_b": [0, 1, 2]},
                            noise_level=level)
            s = scorer.score_all(ref, cand)["score"]
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 100.0)
            scores.append(s)
        self.assertEqual(scores[0], 100.0)
        for lower, higher in zip(scores, scores[1:]):
            self.assertLess(higher, lower)

    def test_confidence_disagreement_lowers_score(self):
        ref = make_doc({"clip": [0, 1]}, confidence=0.9)
        cand = make_doc({"clip": [0, 1]}, confidence=0.1)
        s = scorer.score_all(ref, cand)["score"]
        self.assertLess(s, 100.0)
        self.assertGreater(s, 0.0)

    def test_score_frame_returns_per_joint_breakdown(self):
        joints = make_joints()
        frame_score, per_joint = scorer.score_frame(joints, joints)
        self.assertEqual(frame_score, 1.0)
        self.assertEqual(len(per_joint), 17)
        self.assertTrue(all(q == 1.0 for q in per_joint))


if __name__ == "__main__":
    unittest.main()
