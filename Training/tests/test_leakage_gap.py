"""Unit tests for Training/leakage_gap.py (turnip-ml#42).

Run: python -m unittest discover -s Training/tests -t .
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import evaluate
import leakage_gap


def write_json(path: Path, obj) -> Path:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seg(name: str, start: int, end: int) -> dict:
    return {"name": name, "start_frame": start, "end_frame": end}


class LeakageGapTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="leakage-gap-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.samples_dir = self.tmp / "samples"
        self.samples_dir.mkdir()
        self.split_path = self.tmp / "split.json"
        self.metrics_path = self.tmp / "metrics.json"
        self.report_path = self.tmp / "report.json"
        self.registry_path = self.tmp / "registry.json"

    def _sample(self, sid: str, tricks: list[dict]) -> None:
        write_json(
            self.samples_dir / f"{sid}.json",
            {"sample_id": sid, "fps": 10, "tricks": tricks},
        )

    def _write_inputs(
        self,
        val_accuracy: float = 0.9,
        val_segments: list[dict] | None = None,
        holdout_f1: float = 0.9,
        holdout_acc: float = 0.7,
    ) -> None:
        # One val sample, one GT segment; the default val_segments match
        # it exactly, so val_f1 == 1.0.
        self._sample("v1", [seg("cork", 0, 8)])
        self._sample("h1", [seg("cork", 0, 8)])
        write_json(self.split_path, {"v1": "val", "h1": "holdout", "t1": "train"})
        split_sha = sha256_file(self.split_path)
        if val_segments is None:
            val_segments = [
                {
                    "sample_id": "v1",
                    "name": "cork",
                    "start_frame": 0,
                    "end_frame": 8,
                    "is_combo": False,
                }
            ]
        write_json(
            self.metrics_path,
            {
                "val_accuracy": val_accuracy,
                "val_segments": val_segments,
                "split_sha256": split_sha,
            },
        )
        write_json(
            self.report_path,
            {
                "f1": holdout_f1,
                "name_accuracy_given_detection": holdout_acc,
                "iou_threshold": 0.5,
                "inputs": {"split_sha256": split_sha},
            },
        )

    def _run(self, promotion: str = "promote") -> int:
        return leakage_gap.main(
            [
                "--metrics",
                str(self.metrics_path),
                "--report",
                str(self.report_path),
                "--split",
                str(self.split_path),
                "--samples-dir",
                str(self.samples_dir),
                "--registry",
                str(self.registry_path),
                "--promotion",
                promotion,
            ]
        )

    def _report(self) -> dict:
        return json.loads(self.report_path.read_text())

    def _registry(self) -> list:
        return json.loads(self.registry_path.read_text())

    def test_gaps_written_to_report(self) -> None:
        self._write_inputs()
        self.assertEqual(self._run("archive"), 0)
        gap = self._report()["val_minus_holdout"]
        # promotion: 0.9 - 0.7 = 0.2; segment_f1: 1.0 - 0.9 = 0.1.
        self.assertEqual(gap["promotion_metric"], 0.2)
        self.assertEqual(gap["segment_f1"], 0.1)
        self.assertEqual(gap["val_f1"], 1.0)
        self.assertEqual(gap["streak_promoted_runs_over_threshold"], 0)
        self.assertFalse(gap["flagged"])

    def test_streak_advances_and_flags_on_third(self) -> None:
        self._write_inputs()
        prior = [
            {
                "kind": "leakage_gap",
                "promotion": "promote",
                "val_minus_holdout": {
                    "promotion_metric": 0.2,
                    "segment_f1": 0.1,
                },
            },
            {
                "kind": "leakage_gap",
                "promotion": "promote",
                "val_minus_holdout": {
                    "promotion_metric": 0.2,
                    "segment_f1": 0.1,
                },
            },
        ]
        write_json(self.registry_path, prior)
        self.assertEqual(self._run("promote"), 0)
        gap = self._report()["val_minus_holdout"]
        self.assertEqual(gap["streak_promoted_runs_over_threshold"], 3)
        self.assertTrue(gap["flagged"])
        self.assertEqual(len(self._registry()), 3)

    def test_streak_resets_on_archive(self) -> None:
        self._write_inputs()
        write_json(
            self.registry_path,
            [
                {
                    "kind": "leakage_gap",
                    "promotion": "promote",
                    "val_minus_holdout": {
                        "promotion_metric": 0.2,
                        "segment_f1": 0.1,
                    },
                }
            ],
        )
        self.assertEqual(self._run("archive"), 0)
        gap = self._report()["val_minus_holdout"]
        self.assertEqual(gap["streak_promoted_runs_over_threshold"], 0)
        self.assertFalse(gap["flagged"])

    def test_streak_resets_when_under_threshold(self) -> None:
        self._write_inputs(val_accuracy=0.72, holdout_f1=0.99)
        write_json(
            self.registry_path,
            [
                {
                    "kind": "leakage_gap",
                    "promotion": "promote",
                    "val_minus_holdout": {
                        "promotion_metric": 0.2,
                        "segment_f1": 0.1,
                    },
                }
            ],
        )
        self.assertEqual(self._run("promote"), 0)
        gap = self._report()["val_minus_holdout"]
        # promotion 0.02, segment_f1 0.01: both under 0.05.
        self.assertEqual(gap["promotion_metric"], 0.02)
        self.assertEqual(gap["segment_f1"], 0.01)
        self.assertEqual(gap["streak_promoted_runs_over_threshold"], 0)

    def test_bootstrap_leaves_streak_unchanged(self) -> None:
        self._write_inputs()
        write_json(
            self.registry_path,
            [
                {
                    "kind": "leakage_gap",
                    "promotion": "promote",
                    "val_minus_holdout": {
                        "promotion_metric": 0.2,
                        "segment_f1": 0.1,
                    },
                }
            ],
        )
        self.assertEqual(self._run("bootstrap"), 0)
        gap = self._report()["val_minus_holdout"]
        self.assertEqual(gap["streak_promoted_runs_over_threshold"], 1)
        self.assertFalse(gap["flagged"])

    def test_other_registry_kinds_ignored(self) -> None:
        self._write_inputs()
        write_json(
            self.registry_path,
            [
                {
                    "decision": "promote",
                    "metric": "val_accuracy",
                }
            ],
        )
        self.assertEqual(self._run("promote"), 0)
        gap = self._report()["val_minus_holdout"]
        self.assertEqual(gap["streak_promoted_runs_over_threshold"], 1)

    def test_split_mismatch_fails_closed(self) -> None:
        self._write_inputs()
        metrics = json.loads(self.metrics_path.read_text())
        metrics["split_sha256"] = "0" * 64
        write_json(self.metrics_path, metrics)
        before = self.report_path.read_text()
        self.assertEqual(self._run("promote"), 2)
        self.assertEqual(self.report_path.read_text(), before)
        self.assertFalse(self.registry_path.exists())

    def test_val_segment_outside_val_fails(self) -> None:
        self._write_inputs(
            val_segments=[
                {
                    "sample_id": "h1",
                    "name": "cork",
                    "start_frame": 0,
                    "end_frame": 8,
                    "is_combo": False,
                }
            ]
        )
        self.assertEqual(self._run("promote"), 2)

    def test_missing_report_key_fails(self) -> None:
        self._write_inputs()
        report = json.loads(self.report_path.read_text())
        del report["f1"]
        write_json(self.report_path, report)
        self.assertEqual(self._run("promote"), 2)


if __name__ == "__main__":
    unittest.main()
