"""Unit tests for Training/evaluate.py.

Run: python -m unittest discover -s Training/tests -t .
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from Training import evaluate


def write_json(path: Path, obj) -> Path:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


class EvaluateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="evaluate-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.samples_dir = self.tmp / "samples"
        self.samples_dir.mkdir()
        self.split_path = self.tmp / "split.json"
        self.pred_path = self.tmp / "predictions.json"

    def _sample(self, sid: str, tricks: list[dict]) -> None:
        write_json(
            self.samples_dir / f"{sid}.json",
            {"sample_id": sid, "user_id": "user-a", "fps": 10, "tricks": tricks},
        )

    def _split(self, mapping: dict[str, str]) -> None:
        write_json(self.split_path, mapping)

    def _predictions(self, mapping: dict) -> None:
        write_json(self.pred_path, mapping)

    def _run(self, *extra: str) -> dict:
        out = self.tmp / "report.json"
        rc = evaluate.main(
            [
                "--predictions", str(self.pred_path),
                "--split", str(self.split_path),
                "--samples-dir", str(self.samples_dir),
                "--out", str(out),
                *extra,
            ]
        )
        self.assertEqual(rc, 0)
        return json.loads(out.read_text(encoding="utf-8"))

    def _basic_holdout(self) -> None:
        # One holdout sample with a single trick; one train sample.
        self._sample("h1", [{"name": "gainer", "start_frame": 4, "end_frame": 20}])
        self._sample("t1", [{"name": "gainer", "start_frame": 0, "end_frame": 8}])
        self._split({"h1": "holdout", "t1": "train"})

    def test_perfect_predictions(self) -> None:
        self._basic_holdout()
        self._predictions(
            {"h1": [{"start_frame": 4, "end_frame": 20, "name": "gainer"}]}
        )
        report = self._run()
        self.assertEqual(report["n_holdout_samples"], 1)
        self.assertEqual(report["n_ground_truth_segments"], 1)
        self.assertEqual(report["n_predicted_segments"], 1)
        self.assertEqual(report["detection_rate"], 1.0)
        self.assertEqual(report["precision"], 1.0)
        self.assertEqual(report["recall"], 1.0)
        self.assertEqual(report["f1"], 1.0)
        self.assertEqual(report["name_accuracy_given_detection"], 1.0)
        self.assertEqual(report["n_background_false_positives"], 0)
        self.assertEqual(report["n_unmatched_predictions"], 0)
        self.assertEqual(
            report["per_name"]["gainer"],
            {"support": 1, "predicted": 1, "true_positives": 1,
             "precision": 1.0, "recall": 1.0},
        )

    def test_empty_predictions(self) -> None:
        self._basic_holdout()
        self._predictions({})
        report = self._run()
        self.assertEqual(report["detection_rate"], 0.0)
        self.assertEqual(report["recall"], 0.0)
        self.assertEqual(report["precision"], 0.0)
        self.assertEqual(report["f1"], 0.0)
        self.assertEqual(report["per_sample"]["h1"]["n_detections"], 0)

    def test_no_tricks_no_predictions_is_vacuous(self) -> None:
        self._sample("h1", [])
        self._split({"h1": "holdout"})
        self._predictions({})
        report = self._run()
        self.assertEqual(report["precision"], 1.0)
        self.assertEqual(report["recall"], 1.0)
        self.assertEqual(report["f1"], 1.0)
        self.assertEqual(report["detection_rate"], 1.0)

    def test_byte_identical_reruns(self) -> None:
        self._basic_holdout()
        self._predictions(
            {
                "h1": [
                    {"start_frame": 4, "end_frame": 20, "name": "gainer"},
                    {"start_frame": 40, "end_frame": 48, "name": "cork"},
                ]
            }
        )
        out = self.tmp / "report.json"
        base = [
            "--predictions", str(self.pred_path),
            "--split", str(self.split_path),
            "--samples-dir", str(self.samples_dir),
            "--out", str(out),
        ]
        self.assertEqual(evaluate.main(base), 0)
        first = out.read_bytes()
        self.assertEqual(evaluate.main(base), 0)
        self.assertEqual(out.read_bytes(), first)

    def test_iou_threshold_boundary(self) -> None:
        # GT 4..20 (len 16); pred 14..24 (len 10) overlaps 6 frames:
        # IoU = 6 / (16 + 10 - 6) = 6/20 = 0.3. Threshold 0.25 detects it,
        # but the no-intersection-subtraction mutant (IoU = 6/26 ~= 0.23)
        # stays below 0.25, so the subtraction in temporal_iou is guarded.
        self._basic_holdout()
        self._predictions(
            {"h1": [{"start_frame": 14, "end_frame": 24, "name": "gainer"}]}
        )
        default = self._run()
        self.assertEqual(default["n_detections"], 0)
        lowered = self._run("--iou-threshold", "0.25")
        self.assertEqual(lowered["iou_threshold"], 0.25)
        self.assertEqual(lowered["n_detections"], 1)
        self.assertEqual(lowered["n_name_matches"], 1)

    def test_greedy_best_iou_matching(self) -> None:
        # GT (0,10)/gainer with competing preds (0,6)/cork (IoU 0.6) and
        # (0,9)/gainer (IoU 0.9): greedy best-IoU matches gainer and keeps
        # the name match; first-fit would match cork and lose it.
        self._sample(
            "h1", [{"name": "gainer", "start_frame": 0, "end_frame": 10}]
        )
        self._split({"h1": "holdout"})
        self._predictions(
            {
                "h1": [
                    {"start_frame": 0, "end_frame": 6, "name": "cork"},
                    {"start_frame": 0, "end_frame": 9, "name": "gainer"},
                ]
            }
        )
        report = self._run()
        self.assertEqual(report["n_detections"], 1)
        self.assertEqual(report["n_name_matches"], 1)
        self.assertEqual(report["n_unmatched_predictions"], 1)

    def test_iou_threshold_boundary_is_inclusive(self) -> None:
        # GT (4,20) vs pred (12,20): overlap 8, union 16 + 8 - 8 = 16,
        # IoU = 8/16 = 0.5 exactly. The spec'd "IoU >= threshold" detects at
        # threshold 0.5; a strict ">" would drop it.
        self._sample("h1", [{"name": "gainer", "start_frame": 4, "end_frame": 20}])
        self._split({"h1": "holdout"})
        self._predictions(
            {"h1": [{"start_frame": 12, "end_frame": 20, "name": "gainer"}]}
        )
        report = self._run()
        self.assertEqual(report["n_detections"], 1)
        self.assertEqual(report["n_name_matches"], 1)

    def test_name_mismatch_is_detection_not_match(self) -> None:
        self._basic_holdout()
        self._predictions(
            {"h1": [{"start_frame": 4, "end_frame": 20, "name": "cork"}]}
        )
        report = self._run()
        self.assertEqual(report["detection_rate"], 1.0)  # localized...
        self.assertEqual(report["precision"], 0.0)  # ...but misnamed
        self.assertEqual(report["recall"], 0.0)
        self.assertEqual(report["name_accuracy_given_detection"], 0.0)
        self.assertEqual(report["per_name"]["cork"]["predicted"], 1)
        self.assertEqual(report["per_name"]["cork"]["true_positives"], 0)

    def test_combo_name_forms_are_equal(self) -> None:
        self._sample(
            "h1", [{"name": "gainer+cork", "start_frame": 4, "end_frame": 20}]
        )
        self._split({"h1": "holdout"})
        self._predictions(
            {
                "h1": [
                    {
                        "start_frame": 4,
                        "end_frame": 20,
                        "names": ["cork", "gainer"],
                        "is_combo": True,
                    }
                ]
            }
        )
        report = self._run()
        self.assertEqual(report["n_name_matches"], 1)
        self.assertEqual(report["precision"], 1.0)

    def test_background_false_positive(self) -> None:
        self._basic_holdout()
        # Far from the GT segment: zero IoU -> hallucination.
        self._predictions(
            {"h1": [{"start_frame": 100, "end_frame": 110, "name": "cork"}]}
        )
        report = self._run()
        self.assertEqual(report["n_background_false_positives"], 1)
        self.assertEqual(report["n_unmatched_predictions"], 1)
        self.assertEqual(report["n_detections"], 0)

    def test_split_discipline_rejects_train_sample(self) -> None:
        self._basic_holdout()
        self._predictions(
            {"t1": [{"start_frame": 0, "end_frame": 8, "name": "gainer"}]}
        )
        with self.assertRaises(ValueError):
            evaluate.main(
                [
                    "--predictions", str(self.pred_path),
                    "--split", str(self.split_path),
                    "--samples-dir", str(self.samples_dir),
                    "--out", str(self.tmp / "report.json"),
                ]
            )

    def test_split_discipline_rejects_unknown_sample(self) -> None:
        self._basic_holdout()
        self._predictions(
            {"nope": [{"start_frame": 0, "end_frame": 8, "name": "gainer"}]}
        )
        with self.assertRaises(ValueError):
            evaluate.main(
                [
                    "--predictions", str(self.pred_path),
                    "--split", str(self.split_path),
                    "--samples-dir", str(self.samples_dir),
                    "--out", str(self.tmp / "report.json"),
                ]
            )

    def test_invalid_segment_rejected(self) -> None:
        self._basic_holdout()
        self._predictions(
            {"h1": [{"start_frame": 20, "end_frame": 20, "name": "gainer"}]}
        )
        with self.assertRaises(ValueError):
            evaluate.main(
                [
                    "--predictions", str(self.pred_path),
                    "--split", str(self.split_path),
                    "--samples-dir", str(self.samples_dir),
                    "--out", str(self.tmp / "report.json"),
                ]
            )

    def test_is_combo_contradiction_rejected(self) -> None:
        self._basic_holdout()
        self._predictions(
            {
                "h1": [
                    {
                        "start_frame": 4,
                        "end_frame": 20,
                        "name": "gainer",
                        "is_combo": True,
                    }
                ]
            }
        )
        with self.assertRaises(ValueError):
            evaluate.main(
                [
                    "--predictions", str(self.pred_path),
                    "--split", str(self.split_path),
                    "--samples-dir", str(self.samples_dir),
                    "--out", str(self.tmp / "report.json"),
                ]
            )

    def test_name_and_names_together_rejected(self) -> None:
        self._basic_holdout()
        self._predictions(
            {
                "h1": [
                    {
                        "start_frame": 4,
                        "end_frame": 20,
                        "name": "gainer",
                        "names": ["cork"],
                    }
                ]
            }
        )
        with self.assertRaises(ValueError):
            evaluate.main(
                [
                    "--predictions", str(self.pred_path),
                    "--split", str(self.split_path),
                    "--samples-dir", str(self.samples_dir),
                    "--out", str(self.tmp / "report.json"),
                ]
            )

    def test_empty_name_component_rejected(self) -> None:
        self._basic_holdout()
        self._predictions(
            {"h1": [{"start_frame": 4, "end_frame": 20, "name": "gainer+"}]}
        )
        with self.assertRaises(ValueError):
            evaluate.main(
                [
                    "--predictions", str(self.pred_path),
                    "--split", str(self.split_path),
                    "--samples-dir", str(self.samples_dir),
                    "--out", str(self.tmp / "report.json"),
                ]
            )

    def test_default_out_sits_next_to_predictions(self) -> None:
        self._basic_holdout()
        self._predictions(
            {"h1": [{"start_frame": 4, "end_frame": 20, "name": "gainer"}]}
        )
        rc = evaluate.main(
            [
                "--predictions", str(self.pred_path),
                "--split", str(self.split_path),
                "--samples-dir", str(self.samples_dir),
            ]
        )
        self.assertEqual(rc, 0)
        self.assertTrue((self.pred_path.parent / "holdout_report.json").is_file())

    def test_bad_iou_threshold_rejected(self) -> None:
        self._basic_holdout()
        self._predictions({})
        rc = evaluate.main(
            [
                "--predictions", str(self.pred_path),
                "--split", str(self.split_path),
                "--samples-dir", str(self.samples_dir),
                "--out", str(self.tmp / "report.json"),
                "--iou-threshold", "1.5",
            ]
        )
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
