"""Unit tests for Training/split_dataset.py.

Fast and hermetic: synthetic manifests only, no network, no data files.
Run with:  python -m unittest discover -s Training/tests -t .
"""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from split_dataset import dump_split, split_records


def make_manifest(
    n_sources=40, seed=1234, min_clips=1, max_clips=12, holdout_sources=()
):
    """Deterministic synthetic manifest: source_i holds a few clips each;
    the sources numbered in holdout_sources are flagged holdout."""
    import random

    rng = random.Random(seed)
    records = []
    for u in range(n_sources):
        n = rng.randint(min_clips, max_clips)
        for c in range(n):
            record = {
                "sample_id": f"clip_s{u:03d}_{c:02d}",
                "source_id": f"source_{u:03d}",
                "pose_frames": 90,  # extra fields are ignored
            }
            if u in holdout_sources:
                record["holdout"] = True
            records.append(record)
    return records


class SplitContractTest(unittest.TestCase):
    def test_no_source_leaks_across_splits(self):
        """Acceptance: no source_id appears in two splits."""
        mapping = split_records(make_manifest(), seed=7)
        seen: dict[str, str] = {}
        for rec in make_manifest():
            split = mapping[rec["sample_id"]]
            src = rec["source_id"]
            if src in seen:
                self.assertEqual(seen[src], split, f"source {src} leaked across splits")
            seen[src] = split

    def test_byte_identical_across_runs(self):
        """Acceptance: same manifest + seed = byte-identical."""
        manifest = make_manifest()
        first = dump_split(split_records(manifest, seed=42))
        for _ in range(3):
            self.assertEqual(first, dump_split(split_records(manifest, seed=42)))

    def test_different_seeds_may_differ(self):
        """The seed actually influences the assignment (not decorative)."""
        manifest = make_manifest()
        outs = {dump_split(split_records(manifest, seed=s)) for s in range(5)}
        self.assertGreater(len(outs), 1)

    def test_input_order_does_not_matter(self):
        manifest = make_manifest()
        shuffled = list(reversed(manifest))
        self.assertEqual(
            dump_split(split_records(manifest, seed=3)),
            dump_split(split_records(shuffled, seed=3)),
        )

    def test_ratios_approximate_targets(self):
        """90/10 train/val targets over the unflagged samples are hit
        approximately when sources are small."""
        manifest = make_manifest(n_sources=200, holdout_sources=range(20))
        mapping = split_records(manifest, seed=11)
        rest = [r["sample_id"] for r in manifest if not r.get("holdout")]
        counts = {
            s: sum(1 for sid in rest if mapping[sid] == s) for s in ("train", "val")
        }
        for split, target in (("train", 0.9), ("val", 0.1)):
            self.assertAlmostEqual(
                counts[split] / len(rest), target, delta=0.03, msg=f"{split}: {counts}"
            )

    def test_holdout_is_exactly_the_flagged_samples(self):
        """The farm's flag decides holdout: flagged sources never train, and
        nothing unflagged lands in holdout."""
        manifest = make_manifest(n_sources=10, holdout_sources=(3, 7))
        mapping = split_records(manifest, seed=1)
        for r in manifest:
            flagged = r["source_id"] in ("source_003", "source_007")
            self.assertEqual(mapping[r["sample_id"]] == "holdout", flagged, r)
        self.assertEqual(
            {mapping[r["sample_id"]] for r in manifest if not r.get("holdout")},
            {"train", "val"},
        )

    def test_a_false_flag_is_the_same_as_none(self):
        manifest = make_manifest(n_sources=10, holdout_sources=(3,))
        explicit = [{**r, "holdout": r.get("holdout", False)} for r in manifest]
        self.assertEqual(
            dump_split(split_records(manifest, seed=2)),
            dump_split(split_records(explicit, seed=2)),
        )

    def test_no_flag_means_no_holdout(self):
        mapping = split_records(make_manifest(n_sources=10), seed=1)
        self.assertNotIn("holdout", mapping.values())

    def test_require_holdout_refuses_a_manifest_with_no_flag(self):
        with self.assertRaisesRegex(ValueError, "no sample is flagged holdout"):
            split_records(make_manifest(n_sources=5), require_holdout=True)
        with self.assertRaisesRegex(ValueError, "no sample is flagged holdout"):
            split_records([], require_holdout=True)
        mapping = split_records(
            make_manifest(n_sources=5, holdout_sources=(0,)), require_holdout=True
        )
        self.assertIn("holdout", mapping.values())

    def test_a_source_that_disagrees_on_holdout_fails_closed(self):
        manifest = [
            {"sample_id": "a1", "source_id": "s", "holdout": True},
            {"sample_id": "a2", "source_id": "s"},
        ]
        with self.assertRaisesRegex(ValueError, "disagree on holdout") as cm:
            split_records(manifest)
        self.assertNotIn("'s'", str(cm.exception))

    def test_a_non_boolean_flag_fails_closed(self):
        for bad in ("true", 1, None, 0):
            with self.subTest(holdout=bad), self.assertRaises(ValueError):
                split_records([{"sample_id": "a", "source_id": "s", "holdout": bad}])

    def test_duplicate_sample_id_fails_closed(self):
        manifest = make_manifest(n_sources=5)
        manifest.append(dict(manifest[0]))
        with self.assertRaises(ValueError):
            split_records(manifest)

    def test_missing_fields_fail_closed(self):
        with self.assertRaises(ValueError):
            split_records([{"sample_id": "a"}])
        with self.assertRaises(ValueError):
            split_records([{"sample_id": "", "source_id": "s"}])
        # Account ids are not a grouping key: training data carries none.
        with self.assertRaises(ValueError):
            split_records([{"sample_id": "a", "user_id": "u"}])

    def test_bad_ratios_fail_closed(self):
        with self.assertRaises(ValueError):
            split_records(make_manifest(n_sources=5), train_ratio=0.9, val_ratio=0.2)
        with self.assertRaises(ValueError):
            split_records(make_manifest(n_sources=5), train_ratio=-0.1, val_ratio=1.1)

    def test_empty_manifest(self):
        self.assertEqual(split_records([]), {})

    def test_single_source_goes_to_train(self):
        """Indivisible sources keep integrity over exact ratios."""
        manifest = [{"sample_id": f"c{i}", "source_id": "only"} for i in range(5)]
        mapping = split_records(manifest, seed=0)
        self.assertTrue(all(v == "train" for v in mapping.values()))

    def test_cli_round_trip_is_byte_identical(self):
        """End-to-end: manifest.jsonl -> split.json, twice, identical bytes."""
        import subprocess
        import tempfile

        manifest = make_manifest(n_sources=30)
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "manifest.jsonl"
            inp.write_text("\n".join(json.dumps(r) for r in manifest) + "\n")
            outs = []
            for _ in range(2):
                out = Path(td) / "split.json"
                r = subprocess.run(
                    [
                        sys.executable,
                        str(
                            Path(__file__).resolve().parent.parent / "split_dataset.py"
                        ),
                        "--in",
                        str(inp),
                        "--out",
                        str(out),
                        "--seed",
                        "9",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(r.returncode, 0, r.stderr)
                outs.append(out.read_bytes())
                out.unlink()
            self.assertEqual(outs[0], outs[1])
            parsed = json.loads(outs[0])
            self.assertEqual(len(parsed), len(manifest))


if __name__ == "__main__":
    unittest.main()
