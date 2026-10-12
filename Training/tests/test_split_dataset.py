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


def make_manifest(n_sources=40, seed=1234, min_clips=1, max_clips=12):
    """Deterministic synthetic manifest: source_i holds a few clips each."""
    import random

    rng = random.Random(seed)
    records = []
    for u in range(n_sources):
        n = rng.randint(min_clips, max_clips)
        for c in range(n):
            records.append(
                {
                    "sample_id": f"clip_s{u:03d}_{c:02d}",
                    "source_id": f"source_{u:03d}",
                    "pose_frames": 90,
                }
            )  # extra fields are ignored
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
        """80/10/10 targets are hit approximately when sources are small."""
        mapping = split_records(make_manifest(n_sources=200), seed=11)
        n = len(mapping)
        counts = {
            s: sum(1 for v in mapping.values() if v == s)
            for s in ("train", "val", "holdout")
        }
        for split, target in (("train", 0.8), ("val", 0.1), ("holdout", 0.1)):
            self.assertAlmostEqual(
                counts[split] / n, target, delta=0.03, msg=f"{split}: {counts}"
            )

    def test_holdout_pins_pull_whole_source(self):
        """Admin-curated holdout: pinned samples (and siblings) never train."""
        manifest = make_manifest(n_sources=10)
        pinned = ["clip_s003_00", "clip_s007_01"]
        mapping = split_records(manifest, seed=1, holdout_ids=pinned)
        for sid in pinned:
            self.assertEqual(mapping[sid], "holdout")
        # Siblings of pinned samples follow them into holdout.
        siblings = [
            r["sample_id"]
            for r in manifest
            if r["source_id"] in ("source_003", "source_007")
        ]
        for sid in siblings:
            self.assertEqual(mapping[sid], "holdout")
        # And nothing from those sources appears in train/val.
        self.assertNotIn(
            "source_003",
            {r["source_id"] for r in manifest if mapping[r["sample_id"]] != "holdout"},
        )
        self.assertNotIn(
            "source_007",
            {r["source_id"] for r in manifest if mapping[r["sample_id"]] != "holdout"},
        )

    def test_unknown_holdout_id_fails_closed(self):
        with self.assertRaises(ValueError):
            split_records(make_manifest(n_sources=5), holdout_ids=["nope"])

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
            split_records(
                make_manifest(n_sources=5),
                train_ratio=0.5,
                val_ratio=0.5,
                holdout_ratio=0.5,
            )
        with self.assertRaises(ValueError):
            split_records(
                make_manifest(n_sources=5),
                train_ratio=-0.1,
                val_ratio=0.6,
                holdout_ratio=0.5,
            )

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
