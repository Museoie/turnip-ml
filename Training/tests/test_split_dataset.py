"""Unit tests for Training/split_dataset.py.

Fast and hermetic: synthetic manifests only, no network, no data files.
Run with:  python -m unittest discover -s Training/tests -t .
"""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from split_dataset import (
    SPLIT_SALT,
    dump_split,
    is_val_source,
    source_bucket,
    split_records,
)


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
        mapping = split_records(make_manifest())
        seen: dict[str, str] = {}
        for rec in make_manifest():
            split = mapping[rec["sample_id"]]
            src = rec["source_id"]
            if src in seen:
                self.assertEqual(seen[src], split, f"source {src} leaked across splits")
            seen[src] = split

    def test_byte_identical_across_runs(self):
        """Acceptance: same manifest = byte-identical."""
        manifest = make_manifest()
        first = dump_split(split_records(manifest))
        for _ in range(3):
            self.assertEqual(first, dump_split(split_records(manifest)))

    def test_new_sources_never_move_an_existing_one(self):
        """A source keeps its split however the manifest grows, so a source
        the champion trained on never comes back as validation."""
        before = split_records(make_manifest(n_sources=40))
        # The same seed draws the same clips for the first 40 sources, so
        # the grown manifest holds every earlier sample plus 360 new sources,
        # in another order.
        after = split_records(list(reversed(make_manifest(n_sources=400))))
        self.assertGreater(len(after), len(before))
        for sid, part in before.items():
            self.assertEqual(after[sid], part, sid)
        # And a source's split does not depend on the other sources at all.
        for rec in make_manifest(n_sources=40):
            alone = split_records([rec])
            self.assertEqual(alone[rec["sample_id"]], before[rec["sample_id"]])

    def test_input_order_does_not_matter(self):
        manifest = make_manifest()
        shuffled = list(reversed(manifest))
        self.assertEqual(
            dump_split(split_records(manifest)),
            dump_split(split_records(shuffled)),
        )

    def test_assignments_are_pinned(self):
        """The hash is part of the contract: these never change."""
        self.assertEqual(SPLIT_SALT, "turnip-split-v2:")
        self.assertEqual(source_bucket("0" * 64), 4690624720034527559)
        self.assertEqual(source_bucket("3" * 64), 399582391615860395)
        pinned = {c * 64: c in "3e" for c in "0123456789abcdef"}
        for src, val in pinned.items():
            with self.subTest(source=src[:1]):
                self.assertIs(is_val_source(src, 0.1), val)
        # The smoke fixture's sources, which its committed split relies on.
        self.assertEqual(
            {s: is_val_source(s, 0.5) for s in ("source-a", "source-b", "source-c")},
            {"source-a": False, "source-b": True, "source-c": False},
        )

    def test_the_ratio_is_a_per_source_probability(self):
        """Over many sources about val_ratio of them land in val."""
        n = 20000
        share = sum(is_val_source(f"{i:064x}", 0.1) for i in range(n)) / n
        self.assertAlmostEqual(share, 0.1, delta=0.01)
        # A bigger ratio only adds sources to val; none leaves it.
        for i in range(2000):
            src = f"{i:064x}"
            if is_val_source(src, 0.1):
                self.assertTrue(is_val_source(src, 0.2), src)

    def test_holdout_is_exactly_the_flagged_samples(self):
        """The farm's flag decides holdout: flagged sources never train, and
        nothing unflagged lands in holdout."""
        manifest = make_manifest(n_sources=10, holdout_sources=(3, 7))
        mapping = split_records(manifest)
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
            dump_split(split_records(manifest)),
            dump_split(split_records(explicit)),
        )

    def test_no_flag_means_no_holdout(self):
        mapping = split_records(make_manifest(n_sources=10))
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
        for bad in (0.0, 1.0, -0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(val_ratio=bad), self.assertRaises(ValueError):
                split_records(make_manifest(n_sources=5), val_ratio=bad)

    def test_empty_manifest(self):
        self.assertEqual(split_records([]), {})

    def test_a_source_is_indivisible(self):
        """All of a source's samples go where its hash sends it."""
        manifest = [{"sample_id": f"c{i}", "source_id": "0" * 64} for i in range(5)]
        mapping = split_records(manifest)
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
