"""Unit tests for Training/retrain_schedule.py (TRAINING_DESIGN.md §2.1).

Run: python -m unittest discover -s Training/tests -t .
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import retrain_schedule as rs

TODAY = "2026-11-13"
WEBHOOK = "https://discord.example/api/webhooks/1/token"


def outcome(
    promotion: str,
    *,
    from_scratch: bool = True,
    refresh: bool = True,
    gate: str = "pass",
    membership: str = "pass",
) -> dict:
    """pipeline.sh's outcome.json, the fields record reads."""
    return {
        "from_scratch": from_scratch,
        "refresh": refresh,
        "promotion": promotion,
        "fixture_gate": gate,
        "membership_inference": membership,
    }


class FakeSender:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.error = error

    def __call__(self, url: str, payload: dict) -> None:
        self.calls.append((url, payload))
        if self.error:
            raise self.error


class ScheduleTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="retrain-schedule-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.state = self.tmp / "retrain_state.json"
        self.outcome = self.tmp / "outcome.json"
        self.stdout = self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.stderr = self.enterContext(contextlib.redirect_stderr(io.StringIO()))

    def last(self, date: str) -> None:
        self.state.write_text(json.dumps({"last_full_retrain": date}))

    def due(self, today: str = TODAY) -> tuple[int, dict]:
        argv = ["due", "--state", str(self.state), "--today", today]
        code = rs.main(argv)
        out = self.stdout.getvalue()
        self.stdout.seek(0)
        self.stdout.truncate()
        return code, json.loads(out) if code == 0 else {}

    def record(self, outcome: object, today: str = TODAY, send=None) -> int:
        self.outcome.write_text(json.dumps(outcome))
        argv = ["record", "--state", str(self.state), "--today", today]
        return rs.main([*argv, "--outcome", str(self.outcome)], send=send)

    def recorded(self, outcome: dict, today: str = TODAY, send=None) -> dict:
        """Run record; return its printed result."""
        self.stdout.seek(0)
        self.stdout.truncate()
        self.assertEqual(self.record(outcome, today, send), rs.EXIT_OK)
        return json.loads(self.stdout.getvalue())

    def stored(self) -> dict:
        return json.loads(self.state.read_text())


class RetrainScheduleTest(ScheduleTestCase):
    def test_no_recorded_retrain_is_due(self) -> None:
        self.assertEqual(
            self.due(), (0, {"due": True, "days_since_full_retrain": None})
        )

    def test_due_at_the_interval(self) -> None:
        # 2026-10-14 is 30 days before 2026-11-13; 2026-10-15 is 29.
        self.assertEqual(rs.RETRAIN_INTERVAL_DAYS, 30)
        self.last("2026-10-15")
        self.assertEqual(self.due(), (0, {"due": False, "days_since_full_retrain": 29}))
        self.last("2026-10-14")
        self.assertEqual(self.due(), (0, {"due": True, "days_since_full_retrain": 30}))

    def test_a_retrain_after_today_fails_closed(self) -> None:
        self.last("2026-11-14")
        self.assertEqual(self.due()[0], rs.EXIT_ERROR)
        self.state.write_text('{"refused_since": "2026-11-14"}')
        self.assertEqual(self.due()[0], rs.EXIT_ERROR)

    def test_a_refusal_streak_without_a_recorded_retrain_is_due(self) -> None:
        self.state.write_text('{"refused_since": "2026-11-10"}')
        self.assertEqual(
            self.due(), (0, {"due": True, "days_since_full_retrain": None})
        )

    def test_malformed_inputs_fail_closed(self) -> None:
        for bad_state in (
            "{}",
            '{"last_full_retrain": "2026-1-5"}',
            '{"refused_since": null}',
            '{"last_full_retrain": "2026-01-05", "refused": "2026-11-01"}',
            "[]",
            "nope",
        ):
            with self.subTest(state=bad_state):
                self.state.write_text(bad_state)
                self.assertEqual(self.due()[0], rs.EXIT_ERROR)
        self.state.unlink()
        for bad_today in ("2026-13-01", "13/11/2026", "2026-11-13T00:00"):
            with self.subTest(today=bad_today):
                self.assertEqual(self.due(bad_today)[0], rs.EXIT_ERROR)

    def test_a_from_scratch_champion_is_recorded(self) -> None:
        for promotion in ("promote", "bootstrap"):
            with self.subTest(promotion=promotion):
                self.last("2026-01-01")
                code = self.record(outcome(promotion))
                self.assertEqual(code, rs.EXIT_OK)
                self.assertEqual(
                    json.loads(self.state.read_text()), {"last_full_retrain": TODAY}
                )
                self.assertFalse(self.state.with_suffix(".json.tmp").exists())
        self.assertIn('"recorded": true', self.stdout.getvalue())

    def test_a_landing_ends_the_refusal_streak(self) -> None:
        self.state.write_text(
            '{"last_full_retrain": "2026-09-01", "refused_since": "2026-10-01"}'
        )
        result = self.recorded(outcome("promote"))
        self.assertEqual(result["recorded"], True)
        self.assertEqual(result["state_changed"], True)
        self.assertEqual(self.stored(), {"last_full_retrain": TODAY})

    def test_nothing_else_is_recorded(self) -> None:
        # Not a scheduled retrain: a refusal starts no streak either.
        for case in (
            outcome("archive", refresh=False),
            outcome("not-run", refresh=False),
            outcome("promote", from_scratch=False, refresh=False),
        ):
            with self.subTest(**case):
                self.last("2026-01-01")
                result = self.recorded(case)
                self.assertEqual(result["state_changed"], False)
                self.assertEqual(result["refused"], False)
                self.assertEqual(self.stored(), {"last_full_retrain": "2026-01-01"})

    def test_a_malformed_outcome_fails_closed(self) -> None:
        good = outcome("promote")
        for key in good:
            with self.subTest(missing=key):
                self.assertEqual(
                    self.record({k: v for k, v in good.items() if k != key}),
                    rs.EXIT_ERROR,
                )
        self.assertEqual(self.record({**good, "from_scratch": "true"}), rs.EXIT_ERROR)
        self.assertEqual(self.record([good]), rs.EXIT_ERROR)
        self.assertFalse(self.state.exists())


class OverdueAlertTest(ScheduleTestCase):
    """A scheduled retrain refused OVERDUE_ALERT_DAYS days in a row alerts."""

    def setUp(self) -> None:
        super().setUp()
        self.enterContext(mock.patch.dict(os.environ, {rs.WEBHOOK_ENV: WEBHOOK}))
        self.last("2026-10-01")

    def test_the_threshold_is_seven_days(self) -> None:
        self.assertEqual(rs.OVERDUE_ALERT_DAYS, 7)

    def test_the_first_refused_night_starts_the_streak(self) -> None:
        sender = FakeSender()
        result = self.recorded(outcome("archive"), send=sender)
        self.assertEqual(
            result,
            {
                "recorded": False,
                "refused": True,
                "days_overdue": 1,
                "refusing_gate": "promotion",
                "alert_status": "not-needed",
                "state_changed": True,
            },
        )
        self.assertEqual(
            self.stored(), {"last_full_retrain": "2026-10-01", "refused_since": TODAY}
        )
        self.assertEqual(sender.calls, [])
        # The next night keeps the first date and leaves the state alone.
        result = self.recorded(outcome("archive"), "2026-11-14", sender)
        self.assertEqual((result["days_overdue"], result["state_changed"]), (2, False))
        self.assertEqual(self.stored()["refused_since"], TODAY)

    def test_the_alert_fires_from_the_seventh_refused_day(self) -> None:
        self.state.write_text(
            '{"last_full_retrain": "2026-10-01", "refused_since": "2026-11-07"}'
        )
        sender = FakeSender()
        # 2026-11-07 .. 2026-11-12 is six days: no alert yet.
        result = self.recorded(outcome("archive"), "2026-11-12", sender)
        self.assertEqual(
            (result["days_overdue"], result["alert_status"]), (6, "not-needed")
        )
        self.assertEqual(sender.calls, [])
        # The seventh day alerts, and so does every refused night after it.
        for today, days in (("2026-11-13", 7), ("2026-11-20", 14)):
            result = self.recorded(outcome("archive"), today, sender)
            self.assertEqual(
                (result["days_overdue"], result["alert_status"]), (days, "sent")
            )
        self.assertEqual([url for url, _ in sender.calls], [WEBHOOK, WEBHOOK])
        payload = sender.calls[0][1]
        self.assertEqual(payload["allowed_mentions"], {"parse": []})
        self.assertEqual(
            payload["content"].splitlines()[1],
            "refused 7 day(s) in a row (alert from 7); refusing gate: `promotion`",
        )
        self.assertEqual(self.stored()["refused_since"], "2026-11-07")

    def test_the_alert_names_the_refusing_gate(self) -> None:
        self.state.write_text('{"refused_since": "2026-11-01"}')
        for case, gate in (
            (outcome("not-run", gate="regression"), "fixture_gate"),
            (outcome("archive", membership="fail"), "membership_inference"),
            (outcome("archive"), "promotion"),
            (outcome("not-run", gate="not-run"), "not_run"),
            (outcome("not-run", gate="pass"), "not_run"),
        ):
            with self.subTest(gate=gate):
                sender = FakeSender()
                result = self.recorded(case, send=sender)
                self.assertEqual(result["refusing_gate"], gate)
                content = sender.calls[0][1]["content"]
                self.assertIsNotNone(
                    re.search(rf"refusing gate: `{gate}`$", content, re.MULTILINE)
                )

    def test_an_undelivered_alert_is_reported_not_fatal(self) -> None:
        self.state.write_text('{"refused_since": "2026-11-01"}')
        for env, sender, status in (
            ({}, FakeSender(), "not-configured"),
            ({rs.WEBHOOK_ENV: WEBHOOK}, FakeSender(OSError("boom")), "failed: OSError"),
        ):
            with (
                self.subTest(status=status),
                mock.patch.dict(os.environ, env, clear=True),
            ):
                result = self.recorded(outcome("archive"), send=sender)
                self.assertEqual(result["alert_status"], status)
        self.assertNotIn("token", self.stderr.getvalue())
        self.assertIn("retrain_schedule: alert not delivered", self.stderr.getvalue())

    def test_the_alert_carries_counts_and_the_gate_only(self) -> None:
        content = rs.overdue_alert_content(9, "fixture_gate")
        self.assertEqual(
            content.splitlines(),
            [
                "**turnip-ml: scheduled retrain overdue**",
                "refused 9 day(s) in a row (alert from 7); refusing gate: `fixture_gate`",
                "the champion stays; the next nightly run retrains from scratch again",
            ],
        )


if __name__ == "__main__":
    unittest.main()
