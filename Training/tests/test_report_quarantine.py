"""Unit tests for Training/report_quarantine.py (turnip-ml#51).

The fixture gate's quarantine.json -> POST /api/quarantine against a fake
farm that validates like turnip-farm's readQuarantineInput, resolves
segment ids to sources like its segment_sources join,
answers counts only, and keeps one open quarantine per source, like its
data_quarantine_open_uniq index.
Hermetic: no network beyond a local HTTP stand-in, no secrets.

Run: python -m unittest discover -s Training/tests -t .
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import threading
import unittest
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fixture_gate
import report_quarantine

FARM = "https://farm.example"
KEY = "pipeline-secret-for-tests"
HEX64 = re.compile(r"[0-9a-f]{64}")
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
BEARER = re.compile(r"^Bearer (\S+)$", re.IGNORECASE)
REASON = "fixture regression: challenger composite 0.400000 < champion 0.900000"
METRICS = {"delta": -0.5, "fixture_set": "f" * 64, "tolerance": 0.01}


def source_id(n: int) -> str:
    return hashlib.sha256(f"source-{n}".encode()).hexdigest()


def segment_id(n: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"turnip-test/segment/{n}"))


def ids(count: int) -> list[str]:
    """Segment ids; segments 2k and 2k+1 are cut from source k (LINKS)."""
    return [segment_id(n) for n in range(count)]


# segment_sources as the fake farm holds it: two segments per source.
LINKS = {segment_id(n): source_id(n // 2) for n in range(2 * 10000 + 2)}


def rows(
    segments: list[str], reason: str = REASON, metrics: Any = METRICS
) -> list[dict[str, Any]]:
    return fixture_gate.quarantine_rows(segments, reason, metrics)


SUMMARY = re.compile(
    r"report_quarantine: (\d+) segments in (\d+) request\(s\): (\d+) sources "
    r"quarantined, (\d+) already open; (\d+) segments unlinked "
)


def totals(stdout: str) -> dict[str, int]:
    """The last summary line's counts, compared whole (never as substrings)."""
    found = SUMMARY.findall(stdout)
    assert found, f"no summary line in {stdout!r}"
    keys = ("segments", "requests", "quarantined", "open", "unlinked")
    return dict(zip(keys, map(int, found[-1]), strict=True))


def counts(
    segments: int, requests: int, quarantined: int, open: int, unlinked: int = 0
) -> dict[str, int]:
    return {
        "segments": segments,
        "requests": requests,
        "quarantined": quarantined,
        "open": open,
        "unlinked": unlinked,
    }


CODES = {401: "UNAUTHENTICATED", 404: "NOT_FOUND", 422: "VALIDATION_ERROR"}


def error(status: int, **details: Any) -> tuple[int, Any]:
    code = CODES.get(status, "INTERNAL")
    body: dict[str, Any] = {"code": code, "message": f"farm says {code}"}
    if details:
        body["details"] = details
    return status, {"error": body}


# -- the fake farm ---------------------------------------------------------------


def js_length(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def farm_validation_error(body: Any) -> str | None:
    """turnip-farm src/quarantine.ts readQuarantineInput, rule for rule."""
    if not isinstance(body, dict):
        return "body must be a JSON object"
    lists = {"source_ids": HEX64, "segment_ids": UUID}
    for key, pattern in lists.items():
        if key not in body:
            continue
        values = body[key]
        if not isinstance(values, list) or len(values) > 10000:
            return f"{key} must be an array of at most 10000 ids"
        if not all(isinstance(i, str) and pattern.fullmatch(i) for i in values):
            return f"{key} holds a malformed id"
    if not any(body.get(key) for key in lists):
        return "source_ids or segment_ids must be non-empty"
    reason = body.get("reason")
    if not isinstance(reason, str) or not reason.strip() or js_length(reason) > 1000:
        return "reason must be a non-empty string"
    metrics = body.get("metrics")
    if metrics is not None and not isinstance(metrics, dict):
        return "metrics must be a JSON object or null"
    return None


class FakeFarm:
    """POST /api/quarantine as turnip-farm serves it."""

    def __init__(self, deleted: list[str] | None = None):
        # segment_sources, less the segments deleted since the export pull.
        self.links = {k: v for k, v in LINKS.items() if k not in set(deleted or [])}
        # Open quarantines, by source id.
        self.open: dict[str, tuple[str, Any]] = {}
        self.posts: list[dict[str, Any]] = []
        self.calls = 0
        # Statuses answered before the farm does anything, one per POST.
        self.fail: list[int] = []
        # Every POST fails with a 503 once this many bodies have landed.
        self.down_after: int | None = None
        # Insert, then lose the answer (a timeout after commit).
        self.drop_next_answer = False
        # Rewrites a well-formed answer, to model a farm that drifted.
        self.tamper: Any = None

    def transport(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None
    ) -> tuple[int, Any]:
        self.calls += 1
        if method != "POST" or url != FARM + "/api/quarantine":
            return error(404)
        match = BEARER.match(headers.get("Authorization", "").strip())
        if match is None or match.group(1) != KEY:
            return error(401)
        if self.fail:
            # A 422's details name the offending id, as the farm's do.
            return error(self.fail.pop(0), id=segment_id(0))
        if self.down_after is not None and len(self.posts) >= self.down_after:
            return error(503)
        try:
            payload = json.loads(body or b"")
        except ValueError:
            return error(422)
        if farm_validation_error(payload):
            return error(422)
        if set(payload) != {"segment_ids", "reason", "metrics"}:
            raise AssertionError(f"unexpected keys sent: {set(payload)}")
        self.posts.append(payload)
        # source_ids fill the lists; this pipeline sends segment_ids only.
        answer: dict[str, Any] = {"quarantined": [], "already_open": [], "unknown": []}
        segments = list(dict.fromkeys(payload["segment_ids"]))
        sources = {self.links[s] for s in segments if s in self.links}
        counts = {
            "sources_quarantined": 0,
            "sources_already_open": 0,
            "unlinked": sum(1 for s in segments if s not in self.links),
        }
        for sid in sorted(sources):
            if sid in self.open:
                counts["sources_already_open"] += 1
            else:
                self.open[sid] = (payload["reason"].strip(), payload.get("metrics"))
                counts["sources_quarantined"] += 1
        answer["segments"] = counts
        if self.drop_next_answer:
            self.drop_next_answer = False
            raise report_quarantine.Transient("farm request failed: TimeoutError")
        if self.tamper is not None:
            return 200, self.tamper(answer)
        return 200, answer


# -- tests -----------------------------------------------------------------------


class ReportTestCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = self.enterContext(tempfile.TemporaryDirectory(prefix="quarantine-"))
        self.path = Path(tmp) / "quarantine.json"
        self.stdout = self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.stderr = self.enterContext(contextlib.redirect_stderr(io.StringIO()))
        env = {"TURNIP_FARM_URL": FARM, "TURNIP_PIPELINE_KEY": KEY}
        self.enterContext(mock.patch.dict(os.environ, env))
        self.farm = FakeFarm()
        self.sleeps: list[float] = []

    def write(self, data: Any) -> None:
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

    def main(self, farm: FakeFarm | None = None) -> int:
        return report_quarantine.main(
            ["--quarantine", str(self.path)],
            transport=(farm or self.farm).transport,
            sleep=self.sleeps.append,
        )

    def logs(self) -> str:
        return self.stdout.getvalue() + self.stderr.getvalue()

    def clear_logs(self) -> None:
        for stream in (self.stdout, self.stderr):
            stream.seek(0)
            stream.truncate()

    def assertLogsArePrivate(self) -> None:
        logs = self.logs()
        self.assertIsNone(HEX64.search(logs), "a source id reached the log")
        self.assertIsNone(UUID.search(logs), "a segment id reached the log")
        for secret in (KEY, FARM, "farm says"):
            self.assertNotIn(secret, logs)


class RoundTripTest(ReportTestCase):
    def test_rows_become_one_request(self) -> None:
        segments = ids(3)
        self.write(rows(segments))
        self.assertEqual(self.main(), 0, self.stderr.getvalue())
        self.assertEqual(len(self.farm.posts), 1)
        self.assertEqual(
            self.farm.posts[0],
            {"segment_ids": sorted(segments), "reason": REASON, "metrics": METRICS},
        )
        # Segments 0 and 1 share source 0; segment 2 is source 1's.
        self.assertEqual(set(self.farm.open), {source_id(0), source_id(1)})
        self.assertEqual(totals(self.stdout.getvalue()), counts(3, 1, 2, 0))
        self.assertLogsArePrivate()

    def test_the_gates_own_output_is_sent(self) -> None:
        out = self.path.parent / "gate"
        champion, challenger, ingested = (
            self.path.parent / name
            for name in ("champion.json", "challenger.json", "ingested.json")
        )
        for path, score in ((champion, 0.9), (challenger, 0.4)):
            path.write_text(
                json.dumps(
                    {
                        "fixture_set": "f" * 64,
                        "segment_iou": score,
                        "name_accuracy": score,
                    }
                )
            )
        ingested.write_text(json.dumps(ids(4)))
        _, code = fixture_gate.run(
            champion,
            challenger,
            ingested,
            out,
            0.01,
            self.path.parent / "registry.json",
        )
        self.assertEqual(code, fixture_gate.EXIT_REGRESSION)
        self.path = out / fixture_gate.QUARANTINE_FILE
        self.assertEqual(self.main(), 0, self.stderr.getvalue())
        self.assertEqual(set(self.farm.open), {source_id(0), source_id(1)})
        reason, metrics = self.farm.open[source_id(0)]
        self.assertTrue(reason.startswith("fixture regression"))
        self.assertEqual(metrics["delta"], -0.5)

    def test_chunks_at_the_farms_limit(self) -> None:
        segments = ids(2 * report_quarantine.MAX_SEGMENT_IDS + 1)
        self.write(rows(segments))
        self.assertEqual(self.main(), 0, self.stderr.getvalue())
        sent = [post["segment_ids"] for post in self.farm.posts]
        self.assertEqual([len(chunk) for chunk in sent], [10000, 10000, 1])
        self.assertEqual([i for chunk in sent for i in chunk], sorted(segments))
        self.assertEqual(set(self.farm.open), {LINKS[s] for s in segments})
        # A source whose two segments straddle a chunk boundary is newly
        # quarantined by one body and already open for the next.
        seen: set[str] = set()
        straddling = 0
        for chunk in sent:
            chunk_sources = {LINKS[s] for s in chunk}
            straddling += len(chunk_sources & seen)
            seen |= chunk_sources
        self.assertGreater(straddling, 0)
        self.assertEqual(
            totals(self.stdout.getvalue()), counts(20001, 3, len(seen), straddling)
        )

    def test_a_full_chunk_is_one_request(self) -> None:
        self.write(rows(ids(report_quarantine.MAX_SEGMENT_IDS)))
        self.assertEqual(self.main(), 0, self.stderr.getvalue())
        self.assertEqual(len(self.farm.posts), 1)

    def test_each_reason_and_metrics_gets_its_own_body(self) -> None:
        segments = ids(6)
        self.write(
            rows(segments[:2])
            + rows(segments[2:4], reason="another reason")
            + rows(segments[4:], metrics=None)
        )
        self.assertEqual(self.main(), 0, self.stderr.getvalue())
        bodies = sorted(
            (p["reason"], json.dumps(p["metrics"]), p["segment_ids"])
            for p in self.farm.posts
        )
        self.assertEqual(
            bodies,
            sorted(
                [
                    (REASON, json.dumps(METRICS), sorted(segments[:2])),
                    ("another reason", json.dumps(METRICS), sorted(segments[2:4])),
                    (REASON, "null", sorted(segments[4:])),
                ]
            ),
        )

    def test_requests_are_deterministic(self) -> None:
        segments = ids(5)
        self.write(list(reversed(rows(segments))))
        first = report_quarantine.build_requests(report_quarantine.load_rows(self.path))
        self.write(rows(segments))
        again = report_quarantine.build_requests(report_quarantine.load_rows(self.path))
        self.assertEqual(first, again)

    def test_empty_quarantine_sends_nothing(self) -> None:
        self.write([])
        with mock.patch.dict(os.environ, {"TURNIP_PIPELINE_KEY": ""}):
            self.assertEqual(self.main(), 0, self.stderr.getvalue())
        self.assertEqual(self.farm.calls, 0)
        self.assertIn("nothing to quarantine", self.stdout.getvalue())

    def test_unlinked_segments_are_reported_not_fatal(self) -> None:
        segments = ids(4)
        self.write(rows(segments))
        # Segment 2 was deleted since the pull; segment 3 still links source 1.
        farm = FakeFarm(deleted=segments[2:3])
        self.assertEqual(self.main(farm), 0, self.stderr.getvalue())
        self.assertEqual(totals(self.stdout.getvalue()), counts(4, 1, 2, 0, 1))

    def test_every_segment_unlinked_is_not_fatal(self) -> None:
        segments = ids(2)
        self.write(rows(segments))
        farm = FakeFarm(deleted=segments)
        self.assertEqual(self.main(farm), 0, self.stderr.getvalue())
        self.assertEqual(farm.open, {})
        self.assertEqual(totals(self.stdout.getvalue()), counts(2, 1, 0, 0, 2))

    def test_a_source_split_across_bodies_counts_once_per_body(self) -> None:
        segments = ids(2)  # both cut from source 0
        self.write(rows(segments[:1]) + rows(segments[1:], reason="another reason"))
        self.assertEqual(self.main(), 0, self.stderr.getvalue())
        self.assertEqual(set(self.farm.open), {source_id(0)})
        self.assertEqual(totals(self.stdout.getvalue()), counts(2, 2, 1, 1))


class IdempotenceTest(ReportTestCase):
    def test_resubmission_changes_nothing(self) -> None:
        self.write(rows(ids(4)))
        self.assertEqual(self.main(), 0)
        before = dict(self.farm.open)
        self.assertEqual(self.main(), 0, self.stderr.getvalue())
        self.assertEqual(self.farm.open, before)
        self.assertEqual(totals(self.stdout.getvalue()), counts(4, 1, 0, 2))

    def test_a_rerun_after_a_failed_chunk_finishes_the_job(self) -> None:
        segments = ids(report_quarantine.MAX_SEGMENT_IDS + 1)
        self.write(rows(segments))
        # The first chunk lands; the second fails on every try.
        self.farm.down_after = 1
        self.assertEqual(self.main(), 2)
        first = {LINKS[s] for s in self.farm.posts[0]["segment_ids"]}
        self.assertEqual(set(self.farm.open), first)
        self.assertIn(
            "HTTP 503 INTERNAL (request 2 of 2; 1 sent", self.stderr.getvalue()
        )

        self.farm.down_after = None
        self.clear_logs()
        self.assertEqual(self.main(), 0, self.stderr.getvalue())
        self.assertEqual(set(self.farm.open), {LINKS[s] for s in segments})
        # The rerun resends the first chunk, whose sources are all open
        # already; the second chunk's segment may share one of them.
        second = {LINKS[s] for s in self.farm.posts[-1]["segment_ids"]}
        self.assertEqual(
            totals(self.stdout.getvalue()),
            counts(10001, 2, len(second - first), len(first) + len(second & first)),
        )

    def test_a_lost_answer_is_retried_into_already_open(self) -> None:
        self.write(rows(ids(3)))
        self.farm.drop_next_answer = True
        self.assertEqual(self.main(), 0, self.stderr.getvalue())
        self.assertEqual(len(self.farm.posts), 2)
        self.assertEqual(self.sleeps, [report_quarantine.RETRY_DELAYS_S[0]])
        self.assertEqual(totals(self.stdout.getvalue()), counts(3, 1, 0, 2))
        self.assertIn("TimeoutError (try 1 of 3); retrying", self.stderr.getvalue())


class FailureTest(ReportTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.segments = ids(3)
        self.write(rows(self.segments))

    def test_5xx_is_retried(self) -> None:
        self.farm.fail = [503]
        self.assertEqual(self.main(), 0, self.stderr.getvalue())
        self.assertEqual(set(self.farm.open), {LINKS[s] for s in self.segments})
        self.assertEqual(self.sleeps, [10.0])

    def test_5xx_on_every_try_is_an_error(self) -> None:
        self.farm.fail = [500, 502, 503]
        self.assertEqual(self.main(), 2)
        self.assertEqual(self.farm.calls, report_quarantine.ATTEMPTS)
        self.assertEqual(self.sleeps, list(report_quarantine.RETRY_DELAYS_S))
        self.assertIn(
            "error: POST /api/quarantine failed: HTTP 503", self.stderr.getvalue()
        )
        self.assertEqual(self.farm.open, {})
        self.assertLogsArePrivate()

    def test_422_fails_at_once_without_echoing_the_farm(self) -> None:
        self.farm.fail = [422]
        self.assertEqual(self.main(), 2)
        self.assertEqual(self.farm.calls, 1)
        self.assertEqual(self.sleeps, [])
        self.assertIn("HTTP 422 VALIDATION_ERROR", self.stderr.getvalue())
        # The farm's details carried an id and its message free text.
        self.assertLogsArePrivate()

    def test_wrong_pipeline_key(self) -> None:
        with mock.patch.dict(os.environ, {"TURNIP_PIPELINE_KEY": "wrong-key"}):
            self.assertEqual(self.main(), 2)
        self.assertEqual(self.farm.calls, 1)
        self.assertIn("HTTP 401 UNAUTHENTICATED", self.stderr.getvalue())
        self.assertNotIn("wrong-key", self.logs())
        self.assertEqual(self.farm.open, {})

    def test_other_4xx_fails_at_once(self) -> None:
        self.farm.fail = [404]
        self.assertEqual(self.main(), 2)
        self.assertEqual(self.farm.calls, 1)
        self.assertIn("HTTP 404 NOT_FOUND", self.stderr.getvalue())

    def test_unprintable_error_codes_are_dropped(self) -> None:
        def odd(
            method: str, url: str, headers: dict[str, str], body: bytes | None
        ) -> tuple[int, Any]:
            return 400, {"error": {"code": f"BAD {segment_id(0)}"}}

        code = report_quarantine.main(
            ["--quarantine", str(self.path)], transport=odd, sleep=self.sleeps.append
        )
        self.assertEqual(code, 2)
        self.assertIn("HTTP 400 (request 1 of 1", self.stderr.getvalue())
        self.assertLogsArePrivate()

    def test_an_inconsistent_answer_is_an_error(self) -> None:
        # The good answer for segments 0-2: 2 sources quarantined, 0 unlinked.
        def counts(**changed: Any) -> Any:
            return lambda a: {**a, "segments": {**a["segments"], **changed}}

        cases = {
            "names a source": lambda a: {**a, "quarantined": [source_id(0)]},
            "names an unknown": lambda a: {**a, "unknown": [segment_id(0)]},
            "lacks a list": lambda a: {k: v for k, v in a.items() if k != "unknown"},
            "lacks the counts": lambda a: {
                k: v for k, v in a.items() if k != "segments"
            },
            "lacks a count": lambda a: {**a, "segments": {"unlinked": 0}},
            "a negative count": counts(sources_already_open=-1),
            "a boolean count": counts(unlinked=False),
            "a float count": counts(sources_quarantined=2.0),
            "more unlinked than sent": counts(unlinked=4),
            "more sources than linked": counts(sources_already_open=2),
            "no source for linked segments": counts(sources_quarantined=0),
            "a source for no linked segment": counts(unlinked=3),
            "not an object": lambda a: [a],
            "empty": lambda a: None,
        }
        for name, tamper in cases.items():
            with self.subTest(name):
                farm = FakeFarm()
                farm.tamper = tamper
                self.clear_logs()
                self.assertEqual(self.main(farm), 2)
                # Not retried: the farm answered, just not as agreed.
                self.assertEqual(farm.calls, 1)
                self.assertIn("POST /api/quarantine answered", self.stderr.getvalue())
                self.assertLogsArePrivate()

    def test_missing_environment(self) -> None:
        with mock.patch.dict(os.environ, {"TURNIP_FARM_URL": ""}):
            self.assertEqual(self.main(), 2)
        self.assertIn("missing environment: TURNIP_FARM_URL", self.stderr.getvalue())
        self.assertEqual(self.farm.calls, 0)

    def test_plain_http_farm_is_refused(self) -> None:
        with mock.patch.dict(os.environ, {"TURNIP_FARM_URL": "http://farm.example"}):
            self.assertEqual(self.main(), 2)
        self.assertEqual(self.farm.calls, 0)


class RefusalTest(ReportTestCase):
    """A malformed file sends nothing at all, and the error names no id."""

    def assertRefused(self, data: Any, message: str) -> None:
        self.write(data)
        self.clear_logs()
        self.assertEqual(self.main(), 2)
        self.assertEqual(self.farm.calls, 0)
        self.assertIn(message, self.stderr.getvalue())
        self.assertIsNone(HEX64.search(self.logs()))
        self.assertIsNone(UUID.search(self.logs()))

    def test_malformed_rows_are_refused_before_any_request(self) -> None:
        good = rows(ids(2))
        cases: dict[str, tuple[Any, str]] = {
            "not an array": ({"rows": good}, "must contain a JSON array"),
            "an account id rides along": (
                [good[0], {**good[1], "user_id": "u-1"}],
                "row 1 must have exactly the keys",
            ),
            "a key missing": (
                [{"segment_id": segment_id(0), "reason": REASON}],
                "row 0 must have exactly the keys",
            ),
            "the old source_id key": (
                [{"source_id": source_id(0), "reason": REASON, "metrics": None}],
                "row 0 must have exactly the keys",
            ),
            "a listed twice segment": ([good[0], good[1], good[0]], "listed twice"),
            "an empty reason": ([{**good[0], "reason": "  "}], "row 0: reason"),
            "a long reason": ([{**good[0], "reason": "x" * 1001}], "row 0: reason"),
            "a list of metrics": ([{**good[0], "metrics": [1]}], "row 0: metrics"),
            "an uppercase id": (
                [{**good[0], "segment_id": segment_id(0).upper()}],
                "row 0: segment_id",
            ),
            "a source id": (
                [{**good[0], "segment_id": source_id(0)}],
                "row 0: segment_id",
            ),
            "a braced uuid": (
                [{**good[0], "segment_id": "{" + segment_id(0) + "}"}],
                "row 0: segment_id",
            ),
        }
        for name, (data, message) in cases.items():
            with self.subTest(name):
                self.assertRefused(data, message)

    def test_synthetic_ids_never_reach_the_farm(self) -> None:
        # The synthetic rehearsal's ids (make_smoke_fixture.py).
        self.assertRefused(rows(["source-a", "source-b"]), "row 0: segment_id")

    def test_reason_length_is_counted_as_the_farm_counts_it(self) -> None:
        # U+1F6F9 (skateboard) is two UTF-16 code units in JavaScript.
        self.write(rows(ids(1), reason="x" * 998 + "\U0001f6f9"))
        self.assertEqual(self.main(), 0, self.stderr.getvalue())
        self.farm = FakeFarm()
        self.assertRefused(rows(ids(1), reason="x" * 999 + "\U0001f6f9"), "reason")

    def test_unreadable_files_are_refused(self) -> None:
        self.path.write_text("[{")
        self.assertEqual(self.main(), 2)
        self.assertIn("JSONDecodeError", self.stderr.getvalue())
        self.path.unlink()
        self.assertEqual(self.main(), 2)
        self.assertIn("FileNotFoundError", self.stderr.getvalue())
        self.assertEqual(self.farm.calls, 0)

    def test_non_finite_metrics_are_refused(self) -> None:
        self.path.write_text(json.dumps(rows(ids(1), metrics={"delta": float("nan")})))
        self.assertEqual(self.main(), 2)
        self.assertEqual(self.farm.calls, 0)


class HttpTransportTest(ReportTestCase):
    """The real urllib transport against a local HTTP stand-in of the farm."""

    def start(self, handler: type[BaseHTTPRequestHandler]) -> str:
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        # Cleanups run last-in first-out: stop serving, join, then close.
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_address[1]}"

    def serve(self, farm: FakeFarm, raw: bytes | None = None) -> str:
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length)
                headers = {k: v for k, v in self.headers.items()}
                # FakeFarm routes on the public URL; the server is local.
                status, answer = farm.transport("POST", FARM + self.path, headers, body)
                data = raw if raw is not None else json.dumps(answer).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: Any) -> None:
                pass

        return self.start(Handler)

    def main_at(self, url: str) -> int:
        with mock.patch.dict(os.environ, {"TURNIP_FARM_URL": url}):
            return report_quarantine.main(
                ["--quarantine", str(self.path)], sleep=self.sleeps.append
            )

    def test_quarantine_over_http(self) -> None:
        segments = ids(report_quarantine.MAX_SEGMENT_IDS + 1)
        self.write(rows(segments))
        self.assertEqual(self.main_at(self.serve(self.farm)), 0, self.stderr.getvalue())
        self.assertEqual(set(self.farm.open), {LINKS[s] for s in segments})
        self.assertEqual(len(self.farm.posts), 2)

    def test_a_non_json_5xx_is_retried(self) -> None:
        self.write(rows(ids(2)))
        self.farm.fail = [502, 502, 502]
        url = self.serve(self.farm, raw=b"<html>Bad Gateway</html>")
        self.assertEqual(self.main_at(url), 2)
        self.assertEqual(self.farm.calls, 3)
        self.assertIn("HTTP 502 (request 1 of 1", self.stderr.getvalue())

    def test_redirects_are_never_followed(self) -> None:
        # Following one would send the pipeline key to the Location host.
        seen: list[str | None] = []

        class Elsewhere(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                seen.append(self.headers.get("Authorization"))
                self.send_response(500)
                self.end_headers()

            do_GET = do_POST

            def log_message(self, *args: Any) -> None:
                pass

        elsewhere = self.start(Elsewhere)

        class Redirect(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                self.send_response(307)
                self.send_header("Location", elsewhere + self.path)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *args: Any) -> None:
                pass

        self.write(rows(ids(2)))
        self.assertEqual(self.main_at(self.start(Redirect)), 2)
        self.assertIn("HTTP 307", self.stderr.getvalue())
        self.assertEqual(seen, [])
        self.assertEqual(self.sleeps, [])

    def test_a_dead_farm_is_retried_then_an_error(self) -> None:
        self.write(rows(ids(2)))
        # Nothing listens on a port once its server is closed.
        closed = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
        dead = f"http://127.0.0.1:{closed.server_address[1]}"
        closed.server_close()
        self.assertEqual(self.main_at(dead), 2)
        self.assertEqual(len(self.sleeps), report_quarantine.ATTEMPTS - 1)
        self.assertIn("farm request failed: URLError", self.stderr.getvalue())
        self.assertNotIn(dead, self.logs())


if __name__ == "__main__":
    unittest.main()
