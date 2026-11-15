#!/usr/bin/env python3
"""Fixed retrain schedule: deleted data leaves the model within one cycle.

TRAINING_DESIGN.md §2.1, §8. Every dataset is a full pull, so data
deleted farm-side never reaches a new training run. But the nightly run
fine-tunes the champion, whose weights carry everything it was ever
trained on, and it only runs when more than 50 label rows are new. So
the farm job also retrains from scratch on a fixed schedule: once the
last full retrain is RETRAIN_INTERVAL_DAYS old, the next nightly run
trains from scratch whatever the label count, and promote.py --refresh
lets that challenger replace the champion unless it is clearly worse
(promote.py, "Refresh"). Until a from-scratch champion lands, every
nightly run tries again.

*(Judgment call on 30 days, TRAINING_DESIGN.md §10 Q12.)* Data deleted
right after a full retrain leaves the published model within about 30
days, plus the nights a refresh is refused by a gate.

Overdue alert: a refused refresh keeps the champion, and the next nightly
run retries. Once the scheduled retrain has been refused for
OVERDUE_ALERT_DAYS days in a row, every further refused night also sends
the fixture gate's Discord alert (webhook from TURNIP_DISCORD_WEBHOOK_URL),
counts only: the days overdue and the gate that refused tonight. Days
overdue count calendar days from the first refused night, today included,
so a night that failed for another reason does not reset them; only a
from-scratch champion landing does.

Subcommands
-----------
due --state FILE --today YYYY-MM-DD
    Prints {"due": bool, "days_since_full_retrain": int | null}. Due when
    the state file is absent (no full retrain recorded: the champion's
    history is unknown) or the last full retrain is at least
    RETRAIN_INTERVAL_DAYS before today.
record --state FILE --today YYYY-MM-DD --outcome outcome.json
    Records today as the last full retrain when pipeline.sh's outcome
    says this run trained from scratch and its challenger became the
    champion (promotion "promote" or "bootstrap"), which ends any
    refusal streak. When the outcome is a scheduled retrain (refresh)
    that did not land, records the first refused night if none is, and
    alerts once the retrain is OVERDUE_ALERT_DAYS overdue. Prints
    {"recorded": bool, "refused": bool, "days_overdue": int | null,
    "refusing_gate": str | null, "alert_status": str,
    "state_changed": bool}; the workflow stores the state when
    state_changed. refusing_gate is "fixture_gate" (a regression),
    "membership_inference" (archived for failing the check),
    "promotion" (archived as more than 2% worse) or "not_run" (the gates
    did not run: nothing to score, or a champion without metrics).

The state file is {"last_full_retrain": "YYYY-MM-DD", "refused_since":
"YYYY-MM-DD"}, either key optional but not both absent, written
atomically. The workflow passes today's UTC date, so the script reads no
clock.

Privacy: stdout, stderr and the alert carry dates, counts and gate names
only.

Exit codes: 0 = done (an undelivered alert included; it is reported in
alert_status), 2 = input error (fail closed): a malformed date, state or
outcome, or a recorded date after today.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fixture_gate import WEBHOOK_ENV, AlertSender, deliver_alert

RETRAIN_INTERVAL_DAYS = 30
# Refused nights in a row, the first included, before the overdue alert.
OVERDUE_ALERT_DAYS = 7
STATE_KEYS = frozenset({"last_full_retrain", "refused_since"})
EXIT_OK = 0
EXIT_ERROR = 2


class ScheduleError(Exception):
    """Input error: the schedule refuses to guess."""


def parse_date(value: Any, where: str) -> dt.date:
    if not isinstance(value, str):
        raise ScheduleError(f"{where} must be a YYYY-MM-DD date")
    try:
        date = dt.date.fromisoformat(value)
    except ValueError as e:
        raise ScheduleError(f"{where} must be a YYYY-MM-DD date") from e
    if date.isoformat() != value:
        raise ScheduleError(f"{where} must be a YYYY-MM-DD date")
    return date


def _load_json(path: Path, what: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ScheduleError(f"cannot read {what} {path}: {e}") from e


@dataclass(frozen=True)
class State:
    last_full_retrain: dt.date | None = None
    # The first refused night of the current overdue scheduled retrain.
    refused_since: dt.date | None = None


def read_state(state_path: Path, today: dt.date) -> State:
    if not state_path.exists():
        return State()
    state = _load_json(state_path, "state")
    if not isinstance(state, dict) or not state or not set(state) <= STATE_KEYS:
        raise ScheduleError(
            f"state {state_path} must hold last_full_retrain, refused_since or both"
        )
    dates = {k: parse_date(v, k) for k, v in state.items()}
    if any(d > today for d in dates.values()):
        raise ScheduleError("a recorded date is after today")
    return State(dates.get("last_full_retrain"), dates.get("refused_since"))


def write_state(state_path: Path, state: State) -> None:
    data = {
        k: d.isoformat()
        for k, d in (
            ("last_full_retrain", state.last_full_retrain),
            ("refused_since", state.refused_since),
        )
        if d is not None
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(state_path)


def due(state_path: Path, today: dt.date) -> dict[str, Any]:
    last = read_state(state_path, today).last_full_retrain
    if last is None:
        return {"due": True, "days_since_full_retrain": None}
    days = (today - last).days
    return {"due": days >= RETRAIN_INTERVAL_DAYS, "days_since_full_retrain": days}


def _check_outcome(outcome: Any) -> dict[str, Any]:
    if not isinstance(outcome, dict):
        raise ScheduleError("outcome must be a JSON object")
    for key, kind in (
        ("from_scratch", bool),
        ("refresh", bool),
        ("promotion", str),
        ("fixture_gate", str),
        ("membership_inference", str),
    ):
        if not isinstance(outcome.get(key), kind):
            raise ScheduleError(f"outcome needs {key} ({kind.__name__})")
    return outcome


def should_record(outcome: dict[str, Any]) -> bool:
    """A from-scratch challenger that became the champion."""
    return bool(outcome["from_scratch"]) and outcome["promotion"] in (
        "promote",
        "bootstrap",
    )


def refusing_gate(outcome: dict[str, Any]) -> str:
    """The gate that kept a challenger from becoming the champion."""
    if outcome["fixture_gate"] == "regression":
        return "fixture_gate"
    if outcome["promotion"] == "archive":
        if outcome["membership_inference"] == "fail":
            return "membership_inference"
        return "promotion"
    return "not_run"


def overdue_alert_content(days_overdue: int, gate: str) -> str:
    """The alert text: counts and the gate's name, nothing else."""
    return "\n".join(
        [
            "**turnip-ml: scheduled retrain overdue**",
            (
                f"refused {days_overdue} day(s) in a row "
                f"(alert from {OVERDUE_ALERT_DAYS}); refusing gate: `{gate}`"
            ),
            "the champion stays; the next nightly run retrains from scratch again",
        ]
    )


def record(
    state_path: Path,
    today: dt.date,
    outcome_path: Path,
    webhook_url: str | None = None,
    send: AlertSender | None = None,
) -> dict[str, Any]:
    outcome = _check_outcome(_load_json(outcome_path, "outcome"))
    state = read_state(state_path, today)
    result: dict[str, Any] = {
        "recorded": False,
        "refused": False,
        "days_overdue": None,
        "refusing_gate": None,
        "alert_status": "not-needed",
        "state_changed": False,
    }
    if should_record(outcome):
        write_state(state_path, State(last_full_retrain=today))
        result.update(recorded=True, state_changed=True)
    elif outcome["refresh"]:
        refused_since = state.refused_since or today
        if state.refused_since is None:
            write_state(state_path, State(state.last_full_retrain, refused_since=today))
            result["state_changed"] = True
        days_overdue = (today - refused_since).days + 1
        gate = refusing_gate(outcome)
        result.update(refused=True, days_overdue=days_overdue, refusing_gate=gate)
        if days_overdue >= OVERDUE_ALERT_DAYS:
            result["alert_status"] = deliver_alert(
                overdue_alert_content(days_overdue, gate),
                webhook_url,
                send,
                who="retrain_schedule",
            )
    return result


def main(argv: list[str] | None = None, send: AlertSender | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("due", "record"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--state", required=True, type=Path)
        cmd.add_argument("--today", required=True)
        if name == "record":
            cmd.add_argument("--outcome", required=True, type=Path)
    args = p.parse_args(argv)
    try:
        today = parse_date(args.today, "--today")
        if args.command == "due":
            result: dict[str, Any] = due(args.state, today)
        else:
            result = record(
                args.state, today, args.outcome, os.environ.get(WEBHOOK_ENV), send
            )
    except (ScheduleError, OSError) as e:
        print(f"retrain_schedule.py: error: {e}", file=sys.stderr)
        return EXIT_ERROR
    print(json.dumps(result, sort_keys=True))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
