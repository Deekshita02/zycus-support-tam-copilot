"""
Loads the mock tickets/accounts datasets and provides small, well-tested
join/filter helpers. No LLM calls here -- kept pure so it's cheaply testable.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Optional

from src.config import TICKETS_PATH, ACCOUNTS_PATH


def _parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


@lru_cache(maxsize=1)
def load_tickets() -> list[dict]:
    with open(TICKETS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_accounts() -> list[dict]:
    with open(ACCOUNTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _account_index() -> dict[str, dict]:
    return {a["account_id"]: a for a in load_accounts()}


@lru_cache(maxsize=1)
def dataset_reference_time() -> datetime:
    """This is a static, synthetic dataset (all tickets fall in a fixed
    ~3-month window), not a live feed. Anchoring "last N days" to the
    wall-clock `datetime.now()` would make every "recent tickets" query
    return nothing once the real calendar date moves past the dataset's
    date range. Instead we anchor to the latest `created_at` timestamp
    actually present in the data, so "last 90 days" always means the most
    recent 90 days *of the dataset* -- the behaviour a TAM tool would want
    if pointed at a frozen data export. Callers can still pass an explicit
    `reference_time` to override this (e.g. in tests, or once wired to a
    live ticket feed where real `now()` is correct)."""
    return max(_parse_iso(t["created_at"]) for t in load_tickets())


def get_account(account_id: str) -> Optional[dict]:
    """Returns the account record, or None if it doesn't exist in the mock
    dataset. Callers MUST handle the None case -- README notes ticket
    account_ids don't always resolve to a real account."""
    return _account_index().get(account_id)


def get_account_tickets(account_id: str, days: int = 90, reference_time: Optional[datetime] = None) -> list[dict]:
    """All tickets for an account created within the last `days` days of
    `reference_time` (defaults to the dataset's own latest timestamp -- see
    dataset_reference_time() docstring for why)."""
    reference_time = reference_time or dataset_reference_time()
    cutoff = reference_time - timedelta(days=days)
    return [
        t for t in load_tickets()
        if t.get("account_id") == account_id and _parse_iso(t["created_at"]) > cutoff
    ]


def get_ticket(ticket_id: str) -> Optional[dict]:
    for t in load_tickets():
        if t["ticket_id"] == ticket_id:
            return t
    return None
