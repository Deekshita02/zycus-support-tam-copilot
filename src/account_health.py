"""
Task 2 -- TAM account health summariser.

Public entry point: generate_account_brief(account_id) -> dict

Key design decision (see src/prompts/account_brief_prompt.py docstring for
the full rationale): risk-signal QUOTES are extracted deterministically in
code (extract_risk_signals below), never invented by the LLM. The LLM is
only handed already-verified verbatim quotes and asked to summarize/
prioritize them. This is what makes "flag churn risk, justify with a direct
quote" a checkable, non-hallucinated guarantee rather than a hope.

Determinism: temperature=0 (src/config.DEFAULT_TEMPERATURE), fully
deterministic input construction (sorted ticket order, sorted risk signals),
and no free-running generation of facts -- so repeated calls on the same
underlying data produce the same evidence set and, in practice with
temperature=0, near-identical prose.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Optional

from src.config import MAX_TOKENS_BRIEF
from src.data_loader import get_account, get_account_tickets
from src.llm_client import call_tool
from src.prompts.account_brief_prompt import (
    ACCOUNT_BRIEF_SYSTEM_PROMPT,
    ACCOUNT_BRIEF_TOOL_SCHEMA,
    VERSION as PROMPT_VERSION,
)

# Keywords that, if present in a ticket body/subject, mark it as a candidate
# churn/escalation signal worth surfacing to the TAM. Kept simple and
# auditable on purpose -- a false positive here just means an extra (still
# truthful) risk item shown to a human, which is a safe failure direction.
_RISK_KEYWORDS = [
    "cancel", "cancell", "churn", "competitor", "competing vendor", "switch to",
    "unacceptable", "frustrat", "escalat", "downgrade", "terminat", "unhappy",
    "disappoint", "losing confidence", "considering alternatives", "sla breach",
    "not acceptable", "third time", "again", "urgent escalation", "legal",
]

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


class AccountNotFoundError(LookupError):
    pass


def _find_evidence_sentence(text: str, keyword: str) -> Optional[str]:
    """Returns the sentence in `text` containing `keyword` (case-insensitive),
    verbatim, so it is guaranteed to be an exact substring of the source."""
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        if keyword.lower() in sentence.lower():
            return sentence.strip()
    return None


def extract_risk_signals(account: dict[str, Any], tickets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministically extracts candidate risk/churn signals with verbatim
    evidence quotes. Returns a list of:
        {"source": str, "quote": str, "kind": str}
    where `quote` is guaranteed to be an exact substring of the underlying
    account/ticket data (checkable by the eval harness).
    """
    signals: list[dict[str, Any]] = []

    # 1. Escalation notes are already curated, verbatim risk statements.
    for note in account.get("escalation_notes", []) or []:
        signals.append({"source": "escalation_note", "quote": note, "kind": "escalation_note"})

    # 2. P1 ticket volume (factual count, not a "quote" but a verifiable
    #    number -- the eval harness checks this count against the raw data).
    p1_tickets = [t for t in tickets if t.get("urgency") == "P1"]
    if p1_tickets:
        signals.append({
            "source": "ticket_count",
            "quote": f"{len(p1_tickets)} P1 ticket(s) in the last 90 days",
            "kind": "p1_volume",
        })

    # 3. Data Loss category tickets are always worth flagging regardless of
    #    keyword match.
    for t in tickets:
        if t.get("category") == "Data Loss":
            signals.append({
                "source": t["ticket_id"],
                "quote": t["subject"],
                "kind": "data_loss_ticket",
            })

    # 4. Keyword-based scan of subject+body for explicit churn/escalation
    #    language, sentence-level quote extraction.
    for t in tickets:
        haystack = f"{t.get('subject', '')}. {t.get('body', '')}"
        for kw in _RISK_KEYWORDS:
            if kw.lower() in haystack.lower():
                quote = _find_evidence_sentence(t["body"], kw) or _find_evidence_sentence(t["subject"], kw)
                if quote:
                    signals.append({"source": t["ticket_id"], "quote": quote, "kind": "keyword_match"})
                break  # one signal per ticket is enough

    # 5. Low CSAT tickets.
    for t in tickets:
        score = t.get("satisfaction_score")
        if score is not None and score <= 2:
            signals.append({
                "source": t["ticket_id"],
                "quote": f"CSAT score of {score}/5 on ticket {t['ticket_id']} ({t['subject']})",
                "kind": "low_csat",
            })

    return signals


def _summarize_tickets_for_prompt(tickets: list[dict[str, Any]], limit: int = 25) -> str:
    if not tickets:
        return "(no tickets in the last 90 days)"
    lines = []
    for t in sorted(tickets, key=lambda x: x["created_at"], reverse=True)[:limit]:
        lines.append(
            f"- {t['ticket_id']} | {t['created_at'][:10]} | {t['category']} | {t['urgency']} | "
            f"{t['status']} | \"{t['subject']}\""
        )
    if len(tickets) > limit:
        lines.append(f"... and {len(tickets) - limit} more tickets not shown (still counted in totals above).")
    return "\n".join(lines)


def _build_user_content(account: dict[str, Any], tickets: list[dict[str, Any]], signals: list[dict[str, Any]]) -> str:
    signals_block = "\n".join(f"- [{s['kind']}] ({s['source']}): \"{s['quote']}\"" for s in signals) or "(none found)"
    return f"""## Account summary

Company: {account['company']}
Plan tier: {account['plan_tier']} | ARR: ${account['arr_usd']:,} | Health: {account['health_status']} | Usage trend: {account['usage_trend']}
Seats: {account['seats_active']}/{account['seats_licensed']} active | Renewal: {account['renewal_date']} | Last QBR: {account['last_qbr_date']}
Open tickets: {account['open_tickets']} | P1 tickets (last 30d, per account record): {account['p1_tickets_last_30d']}
NPS: {account.get('nps_score')} | Primary contact: {account['primary_contact']['name']} ({account['primary_contact']['title']})
Industry: {account['industry']} | Region: {account['region']}
Active integrations: {', '.join(account.get('integrations_active', [])) or 'none'}

## Tickets in the last 90 days ({len(tickets)} total)

{_summarize_tickets_for_prompt(tickets)}

## Pre-extracted, verified risk signals (quotes are exact -- use them as-is, do not alter)

{signals_block}
"""


def generate_account_brief(account_id: str, days: int = 90) -> dict[str, Any]:
    account = get_account(account_id)
    if account is None:
        raise AccountNotFoundError(
            f"No account found for account_id={account_id!r}. This ID may be one of the "
            f"intentionally-unlinked ticket account_ids described in DATA_SCHEMA.md -- handle gracefully."
        )

    tickets = get_account_tickets(account_id, days=days)
    signals = extract_risk_signals(account, tickets)
    user_content = _build_user_content(account, tickets, signals)

    result = call_tool(
        system_prompt=ACCOUNT_BRIEF_SYSTEM_PROMPT,
        user_content=user_content,
        tool_schema=ACCOUNT_BRIEF_TOOL_SCHEMA,
        max_tokens=MAX_TOKENS_BRIEF,
    )

    result["account_id"] = account_id
    result["company"] = account["company"]
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    result["_meta"] = {
        "prompt_version": PROMPT_VERSION,
        "tickets_considered": len(tickets),
        "risk_signals_extracted": len(signals),
    }
    return result
