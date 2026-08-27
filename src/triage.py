"""
Task 1 -- Intelligent ticket triage agent.

Public entry point: triage_ticket(subject, body, plan_tier=None) -> dict

Pipeline:
  1. Retrieve top KB chunks relevant to the ticket text (BM25, src/retrieval.py).
  2. Call the LLM with the ticket + retrieved context, forcing structured
     output via the emit_triage tool (src/prompts/triage_prompt.py).
  3. Validate the model's output against our own enums as a safety net
     (belt-and-braces on top of the JSON-schema `enum` constraint) and
     attach retrieval metadata for observability/debugging.
"""
from __future__ import annotations

from typing import Any, Optional

from src.config import PRODUCTS, PRODUCT_AREAS, CATEGORIES, URGENCY_LEVELS, RESPONDER_TEAMS, MAX_TOKENS_TRIAGE
from src.llm_client import call_tool
from src.prompts.triage_prompt import TRIAGE_SYSTEM_PROMPT, TRIAGE_TOOL_SCHEMA, VERSION as PROMPT_VERSION
from src.retrieval import search_kb


class TriageValidationError(ValueError):
    pass


def _build_user_content(subject: str, body: str, plan_tier: Optional[str], kb_results: list[dict]) -> str:
    kb_block = "\n\n".join(
        f"[KB excerpt {i+1}] source={r['source']} | section={r['heading_path']}\n{r['text']}"
        for i, r in enumerate(kb_results)
    ) or "(no relevant KB excerpts found)"

    return f"""## Incoming ticket

Subject: {subject}
Plan tier: {plan_tier or "unknown"}

Body:
{body}

## Retrieved knowledge-base excerpts (may or may not be relevant -- judge for yourself)

{kb_block}
"""


def _validate(result: dict[str, Any]) -> None:
    if result["product"] not in PRODUCTS:
        raise TriageValidationError(f"Unknown product: {result['product']}")
    if result["product_area"] not in PRODUCT_AREAS.get(result["product"], []):
        raise TriageValidationError(
            f"product_area '{result['product_area']}' is not valid for product '{result['product']}'"
        )
    if result["category"] not in CATEGORIES:
        raise TriageValidationError(f"Unknown category: {result['category']}")
    if result["urgency"] not in URGENCY_LEVELS:
        raise TriageValidationError(f"Unknown urgency: {result['urgency']}")
    if result["recommended_team"] not in RESPONDER_TEAMS:
        raise TriageValidationError(f"Unknown responder team: {result['recommended_team']}")
    if not (0 <= result["confidence"] <= 1):
        raise TriageValidationError(f"confidence out of range: {result['confidence']}")


def triage_ticket(
    subject: str,
    body: str,
    plan_tier: Optional[str] = None,
    top_k_kb: int = 3,
) -> dict[str, Any]:
    """Runs the full triage pipeline for a single raw ticket.

    Accepts subject/body as separate args (also works fine if you pass the
    whole ticket text as `body` with subject=""). Returns a dict matching
    the emit_triage schema plus a `_meta` block with retrieval + prompt
    version info for debugging/eval.
    """
    if not body or not body.strip():
        raise ValueError("Ticket body must not be empty.")

    kb_results = search_kb(f"{subject}\n{body}", top_k=top_k_kb)
    user_content = _build_user_content(subject, body, plan_tier, kb_results)

    result = call_tool(
        system_prompt=TRIAGE_SYSTEM_PROMPT,
        user_content=user_content,
        tool_schema=TRIAGE_TOOL_SCHEMA,
        max_tokens=MAX_TOKENS_TRIAGE,
    )
    _validate(result)

    result["_meta"] = {
        "prompt_version": PROMPT_VERSION,
        "kb_candidates_considered": [{"source": r["source"], "heading_path": r["heading_path"]} for r in kb_results],
    }
    return result


def triage_ticket_dict(ticket: dict[str, Any]) -> dict[str, Any]:
    """Convenience wrapper accepting a ticket-shaped dict (as in tickets.json
    or a raw {subject, body[, plan_tier]} payload)."""
    return triage_ticket(
        subject=ticket.get("subject", ""),
        body=ticket.get("body", ""),
        plan_tier=ticket.get("plan_tier"),
    )
