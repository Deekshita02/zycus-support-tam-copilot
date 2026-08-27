"""
Deterministic, rule-based stand-in for the real LLM call, used only when
USE_MOCK_LLM=1 is set in the environment.

Why this exists: it lets (a) CI run the eval harness on every commit without
needing a real ANTHROPIC_API_KEY secret [bonus requirement], and (b) anyone
reviewing this repo sanity-check the full pipeline plumbing (retrieval ->
prompt construction -> schema validation -> report generation) with zero
cost and zero network dependency. It is NOT a substitute for real model
grading -- eval_harness.py labels every result with which backend produced
it, and the design note calls this out explicitly as a known limitation.
"""
from __future__ import annotations

import re
from typing import Any

from src.config import PRODUCT_AREAS


def _guess_product_area(product: str, text: str) -> str:
    areas = PRODUCT_AREAS.get(product, ["Unknown"])
    text_lower = text.lower()
    for area in areas:
        if area.lower() in text_lower:
            return area
    return areas[0]


def mock_triage_response(user_content: str) -> dict[str, Any]:
    text = user_content.lower()

    product = "DataBridge Pro"
    for p in PRODUCT_AREAS:
        if p.lower() in text:
            product = p
            break

    if any(k in text for k in ["outage", "down", "critical", "production is down", "cannot access"]):
        urgency = "P1"
    elif any(k in text for k in ["47 users", "many users", "urgent", "blocking"]):
        urgency = "P2"
    elif any(k in text for k in ["minor", "cosmetic", "would be nice", "feature"]):
        urgency = "P4"
    else:
        urgency = "P3"

    if "invoice" in text or "billing" in text or "charge" in text:
        category = "Billing"
    elif "feature request" in text or "would be nice" in text or "please add" in text:
        category = "Feature Request"
    elif "how do i" in text or "how to" in text or "documentation" in text:
        category = "How-To"
    elif "slow" in text or "timeout" in text or "performance" in text:
        category = "Performance"
    elif "missing" in text or "lost data" in text or "corrupted" in text or "deleted" in text:
        category = "Data Loss"
    elif "integration" in text or "connector" in text or "sso" in text or "saml" in text:
        category = "Integration"
    elif "onboarding" in text or "new organisation" in text or "new org" in text or "setup" in text:
        category = "Onboarding"
    else:
        category = "Bug"

    confidence = 0.55 if len(user_content) < 300 else 0.85

    kb_found = "[kb excerpt 1]" in text
    kb_source, kb_excerpt = "", ""
    if kb_found:
        m = re.search(r"\[kb excerpt 1\] source=(\S+) \| section=([^\n]+)", user_content, re.IGNORECASE)
        if m:
            kb_source, kb_excerpt = m.group(1), m.group(2)

    return {
        "product": product,
        "product_area": _guess_product_area(product, text),
        "category": category,
        "urgency": urgency,
        "confidence": confidence,
        "reasoning": "[mock-llm] Rule-based classification from keyword matches in ticket text.",
        "kb_match_found": kb_found,
        "kb_match_source": kb_source,
        "kb_match_excerpt": kb_excerpt,
        "recommended_team": "Tier-2 Support Engineering" if category in ("Bug", "Performance", "Data Loss") else "Tier-1 Support",
        "draft_first_response": (
            "[mock-llm] Thanks for reaching out -- we've logged this and a member of our team "
            "will follow up shortly with next steps."
        ),
    }


def mock_brief_response(user_content: str) -> dict[str, Any]:
    quotes = re.findall(r'"\s*([^"]{5,200})"\s*\)', user_content)
    quotes = re.findall(r'\): "([^"]{3,200})"', user_content) or quotes

    open_risks = [
        {"risk_summary": f"[mock-llm] Signal detected: {q[:80]}", "evidence_quote": q, "source": "mock"}
        for q in quotes[:5]
    ]

    health_declining = "declining" in user_content.lower() or "at risk" in user_content.lower() or "churning" in user_content.lower()
    churn_level = "High" if "churning" in user_content.lower() else ("Medium" if health_declining else "Low")

    return {
        "executive_summary": (
            "[mock-llm] Deterministic placeholder summary generated without a live model call. "
            "See open_risks for the underlying verified signals."
        ),
        "open_risks": open_risks,
        "talking_points": [
            "[mock-llm] Review each flagged risk signal with the customer.",
            "[mock-llm] Confirm renewal timeline and outstanding blockers.",
        ],
        "churn_risk_level": churn_level,
    }


def mock_call_tool(tool_name: str, user_content: str) -> dict[str, Any]:
    if tool_name == "emit_triage":
        return mock_triage_response(user_content)
    if tool_name == "emit_brief":
        return mock_brief_response(user_content)
    raise ValueError(f"No mock implementation for tool '{tool_name}'")
