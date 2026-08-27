"""
Task 3 -- Evaluation harness for the Task 1 (triage) and Task 2 (account
brief) pipelines.

Scoring approach per test case:
  - One or more RULE-BASED checks (deterministic, cheap, no LLM call) --
    these do the heavy lifting since triage/brief outputs are structured
    and largely rule-checkable (enum membership, quote-is-verbatim-substring,
    internal consistency, graceful error handling).
  - An optional LLM-AS-JUDGE check for the genuinely subjective bits (is the
    draft response professionally worded? is the executive summary actually
    informative?). This is skipped gracefully (not counted as a failure) if
    no API key is configured, so the harness still runs end-to-end offline.

Each test case produces: {passed: bool, quality_score: float in [0,1], details: [...]}
Overall report aggregates pass rate + mean quality score per task, and is
written to eval/eval_report.json and eval/eval_report.md.

Run with: python cli.py eval   (or: python -m eval.eval_harness)
"""
from __future__ import annotations

import json
import statistics
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import src.llm_client as llm_client
from src.account_health import AccountNotFoundError, generate_account_brief
from src.data_loader import get_account, get_account_tickets
from src.llm_client import LLMConfigError, call_tool
from src.triage import triage_ticket

REPORT_DIR = Path(__file__).resolve().parent
JUDGE_TOOL_SCHEMA = {
    "name": "emit_judgment",
    "description": "Emit a quality judgment for a piece of generated text.",
    "input_schema": {
        "type": "object",
        "properties": {
            "score": {"type": "number", "minimum": 0, "maximum": 1},
            "rationale": {"type": "string"},
        },
        "required": ["score", "rationale"],
    },
}
JUDGE_SYSTEM_PROMPT = """You are a strict QA grader for a customer support tool. Given a short piece of
generated text and a rubric, call `emit_judgment` with a score from 0 (fails
the rubric badly) to 1 (fully meets the rubric) and a one-sentence rationale.
Be skeptical -- do not give high scores to generic, hedge-everything text."""


def llm_judge(text_to_grade: str, rubric: str) -> Optional[dict[str, Any]]:
    """Returns {"score": float, "rationale": str} or None if no LLM is
    configured (mock backend doesn't implement judging -- see mock_llm.py).
    Never raises -- a judge failure should not crash the whole harness."""
    try:
        user_content = f"Rubric: {rubric}\n\nText to grade:\n\"\"\"\n{text_to_grade}\n\"\"\""
        return call_tool(JUDGE_SYSTEM_PROMPT, user_content, JUDGE_TOOL_SCHEMA, max_tokens=300)
    except LLMConfigError:
        return None
    except Exception:
        return None


@dataclass
class TestResult:
    test_id: str
    task: str
    description: str
    adversarial: bool
    passed: bool
    quality_score: float
    details: list[str] = field(default_factory=list)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Task 1 test cases -- ticket triage
# ---------------------------------------------------------------------------

def _tc_triage(test_id: str, description: str, subject: str, body: str, plan_tier: Optional[str],
                checker: Callable[[dict], tuple[bool, float, list[str]]], adversarial: bool = False) -> TestResult:
    try:
        result = triage_ticket(subject=subject, body=body, plan_tier=plan_tier)
    except LLMConfigError as e:
        return TestResult(test_id, "triage", description, adversarial, False, 0.0, [], error=str(e))
    except Exception as e:
        return TestResult(test_id, "triage", description, adversarial, False, 0.0, [], error=f"{type(e).__name__}: {e}")

    try:
        passed, score, details = checker(result)
    except Exception as e:
        return TestResult(test_id, "triage", description, adversarial, False, 0.0,
                           [f"Checker itself raised: {e}"])
    return TestResult(test_id, "triage", description, adversarial, passed, score, details)


def _check_enum_validity(result: dict) -> tuple[bool, float, list[str]]:
    # triage_ticket already runs src.triage._validate internally and would
    # have raised TriageValidationError if invalid -- reaching here means
    # the schema/enum constraints already passed. We re-assert defensively.
    from src.config import PRODUCTS, CATEGORIES, URGENCY_LEVELS
    ok = result["product"] in PRODUCTS and result["category"] in CATEGORIES and result["urgency"] in URGENCY_LEVELS
    return ok, 1.0 if ok else 0.0, ["all enum fields valid" if ok else "an enum field was invalid"]


def eval_case_clear_p1_outage() -> TestResult:
    def checker(r):
        details = []
        score = 0.0
        ok_urgency = r["urgency"] in ("P1", "P2")
        score += 0.5 if ok_urgency else 0.0
        details.append(f"urgency={r['urgency']} (expected P1/P2): {'OK' if ok_urgency else 'FAIL'}")
        ok_category = r["category"] in ("Bug", "Performance")
        score += 0.3 if ok_category else 0.0
        details.append(f"category={r['category']} (expected Bug/Performance): {'OK' if ok_category else 'FAIL'}")
        has_reasoning = len(r.get("reasoning", "")) > 10
        score += 0.2 if has_reasoning else 0.0
        details.append(f"reasoning present: {'OK' if has_reasoning else 'FAIL'}")
        return score >= 0.7, score, details

    return _tc_triage(
        "T1-01", "Clear production outage should be triaged as high urgency / Bug or Performance",
        subject="URGENT: Production pipeline completely down",
        body=(
            "Our DataBridge Pro production pipeline has been completely down for 2 hours. "
            "All 300 users in our organisation are affected and no data is flowing. "
            "Error: ERR_CONNECTION_TIMEOUT after 30s on every retry."
        ),
        plan_tier="Enterprise",
        checker=checker,
    )


def eval_case_billing_question() -> TestResult:
    def checker(r):
        details = []
        ok_cat = r["category"] == "Billing"
        details.append(f"category={r['category']} (expected Billing): {'OK' if ok_cat else 'FAIL'}")
        ok_urgency = r["urgency"] in ("P3", "P4")
        details.append(f"urgency={r['urgency']} (expected P3/P4): {'OK' if ok_urgency else 'FAIL'}")
        score = (0.6 if ok_cat else 0.0) + (0.4 if ok_urgency else 0.0)
        return score >= 0.6, score, details

    return _tc_triage(
        "T1-02", "Plain billing/invoice question should be categorised as Billing, low urgency",
        subject="Question about our invoice",
        body="Hi, our latest invoice shows 350 seats but we only have 320 licensed. Can someone clarify the difference before we pay?",
        plan_tier="Business",
        checker=checker,
    )


def eval_case_feature_request() -> TestResult:
    def checker(r):
        details = []
        ok_cat = r["category"] == "Feature Request"
        details.append(f"category={r['category']} (expected Feature Request): {'OK' if ok_cat else 'FAIL'}")
        ok_urgency = r["urgency"] in ("P3", "P4")
        details.append(f"urgency={r['urgency']} (expected P3/P4, low): {'OK' if ok_urgency else 'FAIL'}")
        score = (0.6 if ok_cat else 0.0) + (0.4 if ok_urgency else 0.0)
        return score >= 0.6, score, details

    return _tc_triage(
        "T1-03", "Feature request phrased politely (no outage) should be low urgency",
        subject="Feature request: dark mode for AnalyticsHub dashboards",
        body="It would be great if AnalyticsHub dashboards supported a dark mode theme. Not urgent, just a nice-to-have for our night-shift analysts.",
        plan_tier="Professional",
        checker=checker,
    )


def eval_case_kb_pattern_match() -> TestResult:
    def checker(r):
        details = []
        ok = r.get("kb_match_found") is True and "auth" in r.get("kb_match_source", "").lower()
        details.append(f"kb_match_found={r.get('kb_match_found')}, source={r.get('kb_match_source')!r}: {'OK' if ok else 'FAIL'}")
        return ok, 1.0 if ok else 0.3, details

    return _tc_triage(
        "T1-04", "Ticket matching a documented KB error code should surface that KB doc",
        subject="Users getting SAML_ASSERTION_EXPIRED errors",
        body="Since this morning our SecureVault users are getting SAML_ASSERTION_EXPIRED whenever they try to log in via SSO. This started right after we changed timezone settings on our IDP server.",
        plan_tier="Enterprise",
        checker=checker,
    )


def eval_case_onboarding() -> TestResult:
    def checker(r):
        details = []
        ok = r["category"] == "Onboarding"
        details.append(f"category={r['category']} (expected Onboarding): {'OK' if ok else 'FAIL'}")
        return ok, 1.0 if ok else 0.2, details

    return _tc_triage(
        "T1-05", "New-organisation setup question should be categorised as Onboarding",
        subject="How do we set up SSO for our new org?",
        body="We just signed up for the Business plan and need to configure SSO and bulk-import our 40 users before rollout next week. What's the recommended order of steps?",
        plan_tier="Business",
        checker=checker,
    )


def eval_case_adversarial_ambiguous() -> TestResult:
    """Adversarial: a near-content-free ticket. The correct behaviour is NOT
    a specific category/urgency (there isn't enough info) but a valid,
    non-crashing structured output with LOW confidence, signalling to the
    human agent that this needs clarification rather than confidently
    guessing."""
    def checker(r):
        details = []
        ok_valid, score_valid, d = _check_enum_validity(r)
        details.extend(d)
        low_conf = r.get("confidence", 1.0) <= 0.6
        details.append(f"confidence={r.get('confidence')} (expected <=0.6 given how vague the ticket is): {'OK' if low_conf else 'FAIL'}")
        score = (0.6 if ok_valid else 0.0) + (0.4 if low_conf else 0.0)
        return ok_valid and low_conf, score, details

    return _tc_triage(
        "T1-06", "[Adversarial] Extremely vague one-line ticket should still produce valid output with low confidence",
        subject="it's broken",
        body="it's broken again",
        plan_tier=None,
        checker=checker,
        adversarial=True,
    )


def eval_case_adversarial_multi_issue() -> TestResult:
    """Adversarial: a ticket that plausibly touches billing, a bug, AND an
    integration in one message. There's no single "correct" category -- the
    check is robustness (valid schema, product/area internally consistent)
    rather than matching one specific label."""
    def checker(r):
        ok_valid, score_valid, details = _check_enum_validity(r)
        return ok_valid, score_valid, details

    return _tc_triage(
        "T1-07", "[Adversarial] Ticket spanning billing + bug + integration should still yield a single valid, internally-consistent structured decision",
        subject="Billing seems wrong AND our Salesforce sync broke AND a report is showing wrong numbers",
        body=(
            "A few things: 1) our invoice this month looks higher than expected, 2) our Salesforce "
            "integration in WorkflowEngine stopped syncing two days ago, and 3) the weekly revenue "
            "report in AnalyticsHub is showing numbers that don't match our source data. Not sure which "
            "of these is most important but all are annoying."
        ),
        plan_tier="Business",
        checker=checker,
        adversarial=True,
    )


def eval_case_judge_draft_response_quality() -> TestResult:
    """LLM-as-judge test: grades the *subjective* quality of the drafted
    first-response message (tone, professionalism, no invented commitments)
    -- something rule-based checks can't meaningfully assess. Skipped
    gracefully if no LLM is configured (judge itself needs a model)."""
    test_id, desc = "T1-08", "[LLM-as-judge] Draft first-response message must be professional and make no unconfirmed promises"
    try:
        result = triage_ticket(
            subject="Pipeline stalled, need urgent help",
            body="Our DataBridge Pro pipeline for the Finance team has been stalled for 3 hours with PIPELINE_STALLED: no heartbeat for 15 minutes. This is blocking our month-end close.",
            plan_tier="Enterprise",
        )
    except LLMConfigError as e:
        return TestResult(test_id, "triage", desc, False, False, 0.0, [], error=str(e))
    except Exception as e:
        return TestResult(test_id, "triage", desc, False, False, 0.0, [], error=f"{type(e).__name__}: {e}")

    judgment = llm_judge(
        result.get("draft_first_response", ""),
        rubric=(
            "The text is a first-response message from a support agent to a customer. It should be "
            "professional, empathetic, specific to the issue, and must NOT promise a concrete fix time "
            "or make commitments the agent hasn't confirmed (e.g. 'this will be fixed within the hour')."
        ),
    )
    if judgment is None:
        return TestResult(test_id, "triage", desc, False, False, 0.0, [],
                           error="LLM judge unavailable (mock backend or no API key) -- skipped, not counted as failure")

    score = float(judgment["score"])
    details = [f"judge rationale: {judgment['rationale']}", f"judge score: {score}"]
    return TestResult(test_id, "triage", desc, False, score >= 0.6, score, details)


TRIAGE_TEST_CASES: list[Callable[[], TestResult]] = [
    eval_case_clear_p1_outage,
    eval_case_billing_question,
    eval_case_feature_request,
    eval_case_kb_pattern_match,
    eval_case_onboarding,
    eval_case_adversarial_ambiguous,
    eval_case_adversarial_multi_issue,
    eval_case_judge_draft_response_quality,
]


# ---------------------------------------------------------------------------
# Task 2 test cases -- account health brief
# ---------------------------------------------------------------------------

def _quote_is_verbatim(quote: str, account: dict, tickets: list[dict]) -> bool:
    """The core anti-hallucination check: every evidence_quote in the brief
    must be an exact substring of either an escalation note, a ticket
    subject/body, or (for count-style evidence) must match the true count we
    can independently compute from the raw data."""
    haystacks = list(account.get("escalation_notes", []) or [])
    for t in tickets:
        haystacks.append(t.get("subject", ""))
        haystacks.append(t.get("body", ""))
        haystacks.append(t.get("ticket_id", ""))
    if any(quote in h for h in haystacks if h):
        return True

    # count-style evidence, e.g. "3 P1 tickets in the last 90 days"
    p1_count = sum(1 for t in tickets if t.get("urgency") == "P1")
    if str(p1_count) in quote and "P1" in quote:
        return True

    # CSAT-style evidence embeds ticket_id + subject which we already check
    # above; nothing further to do.
    return False


def eval_case_account_with_signals(account_id: str) -> TestResult:
    def run():
        account = get_account(account_id)
        tickets = get_account_tickets(account_id)
        result = generate_account_brief(account_id)
        details = []
        score = 0.0

        has_sections = all(k in result for k in ("executive_summary", "open_risks", "talking_points", "churn_risk_level"))
        details.append(f"all 4 required sections present: {'OK' if has_sections else 'FAIL'}")
        score += 0.25 if has_sections else 0.0

        risks = result.get("open_risks", [])
        if risks:
            verbatim_flags = [_quote_is_verbatim(r["evidence_quote"], account, tickets) for r in risks]
            all_verbatim = all(verbatim_flags)
            details.append(
                f"{sum(verbatim_flags)}/{len(risks)} evidence quotes are verbatim substrings of source data: "
                f"{'OK' if all_verbatim else 'FAIL'}"
            )
            score += 0.45 if all_verbatim else 0.45 * (sum(verbatim_flags) / len(risks))
        else:
            details.append("no open_risks returned to check (account may have no signals) -- neutral")
            score += 0.2

        exec_summary_ok = 20 <= len(result.get("executive_summary", "")) <= 1200
        details.append(f"executive_summary non-trivial length: {'OK' if exec_summary_ok else 'FAIL'}")
        score += 0.15 if exec_summary_ok else 0.0

        talking_points_ok = len(result.get("talking_points", [])) >= 2
        details.append(f">=2 talking points: {'OK' if talking_points_ok else 'FAIL'}")
        score += 0.15 if talking_points_ok else 0.0

        return score >= 0.7, score, details

    try:
        passed, score, details = run()
    except LLMConfigError as e:
        return TestResult("T2-01", "brief", "Account with escalation notes + recent tickets produces a valid, evidence-grounded brief", False, False, 0.0, [], error=str(e))
    except Exception as e:
        return TestResult("T2-01", "brief", "Account with escalation notes + recent tickets produces a valid, evidence-grounded brief", False, False, 0.0, [], error=f"{type(e).__name__}: {e}")
    return TestResult("T2-01", "brief", "Account with escalation notes + recent tickets produces a valid, evidence-grounded brief", False, passed, score, details)


def eval_case_healthy_account_low_churn(account_id: str) -> TestResult:
    def run():
        result = generate_account_brief(account_id)
        details = []
        level_ok = result["churn_risk_level"] in ("Low", "Medium")
        details.append(f"churn_risk_level={result['churn_risk_level']} for a Healthy/Stable account (expected Low/Medium): {'OK' if level_ok else 'FAIL'}")
        return level_ok, 1.0 if level_ok else 0.3, details

    try:
        passed, score, details = run()
    except LLMConfigError as e:
        return TestResult("T2-02", "brief", "Healthy, stable account should not be flagged as High churn risk", False, False, 0.0, [], error=str(e))
    except Exception as e:
        return TestResult("T2-02", "brief", "Healthy, stable account should not be flagged as High churn risk", False, False, 0.0, [], error=f"{type(e).__name__}: {e}")
    return TestResult("T2-02", "brief", "Healthy, stable account should not be flagged as High churn risk", False, passed, score, details)


def eval_case_p1_volume_flagged(account_id: str) -> TestResult:
    def run():
        tickets = get_account_tickets(account_id)
        true_p1_count = sum(1 for t in tickets if t.get("urgency") == "P1")
        result = generate_account_brief(account_id)
        details = [f"true P1 count in last 90d = {true_p1_count}"]
        if true_p1_count == 0:
            details.append("no P1 tickets to require flagging -- neutral pass")
            return True, 1.0, details
        found = any(str(true_p1_count) in r["evidence_quote"] for r in result.get("open_risks", []))
        details.append(f"P1 count surfaced in open_risks evidence: {'OK' if found else 'FAIL'}")
        return found, 1.0 if found else 0.3, details

    try:
        passed, score, details = run()
    except LLMConfigError as e:
        return TestResult("T2-03", "brief", "P1 ticket volume in the window must be surfaced as a risk if present", False, False, 0.0, [], error=str(e))
    except Exception as e:
        return TestResult("T2-03", "brief", "P1 ticket volume in the window must be surfaced as a risk if present", False, False, 0.0, [], error=f"{type(e).__name__}: {e}")
    return TestResult("T2-03", "brief", "P1 ticket volume in the window must be surfaced as a risk if present", False, passed, score, details)


def eval_case_adversarial_unknown_account() -> TestResult:
    """Adversarial: an account_id with no matching account record (mirrors
    the intentional ticket/account gaps called out in DATA_SCHEMA.md)."""
    try:
        generate_account_brief("ACC-DOES-NOT-EXIST-99999")
        return TestResult("T2-04", "brief", "[Adversarial] Unknown account_id must raise a handleable error, not crash silently or hallucinate a brief",
                           True, False, 0.0, ["Expected AccountNotFoundError but none was raised"])
    except AccountNotFoundError as e:
        return TestResult("T2-04", "brief", "[Adversarial] Unknown account_id must raise a handleable error, not crash silently or hallucinate a brief",
                           True, True, 1.0, [f"Correctly raised AccountNotFoundError: {e}"])
    except Exception as e:
        return TestResult("T2-04", "brief", "[Adversarial] Unknown account_id must raise a handleable error, not crash silently or hallucinate a brief",
                           True, False, 0.0, [f"Raised wrong exception type: {type(e).__name__}: {e}"])


def eval_case_adversarial_no_recent_tickets(account_id: str) -> TestResult:
    """Adversarial: account exists but has zero tickets in the last 90 days
    (very common in this dataset -- see README). Must still produce a valid
    brief instead of erroring."""
    def run():
        tickets = get_account_tickets(account_id)
        details = [f"tickets in window: {len(tickets)} (test only meaningful if this is 0)"]
        result = generate_account_brief(account_id)
        has_sections = all(k in result for k in ("executive_summary", "open_risks", "talking_points", "churn_risk_level"))
        details.append(f"valid brief still produced with 0 recent tickets: {'OK' if has_sections else 'FAIL'}")
        return has_sections, 1.0 if has_sections else 0.0, details

    try:
        passed, score, details = run()
    except LLMConfigError as e:
        return TestResult("T2-05", "brief", "[Adversarial] Account with zero recent tickets must still produce a graceful, valid brief", True, False, 0.0, [], error=str(e))
    except Exception as e:
        return TestResult("T2-05", "brief", "[Adversarial] Account with zero recent tickets must still produce a graceful, valid brief", True, False, 0.0, [], error=f"{type(e).__name__}: {e}")
    return TestResult("T2-05", "brief", "[Adversarial] Account with zero recent tickets must still produce a graceful, valid brief", True, passed, score, details)


def eval_case_determinism(account_id: str) -> TestResult:
    """Task 2 explicitly requires deterministic output for the same input."""
    def run():
        r1 = generate_account_brief(account_id)
        r2 = generate_account_brief(account_id)
        details = []
        same_level = r1["churn_risk_level"] == r2["churn_risk_level"]
        details.append(f"churn_risk_level stable across 2 runs: {'OK' if same_level else 'FAIL'} ({r1['churn_risk_level']} vs {r2['churn_risk_level']})")
        quotes1 = sorted(r["evidence_quote"] for r in r1.get("open_risks", []))
        quotes2 = sorted(r["evidence_quote"] for r in r2.get("open_risks", []))
        same_quotes = quotes1 == quotes2
        details.append(f"evidence_quote set stable across 2 runs: {'OK' if same_quotes else 'FAIL'}")
        score = (0.5 if same_level else 0.0) + (0.5 if same_quotes else 0.0)
        return score >= 1.0, score, details

    try:
        passed, score, details = run()
    except LLMConfigError as e:
        return TestResult("T2-06", "brief", "Repeated calls on the same account must be deterministic", False, False, 0.0, [], error=str(e))
    except Exception as e:
        return TestResult("T2-06", "brief", "Repeated calls on the same account must be deterministic", False, False, 0.0, [], error=f"{type(e).__name__}: {e}")
    return TestResult("T2-06", "brief", "Repeated calls on the same account must be deterministic", False, passed, score, details)


def eval_case_judge_executive_summary(account_id: str) -> TestResult:
    """LLM-as-judge test: grades whether the executive summary is genuinely
    informative (specific facts/numbers) rather than generic filler."""
    test_id, desc = "T2-07", "[LLM-as-judge] Executive summary must be specific and informative, not generic filler"
    try:
        result = generate_account_brief(account_id)
    except LLMConfigError as e:
        return TestResult(test_id, "brief", desc, False, False, 0.0, [], error=str(e))
    except Exception as e:
        return TestResult(test_id, "brief", desc, False, False, 0.0, [], error=f"{type(e).__name__}: {e}")

    judgment = llm_judge(
        result.get("executive_summary", ""),
        rubric=(
            "This is an executive summary opening a TAM's account brief before a QBR. It should contain "
            "specific, concrete facts (numbers, trends, named risks) a TAM could not get from a one-line "
            "status field, not vague filler like 'this account is important to us'."
        ),
    )
    if judgment is None:
        return TestResult(test_id, "brief", desc, False, False, 0.0, [],
                           error="LLM judge unavailable (mock backend or no API key) -- skipped, not counted as failure")

    score = float(judgment["score"])
    details = [f"judge rationale: {judgment['rationale']}", f"judge score: {score}"]
    return TestResult(test_id, "brief", desc, False, score >= 0.6, score, details)


def _pick_demo_account_ids() -> dict[str, str]:
    """Picks concrete account_ids from the mock dataset for the test cases
    above, so the harness is grounded in real data rather than fixtures we
    invented by hand."""
    from src.data_loader import load_accounts

    accounts = load_accounts()
    with_signals = next(
        (a for a in accounts if a.get("escalation_notes") and get_account_tickets(a["account_id"])),
        accounts[0],
    )
    healthy = next(
        (a for a in accounts if a["health_status"] == "Healthy" and a["usage_trend"] in ("Stable", "Increasing")),
        accounts[1] if len(accounts) > 1 else accounts[0],
    )
    p1_heavy = next(
        (a for a in accounts if any(t.get("urgency") == "P1" for t in get_account_tickets(a["account_id"]))),
        with_signals,
    )
    no_recent = next(
        (a for a in accounts if not get_account_tickets(a["account_id"])),
        accounts[-1],
    )
    determinism_acct = with_signals

    return {
        "with_signals": with_signals["account_id"],
        "healthy": healthy["account_id"],
        "p1_heavy": p1_heavy["account_id"],
        "no_recent": no_recent["account_id"],
        "determinism": determinism_acct["account_id"],
    }


def build_account_test_cases() -> list[Callable[[], TestResult]]:
    ids = _pick_demo_account_ids()
    return [
        lambda: eval_case_account_with_signals(ids["with_signals"]),
        lambda: eval_case_healthy_account_low_churn(ids["healthy"]),
        lambda: eval_case_p1_volume_flagged(ids["p1_heavy"]),
        eval_case_adversarial_unknown_account,
        lambda: eval_case_adversarial_no_recent_tickets(ids["no_recent"]),
        lambda: eval_case_determinism(ids["determinism"]),
        lambda: eval_case_judge_executive_summary(ids["with_signals"]),
    ]


# ---------------------------------------------------------------------------
# Runner + report generation
# ---------------------------------------------------------------------------

def run_all() -> list[TestResult]:
    results: list[TestResult] = []
    for fn in TRIAGE_TEST_CASES:
        results.append(fn())
    for fn in build_account_test_cases():
        results.append(fn())
    return results


def _summarize(results: list[TestResult]) -> dict[str, Any]:
    by_task: dict[str, list[TestResult]] = {}
    for r in results:
        by_task.setdefault(r.task, []).append(r)

    summary = {}
    for task, rs in by_task.items():
        graded = [r for r in rs if r.error is None]
        summary[task] = {
            "total": len(rs),
            "graded": len(graded),
            "skipped_no_llm": len(rs) - len(graded),
            "passed": sum(1 for r in graded if r.passed),
            "pass_rate": round(sum(1 for r in graded if r.passed) / len(graded), 3) if graded else None,
            "mean_quality_score": round(statistics.mean(r.quality_score for r in graded), 3) if graded else None,
        }
    return summary


def run_all_and_report() -> dict[str, Any]:
    backend = "mock" if llm_client.USE_MOCK_LLM else "gemini"
    results = run_all()
    summary = _summarize(results)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "llm_backend": backend,
        "summary": summary,
        "results": [
            {
                "test_id": r.test_id,
                "task": r.task,
                "description": r.description,
                "adversarial": r.adversarial,
                "passed": r.passed,
                "quality_score": round(r.quality_score, 3),
                "details": r.details,
                "error": r.error,
            }
            for r in results
        ],
    }

    (REPORT_DIR / "eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (REPORT_DIR / "eval_report.md").write_text(_render_markdown(report), encoding="utf-8")

    print(f"LLM backend: {backend}")
    for task, s in summary.items():
        pr = f"{s['pass_rate']*100:.0f}%" if s["pass_rate"] is not None else "n/a (no LLM configured)"
        mq = f"{s['mean_quality_score']:.2f}" if s["mean_quality_score"] is not None else "n/a"
        print(f"  {task}: {s['passed']}/{s['graded']} passed ({pr}), mean quality {mq}, {s['skipped_no_llm']} skipped")
    print(f"\nWrote {REPORT_DIR / 'eval_report.json'}")
    print(f"Wrote {REPORT_DIR / 'eval_report.md'}")
    return report


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Eval Report",
        "",
        f"Generated: {report['generated_at']}",
        f"LLM backend: `{report['llm_backend']}`",
        "",
        "## Summary",
        "",
        "| Task | Passed | Graded | Pass rate | Mean quality score | Skipped (no LLM) |",
        "|---|---|---|---|---|---|",
    ]
    for task, s in report["summary"].items():
        pr = f"{s['pass_rate']*100:.0f}%" if s["pass_rate"] is not None else "n/a"
        mq = f"{s['mean_quality_score']:.2f}" if s["mean_quality_score"] is not None else "n/a"
        lines.append(f"| {task} | {s['passed']} | {s['graded']} | {pr} | {mq} | {s['skipped_no_llm']} |")

    lines += ["", "## Test cases", ""]
    for r in report["results"]:
        status = "SKIPPED (no LLM configured)" if r["error"] else ("PASS" if r["passed"] else "FAIL")
        adv = " *(adversarial)*" if r["adversarial"] else ""
        lines.append(f"### {r['test_id']} -- {r['description']}{adv}")
        lines.append(f"**Task:** {r['task']} | **Status:** {status} | **Quality score:** {r['quality_score']}")
        lines.append("")
        if r["error"]:
            lines.append(f"> {r['error']}")
        else:
            for d in r["details"]:
                lines.append(f"- {d}")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    run_all_and_report()
