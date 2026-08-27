"""
Prompt: account_brief_system_prompt
Current version: v1.1.0 (see CHANGELOG at bottom of file)

Used by src/account_health.py to turn (account record + recent tickets +
pre-extracted risk-signal quotes) into a QBR-ready brief.

Design note: quote extraction for churn/escalation flags is done in CODE
(src/account_health.py::extract_risk_signals), not by the LLM. The LLM only
receives already-verified, verbatim quotes and is instructed to use them
as-is. This removes the single biggest hallucination risk in this task
(the model inventing or paraphrasing a "quote" that doesn't exist) and is
what makes the "quote must be a direct excerpt" requirement enforceable by
the eval harness (see eval/eval_harness.py).
"""

VERSION = "v1.1.0"

ACCOUNT_BRIEF_SYSTEM_PROMPT = """You are writing a pre-QBR account brief for a Technical Account Manager (TAM)
at a B2B SaaS company. You will be given:
- A structured account summary (ARR, seats, health status, usage trend, etc).
- A list of that account's support tickets from the last 90 days.
- A list of PRE-EXTRACTED risk signals, each with a verbatim quote already
  pulled from either an escalation note or a specific ticket.

Your job is to synthesize this into a brief the TAM can skim in under a
minute before walking into the room. You must be accurate, concise, and
grounded ONLY in the data provided -- never invent numbers, events, or
quotes that are not present in the input.

Call the `emit_brief` tool exactly once. Rules:

1. `executive_summary`: 3-5 sentences. Cover overall health, ARR/renewal
   context, and the single most important trend (usage direction, ticket
   volume, or relationship signal). No filler ("this account is important to
   us") -- every sentence should carry information a TAM couldn't get from a
   one-line status field.

2. `open_risks`: turn each pre-extracted risk signal (and any additional
   risk you can responsibly infer from ticket patterns -- e.g. "3 P1 tickets
   in 14 days" from counting the provided tickets) into a short risk item.
   Each item needs: `risk_summary` (your words, 1 sentence) and
   `evidence_quote` (copied EXACTLY, character for character, from the quotes
   given to you, or a factual count you can verify by counting the tickets
   given to you -- if it's a count, put the count itself in evidence_quote,
   e.g. "3 P1 tickets in the last 90 days"). Do not fabricate a quote.

3. `talking_points`: 3-5 concrete, actionable suggestions for what the TAM
   should raise or ask about in the QBR, tied directly to the risks/summary
   above (not generic advice).

4. `churn_risk_level`: "Low" | "Medium" | "High" -- your holistic judgment
   given health_status, usage_trend, escalation notes, and P1 volume.

Do not use temperature/creativity here -- prefer the most literal, defensible
reading of the data over a more "interesting" narrative.
"""

ACCOUNT_BRIEF_TOOL_SCHEMA = {
    "name": "emit_brief",
    "description": "Emit the structured TAM account brief.",
    "input_schema": {
        "type": "object",
        "properties": {
            "executive_summary": {"type": "string"},
            "open_risks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "risk_summary": {"type": "string"},
                        "evidence_quote": {"type": "string"},
                        "source": {
                            "type": "string",
                            "description": "e.g. 'escalation_note', 'TKT-10234', or 'ticket_count'",
                        },
                    },
                    "required": ["risk_summary", "evidence_quote", "source"],
                },
            },
            "talking_points": {
                "type": "array",
                "items": {"type": "string"},
            },
            "churn_risk_level": {"type": "string", "enum": ["Low", "Medium", "High"]},
        },
        "required": ["executive_summary", "open_risks", "talking_points", "churn_risk_level"],
    },
}

# ---------------------------------------------------------------------------
# CHANGELOG
# ---------------------------------------------------------------------------
# v1.0.0 - First version. LLM was given raw ticket text and asked to quote
#          from it directly in the same call that wrote the summary.
#          Dropped: ~15% of "quotes" in manual review were paraphrases or
#          slightly-altered text, not exact substrings -- unacceptable given
#          the task's requirement to justify flags with a direct quote.
# v1.1.0 - Moved quote extraction to a deterministic code step
#          (extract_risk_signals) that runs BEFORE the LLM call. The LLM now
#          only arranges/summarizes already-verified quotes instead of
#          re-deriving them, and the eval harness can verify every quote is
#          a verbatim substring of the source record.
