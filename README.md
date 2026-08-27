# Support & TAM Copilot — US Delivery Internship Task Round

Production-grade AI tooling for Technical Support and TAM teams, built on the
provided mock dataset (500 tickets, 50 accounts, 9 KB docs). Four parts:
**ticket triage** (Task 1), **account health briefs** (Task 2), an
**eval harness** (Task 3), and this **design note** (Task 4).

> Dataset provenance and full field-level schema: see [`DATASET.md`](DATASET.md)
> and [`DATA_SCHEMA.md`](DATA_SCHEMA.md) (unmodified from the starter repo).

## Repo layout

```
src/
  config.py              domain enums, model settings, prompt version registry
  data_loader.py         tickets/accounts loading + joins
  retrieval.py           dependency-free BM25 search over knowledge_base/
  llm_client.py          Anthropic API wrapper (structured tool-use + streaming)
  mock_llm.py            deterministic offline stand-in (USE_MOCK_LLM=1) for CI/smoke tests
  triage.py              Task 1 pipeline
  account_health.py      Task 2 pipeline
  api.py                 FastAPI endpoints
  prompts/                versioned prompts + changelogs
eval/
  eval_harness.py        Task 3 harness (rule-based + LLM-as-judge, 14 test cases)
  eval_report.json/.md   generated report (run `python cli.py eval` to regenerate)
ui/app.py                Streamlit demo (bonus)
tests/                   pytest smoke tests (mock backend, no API key needed)
.github/workflows/eval.yml  CI: tests on every push, eval harness if a key secret is set (bonus)
cli.py                   single entry point for everything
data/, knowledge_base/    the provided mock dataset, unmodified
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then edit .env and set ANTHROPIC_API_KEY
```

## Sample runs

**Task 1 — triage a ticket** (by id from the dataset, from a JSON file, or raw text):

```bash
python cli.py triage --ticket-id TKT-10000
python cli.py triage --subject "Prod pipeline down" --body "DataBridge Pro pipeline down for 2 hours, 300 users affected"
```

**Task 2 — generate an account brief:**

```bash
python cli.py brief ACC-3336
```

**Task 3 — run the eval harness** (writes `eval/eval_report.json` and `.md`):

```bash
python cli.py eval
```

**REST API:**

```bash
python cli.py serve
curl -X POST localhost:8000/triage -H "Content-Type: application/json" \
  -d '{"subject":"Prod down","body":"DataBridge Pro pipeline down, 300 users affected"}'
curl localhost:8000/account/ACC-3336/brief
```

**Streamlit demo (bonus):**

```bash
python cli.py ui
```

**Offline / no API key:** every command above also works with `USE_MOCK_LLM=1`
prefixed, which swaps in a deterministic rule-based backend (`src/mock_llm.py`)
instead of calling Anthropic — useful for CI and for verifying the plumbing
without spending API credits. `tests/` always runs this way.

```bash
USE_MOCK_LLM=1 python cli.py eval
pytest -v   # always uses the mock backend, see tests/conftest.py
```

---

## Design note (Task 4)

### Failure modes

1. **LLM returns a plausible-but-wrong classification on ambiguous tickets.**
   The dataset intentionally includes ambiguous category/urgency labels, and
   a real ticket stream will too. *Mitigation:* the model must emit a
   `confidence` score, and the eval harness has an explicit adversarial case
   (T1-06) asserting low confidence on vague input. In production I'd route
   anything below a confidence threshold to a "needs human triage" queue
   instead of auto-routing it, and track confidence-vs-actual-correctness
   over time to recalibrate the threshold.

2. **Hallucinated "evidence" in the account brief.** This is the single
   highest-stakes failure mode for Task 2 — a TAM repeating a fabricated
   quote to a customer is worse than no brief at all. *Mitigation:* I moved
   quote extraction out of the LLM entirely (`extract_risk_signals` in
   `account_health.py` runs in plain Python before the model ever sees the
   data) and the eval harness independently re-verifies every returned
   `evidence_quote` is an exact substring of the source ticket/account
   record. The LLM only arranges and prioritizes pre-verified facts.

3. **Silent drift after a prompt or model change.** A prompt edit that
   improves one test case can regress another without anyone noticing until
   a customer complains. *Mitigation:* prompts are version-stamped
   (`src/prompts/*.py`, `PROMPT_VERSIONS` in `config.py`) with changelogs,
   and the eval harness (Task 3) is wired into CI (`.github/workflows/eval.yml`)
   so every commit re-runs the full test suite and a report is attached as a
   build artifact — a regression shows up as a pass-rate drop in the PR, not
   in production.

### Latency vs. quality trade-off

Task 1 makes two sequential calls in the worst case conceptually (retrieval,
then one structured LLM call) but I collapsed it to a **single LLM call**
that receives pre-retrieved KB context and returns the full structured
decision via one forced tool call, rather than chaining separate
"classify → then retrieve → then draft response" calls. That's the
quality-preserving choice: one call with full context beats three
narrower ones on coherence (the draft response can reference the same
reasoning used for classification) and is roughly 3x cheaper in latency.
The cost is that if the KB retrieval is wrong, there's no second pass to
catch it — the model only sees whatever BM25 pulled back.

**If latency were the hard constraint** (e.g. sub-second triage feeding a
live chat widget), I'd change three things: (a) skip the LLM call for
high-confidence lexical matches — many tickets containing a known error
code from the KB error tables could be classified by exact-match lookup
alone, no model call needed; (b) drop max_tokens on the draft response and
generate it asynchronously/streamed after the classification renders, so
the human-facing urgency/category badge appears immediately; (c) move to a
smaller/faster model for classification and reserve the larger model only
for the free-text draft response, which is where quality differences are
most visible to the end customer.

### Data sensitivity

Ticket bodies and account escalation notes are exactly the kind of
free-text field that accumulates PII (names, emails, sometimes pasted
stack traces with internal hostnames). This design sends that text to an
external LLM API, so: the `.env.example` keeps the API key out of the
repo entirely (disqualifier-aware); nothing is logged beyond what
`llm_client.py` needs for the call itself (no separate analytics/logging
pipeline was added); and the mock dataset here is synthetic, so this repo
doesn't itself leak anything real. For a genuine production version I
would add a PII-scrubbing pass (regex + a lightweight NER pass for names/
emails/phone numbers) on ticket bodies *before* they're sent to the KB
retrieval and LLM steps, log only redacted versions, and route anything
tagged Enterprise/regulated-industry through a private-deployment or
zero-data-retention model endpoint rather than the standard API — this is
a config flag (`ANTHROPIC_MODEL`/endpoint), not an architecture change, so
it's cheap to add later.

### Scaling

At 10x ticket volume (5,000 tickets/day-ish equivalent), the first thing
to break is **not** the LLM calls themselves — those parallelize fine
behind a queue — it's the **retrieval step being recomputed from scratch
on every request** (`load_kb_chunks()`/`BM25Index` are `lru_cache`'d
in-process but rebuilt per process and held in memory). At 10x request
volume across multiple API replicas, that's 10x redundant index builds
and no shared cache. Fix: build the BM25 (or swap to a real vector/lexical
store like OpenSearch) index once as a batch job, not per-request. Second
thing to break: the current in-memory `lru_cache` over `tickets.json` /
`accounts.json` doesn't work once data isn't a static file anymore — a
real 10x-scale deployment needs this backed by an actual database with
indexes on `account_id` and `created_at`, not a full-file JSON load. Third:
naive sequential LLM calls would create a latency queue under load; the
API layer would need a request queue with concurrency limits tuned to the
Anthropic rate limit tier, plus backpressure (429s) surfaced to callers
rather than silently queuing forever.
