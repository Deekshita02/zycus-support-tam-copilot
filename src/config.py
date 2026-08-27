"""
Central configuration for the support/TAM tooling.

All environment-dependent values (API keys, model name) are read from
environment variables only -- never hardcoded. See .env.example.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
KB_DIR = ROOT_DIR / "knowledge_base"
TICKETS_PATH = DATA_DIR / "tickets.json"
ACCOUNTS_PATH = DATA_DIR / "accounts.json"

# ---------------------------------------------------------------------------
# LLM settings
# ---------------------------------------------------------------------------
# Model id is read from env so it can be swapped without a code change.
# NOTE: switched from Anthropic to Gemini -- see src/llm_client.py.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Deterministic-by-default: temperature 0 everywhere an LLM is called for a
# structured/production output (Task 2 explicitly requires determinism).
DEFAULT_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0"))
MAX_TOKENS_TRIAGE = int(os.environ.get("MAX_TOKENS_TRIAGE", "1200"))
MAX_TOKENS_BRIEF = int(os.environ.get("MAX_TOKENS_BRIEF", "1500"))

# ---------------------------------------------------------------------------
# Domain enums (must match DATA_SCHEMA.md exactly -- used to constrain
# LLM output and to validate it post-hoc)
# ---------------------------------------------------------------------------
PRODUCTS = [
    "DataBridge Pro",
    "CloudSync",
    "AnalyticsHub",
    "SecureVault",
    "WorkflowEngine",
]

PRODUCT_AREAS = {
    "DataBridge Pro": ["Data Ingestion", "Schema Management", "Pipeline Monitoring", "Connectors", "API"],
    "CloudSync": ["File Sync", "Conflict Resolution", "Permissions", "Bandwidth Limits", "Integrations"],
    "AnalyticsHub": ["Dashboard", "Reports", "Data Sources", "Alerts", "Exports"],
    "SecureVault": ["Authentication", "Encryption", "Audit Logs", "Key Management", "SSO Configuration"],
    "WorkflowEngine": ["Triggers", "Actions", "Scheduling", "Error Handling", "Templates"],
}

CATEGORIES = [
    "Bug", "Feature Request", "How-To", "Performance",
    "Billing", "Integration", "Onboarding", "Data Loss",
]

URGENCY_LEVELS = ["P1", "P2", "P3", "P4"]

RESPONDER_TEAMS = [
    "Tier-1 Support",
    "Tier-2 Support Engineering",
    "Billing Operations",
    "Customer Success / Onboarding",
    "Integrations Engineering",
    "Security & Compliance",
    "Product / Feature Requests",
]

# ---------------------------------------------------------------------------
# Prompt versioning (bonus requirement: each prompt tracked w/ version id)
# ---------------------------------------------------------------------------
PROMPT_VERSIONS = {
    "triage_system_prompt": "v1.2.0",
    "account_brief_system_prompt": "v1.1.0",
}
