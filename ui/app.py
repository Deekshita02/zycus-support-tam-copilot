"""
Thin Streamlit UI for non-technical TAM/support use. Run with:
    python cli.py ui
or:
    streamlit run ui/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.account_health import AccountNotFoundError, generate_account_brief
from src.data_loader import load_accounts, load_tickets
from src.llm_client import LLMConfigError
from src.triage import triage_ticket

st.set_page_config(page_title="Support & TAM Copilot", layout="wide")
st.title("Support & TAM Copilot")

tab1, tab2 = st.tabs(["🎫 Ticket Triage", "📋 Account Brief"])

# ---------------------------------------------------------------------------
# Tab 1: Ticket triage
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Triage an incoming ticket")

    tickets = load_tickets()
    use_sample = st.checkbox("Load a sample ticket from the mock dataset", value=True)

    if use_sample:
        options = {f"{t['ticket_id']} — {t['subject']}": t for t in tickets[:50]}
        choice = st.selectbox("Sample ticket", list(options.keys()))
        sample = options[choice]
        default_subject, default_body, default_plan = sample["subject"], sample["body"], sample["plan_tier"]
    else:
        default_subject, default_body, default_plan = "", "", "Business"

    subject = st.text_input("Subject", value=default_subject)
    body = st.text_area("Body", value=default_body, height=180)
    plan_tier = st.selectbox("Plan tier", ["Starter", "Professional", "Business", "Enterprise"],
                              index=["Starter", "Professional", "Business", "Enterprise"].index(default_plan)
                              if default_plan in ["Starter", "Professional", "Business", "Enterprise"] else 2)

    if st.button("Triage ticket", type="primary"):
        with st.spinner("Retrieving KB context and classifying..."):
            try:
                result = triage_ticket(subject=subject, body=body, plan_tier=plan_tier)
            except LLMConfigError as e:
                st.error(str(e))
                result = None
            except ValueError as e:
                st.error(str(e))
                result = None

        if result:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Product", result["product"])
            c2.metric("Category", result["category"])
            c3.metric("Urgency", result["urgency"])
            c4.metric("Confidence", f"{result['confidence']:.0%}")

            st.markdown(f"**Product area:** {result['product_area']}  \n**Recommended team:** {result['recommended_team']}")
            st.markdown(f"**Reasoning:** {result['reasoning']}")

            if result["kb_match_found"]:
                st.info(f"📚 KB match: `{result['kb_match_source']}` — {result['kb_match_excerpt']}")
            else:
                st.warning("No matching KB doc found for this issue.")

            st.markdown("**Draft first response:**")
            st.text_area("draft", value=result["draft_first_response"], height=140, label_visibility="collapsed")

            with st.expander("Raw JSON"):
                st.json(result)

# ---------------------------------------------------------------------------
# Tab 2: Account brief
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Generate a pre-QBR account brief")

    accounts = load_accounts()
    options = {f"{a['account_id']} — {a['company']} ({a['health_status']})": a["account_id"] for a in accounts}
    choice = st.selectbox("Account", list(options.keys()))
    account_id = options[choice]

    if st.button("Generate brief", type="primary"):
        with st.spinner("Pulling account data, extracting risk signals, and summarizing..."):
            try:
                brief = generate_account_brief(account_id)
            except AccountNotFoundError as e:
                st.error(str(e))
                brief = None
            except LLMConfigError as e:
                st.error(str(e))
                brief = None

        if brief:
            level_color = {"Low": "green", "Medium": "orange", "High": "red"}[brief["churn_risk_level"]]
            st.markdown(f"### {brief['company']}  &nbsp; :{level_color}[Churn risk: {brief['churn_risk_level']}]")

            st.markdown("#### Executive summary")
            st.write(brief["executive_summary"])

            st.markdown("#### Open risks & flagged issues")
            for risk in brief["open_risks"]:
                st.markdown(f"- **{risk['risk_summary']}**  \n  > *\"{risk['evidence_quote']}\"* — `{risk['source']}`")

            st.markdown("#### Recommended talking points")
            for point in brief["talking_points"]:
                st.markdown(f"- {point}")

            with st.expander("Raw JSON"):
                st.json(brief)
