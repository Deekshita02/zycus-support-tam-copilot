import pytest

from src.account_health import AccountNotFoundError, extract_risk_signals, generate_account_brief
from src.data_loader import get_account_tickets, load_accounts


def test_generate_brief_unknown_account_raises():
    with pytest.raises(AccountNotFoundError):
        generate_account_brief("ACC-DOES-NOT-EXIST-99999")


def test_generate_brief_returns_required_sections():
    account_id = load_accounts()[0]["account_id"]
    result = generate_account_brief(account_id)
    for key in ("executive_summary", "open_risks", "talking_points", "churn_risk_level"):
        assert key in result
    assert result["churn_risk_level"] in ("Low", "Medium", "High")


def test_extract_risk_signals_quotes_are_verbatim():
    accounts = load_accounts()
    account = next(a for a in accounts if a.get("escalation_notes"))
    tickets = get_account_tickets(account["account_id"])
    signals = extract_risk_signals(account, tickets)

    escalation_note_signals = [s for s in signals if s["kind"] == "escalation_note"]
    assert len(escalation_note_signals) == len(account["escalation_notes"])
    for s in escalation_note_signals:
        assert s["quote"] in account["escalation_notes"]


def test_extract_risk_signals_handles_no_tickets():
    accounts = load_accounts()
    account = accounts[0]
    signals = extract_risk_signals(account, [])
    # Should not crash even with zero tickets; escalation notes alone still work.
    assert isinstance(signals, list)
