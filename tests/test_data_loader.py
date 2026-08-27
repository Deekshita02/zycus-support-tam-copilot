from src.data_loader import get_account, get_account_tickets, get_ticket, load_accounts, load_tickets


def test_load_tickets_count():
    assert len(load_tickets()) == 500


def test_load_accounts_count():
    assert len(load_accounts()) == 50


def test_get_account_missing_returns_none():
    assert get_account("ACC-DOES-NOT-EXIST") is None


def test_get_account_tickets_within_window():
    accounts = load_accounts()
    acc_id = accounts[0]["account_id"]
    tickets = get_account_tickets(acc_id, days=90)
    for t in tickets:
        assert t["account_id"] == acc_id


def test_get_ticket_roundtrip():
    first = load_tickets()[0]
    assert get_ticket(first["ticket_id"])["ticket_id"] == first["ticket_id"]
