import pytest

from src.config import CATEGORIES, PRODUCT_AREAS, PRODUCTS, URGENCY_LEVELS
from src.triage import triage_ticket


def test_triage_returns_valid_schema():
    result = triage_ticket(
        subject="Production pipeline down",
        body="Our DataBridge Pro pipeline has been down for an hour, 100 users affected.",
        plan_tier="Enterprise",
    )
    assert result["product"] in PRODUCTS
    assert result["product_area"] in PRODUCT_AREAS[result["product"]]
    assert result["category"] in CATEGORIES
    assert result["urgency"] in URGENCY_LEVELS
    assert 0 <= result["confidence"] <= 1
    assert result["draft_first_response"]
    assert "_meta" in result


def test_triage_rejects_empty_body():
    with pytest.raises(ValueError):
        triage_ticket(subject="hi", body="   ")


def test_triage_attaches_kb_candidates_metadata():
    result = triage_ticket(subject="Question", body="How does SSO work with SecureVault?")
    assert "kb_candidates_considered" in result["_meta"]
    assert isinstance(result["_meta"]["kb_candidates_considered"], list)
