"""
FastAPI wrapper exposing Task 1 (triage) and Task 2 (account brief) as REST
endpoints. Run with: python cli.py serve   (or: uvicorn src.api:app --reload)
"""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.account_health import AccountNotFoundError, generate_account_brief
from src.llm_client import LLMConfigError
from src.triage import triage_ticket

app = FastAPI(
    title="Support & TAM Tooling API",
    description="Task 1 (ticket triage) and Task 2 (account health brief) as REST endpoints.",
    version="1.0.0",
)


class TriageRequest(BaseModel):
    subject: str = Field(default="", description="Ticket subject line")
    body: str = Field(..., description="Ticket body text")
    plan_tier: Optional[str] = Field(default=None, description="Optional plan tier context")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/triage")
def triage_endpoint(req: TriageRequest) -> dict:
    try:
        return triage_ticket(subject=req.subject, body=req.body, plan_tier=req.plan_tier)
    except LLMConfigError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/account/{account_id}/brief")
def account_brief_endpoint(account_id: str) -> dict:
    try:
        return generate_account_brief(account_id)
    except AccountNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except LLMConfigError as e:
        raise HTTPException(status_code=503, detail=str(e))
