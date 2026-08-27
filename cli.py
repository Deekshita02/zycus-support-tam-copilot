#!/usr/bin/env python3
"""
Single entry point for the whole project.

Usage:
    python cli.py triage --file path/to/ticket.json
    python cli.py triage --subject "..." --body "..."
    python cli.py triage --ticket-id TKT-10042      # pull a real ticket from the mock dataset
    python cli.py brief ACC-3336
    python cli.py eval                                # runs the eval harness, writes eval/eval_report.{json,md}
    python cli.py serve                                # FastAPI server on :8000
    python cli.py ui                                   # Streamlit demo UI
"""
from __future__ import annotations

import argparse
import json
import sys


def cmd_triage(args: argparse.Namespace) -> None:
    from src.triage import triage_ticket, triage_ticket_dict
    from src.data_loader import get_ticket

    if args.ticket_id:
        ticket = get_ticket(args.ticket_id)
        if ticket is None:
            print(f"No ticket with id {args.ticket_id!r} in the mock dataset.", file=sys.stderr)
            sys.exit(1)
        result = triage_ticket_dict(ticket)
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            ticket = json.load(f)
        result = triage_ticket_dict(ticket)
    elif args.body:
        result = triage_ticket(subject=args.subject or "", body=args.body, plan_tier=args.plan_tier)
    else:
        print("Provide one of --ticket-id, --file, or --body.", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(result, indent=2))


def cmd_brief(args: argparse.Namespace) -> None:
    from src.account_health import generate_account_brief, AccountNotFoundError

    try:
        result = generate_account_brief(args.account_id)
    except AccountNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    print(json.dumps(result, indent=2))


def cmd_eval(args: argparse.Namespace) -> None:
    from eval.eval_harness import run_all_and_report

    run_all_and_report()


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run("src.api:app", host="0.0.0.0", port=args.port, reload=args.reload)


def cmd_ui(args: argparse.Namespace) -> None:
    import subprocess

    subprocess.run([sys.executable, "-m", "streamlit", "run", "ui/app.py"])


def main() -> None:
    parser = argparse.ArgumentParser(description="US Delivery Internship task-round CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_triage = sub.add_parser("triage", help="Run Task 1 triage on a ticket")
    p_triage.add_argument("--ticket-id", help="Ticket id from the mock dataset, e.g. TKT-10042")
    p_triage.add_argument("--file", help="Path to a JSON file with {subject, body[, plan_tier]}")
    p_triage.add_argument("--subject", help="Ticket subject (used with --body)")
    p_triage.add_argument("--body", help="Ticket body text (used with --subject)")
    p_triage.add_argument("--plan-tier", dest="plan_tier", help="Optional plan tier context")
    p_triage.set_defaults(func=cmd_triage)

    p_brief = sub.add_parser("brief", help="Run Task 2 account health brief")
    p_brief.add_argument("account_id", help="e.g. ACC-3336")
    p_brief.set_defaults(func=cmd_brief)

    p_eval = sub.add_parser("eval", help="Run Task 3 eval harness")
    p_eval.set_defaults(func=cmd_eval)

    p_serve = sub.add_parser("serve", help="Run the FastAPI server")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    p_ui = sub.add_parser("ui", help="Run the Streamlit demo UI")
    p_ui.set_defaults(func=cmd_ui)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
