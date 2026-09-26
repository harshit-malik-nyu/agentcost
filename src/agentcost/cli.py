"""
Command line interface.

Three questions a buyer actually has, answerable without running a model:

    breakdown   where does the money go on this process?
    policy      what review rate does an error tolerance require?
    audit       what is every parameter, and who says so?
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

from .calibration import capture_curve, policy_for, simulate
from .economics import (
    BATCHED, PER_QUESTION, PRICE_TIERS, Assumptions, agent_cost, audit,
    human_baseline,
)

DATA = Path(__file__).resolve().parents[2] / "data" / "cuad_stats.json"


def _corpus() -> tuple[float, int, int]:
    """Mean tokens, questions per item, item count — from the real corpus."""
    if DATA.exists():
        d = json.loads(DATA.read_text())
        toks = [x["chars"] / 4 for x in d["documents"]]
        return st.mean(toks), d["questions_per_contract"], len(toks)
    return 11712.0, 41, 102        # measured defaults, if the file is absent


def _model(name: str):
    for m in PRICE_TIERS:
        if m.name == name:
            return m
    raise SystemExit(f"unknown model tier: {name}")


def cmd_breakdown(args) -> int:
    mean_tok, n_q, n = _corpus()
    n = args.items or n
    a = Assumptions()
    arch = PER_QUESTION if args.per_question else BATCHED
    m = _model(args.model)

    c = agent_cost(items=n, mean_tokens=mean_tok, questions_per_item=n_q,
                   model=m, architecture=arch, review_rate=args.review,
                   assumptions=a)
    base = human_baseline(n, a)

    if args.json:
        print(json.dumps({**c.as_dict(), "human_baseline_usd": base}, indent=2))
        return 0

    print(f"{n:,} documents, {n_q} extractions each, {arch.name}, {m.name} model")
    print(f"review rate {args.review:.0%}\n")
    print(f"  inference        ${c.inference:>12,.2f}   {c.inference_share:>7.2%} of total")
    print(f"  human review     ${c.review:>12,.2f}")
    print(f"  total            ${c.total:>12,.2f}   (${c.per_item:,.2f} per document)")
    print(f"  human-only       ${base:>12,.2f}   (${base/n:,.2f} per document)")
    saved = base - c.total
    print(f"\n  saving           ${saved:>12,.2f}   {saved/base:>7.1%}")
    print(f"\n  Inference is {c.inference_share:.2%} of the bill. Choosing a "
          f"cheaper model\n  moves a rounding error; the review rate moves "
          f"everything.")
    return 0


def cmd_policy(args) -> int:
    mean_tok, n_q, _ = _corpus()
    m = _model(args.model)
    curve = capture_curve(simulate(args.accuracy, args.calibration, seed=7))
    pol = policy_for(curve, args.tolerance, mean_tokens=mean_tok,
                     questions_per_item=n_q, model=m, architecture=BATCHED)

    if args.json:
        print(json.dumps({**pol.as_dict(), "lift": curve.lift}, indent=2))
        return 0

    print(f"agent accuracy {args.accuracy:.0%}, calibration {args.calibration:.2f}")
    print(f"error tolerance {args.tolerance:.0%}\n")
    print(f"  confidence ranks errors {curve.lift:.1f}x better than chance")
    print(f"  review rate required      {pol.review_rate:>8.0%}")
    print(f"  cost per document         ${pol.cost_per_item:>7.2f}")
    print(f"  versus human-only         ${human_baseline(1):>7.2f}")
    print(f"  saving                    {pol.saving_pct:>8.0%}")
    if curve.lift < 1.5:
        print("\n  Confidence carries almost no information about correctness,"
              "\n  so reviewing by confidence is barely better than reviewing"
              "\n  at random. This agent cannot triage its own work.")
    return 0


def cmd_audit(args) -> int:
    print(audit())
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="agentcost",
        description="What an AI agent costs to run a document process.")
    ap.add_argument("--json", action="store_true")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("breakdown", help="where the money goes")
    p.add_argument("--items", type=int, default=0)
    p.add_argument("--review", type=float, default=1.0,
                   help="fraction of documents a human checks")
    p.add_argument("--model", default="mid", choices=[m.name for m in PRICE_TIERS])
    p.add_argument("--per-question", action="store_true",
                   help="resend the document for every extraction")
    p.set_defaults(func=cmd_breakdown)

    p = sub.add_parser("policy", help="review rate an error tolerance requires")
    p.add_argument("--accuracy", type=float, default=0.90)
    p.add_argument("--calibration", type=float, default=0.5,
                   help="0 = confidence is noise, 1 = perfect error ranking")
    p.add_argument("--tolerance", type=float, default=0.05)
    p.add_argument("--model", default="mid", choices=[m.name for m in PRICE_TIERS])
    p.set_defaults(func=cmd_policy)

    p = sub.add_parser("audit", help="every parameter and its source")
    p.set_defaults(func=cmd_audit)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
