"""
What it costs to run a document process, with and without an agent.

The question this answers is not "is the model accurate enough". It is "what
does the work cost per item, and which term dominates". Those turn out to be
different questions with different answers, and the second one is the one that
decides whether automation saves money.

The structure
-------------
A document-extraction process has three costs:

    inference   tokens in and out, priced at list
    review      human time spent checking what the agent produced
    error       what escapes review and costs something later

Vendor comparisons are almost entirely about the first term. Measured against
the real corpus in `data/`, inference is roughly **one four-hundredth** of the
cost of the human review it is supposed to replace. Choosing a cheaper model
moves a rounding error.

The review term dominates, and the review term is set by what fraction of items
a human still has to look at. That fraction is not a property of the model's
accuracy — it is a property of whether the model can tell when it is wrong,
which `calibration.py` measures separately.

Evidence classes
----------------
Every parameter carries one, because the difference between a published list
price and my estimate of paralegal review time is the difference between a
number you can check and one you have to trust:

    PUBLISHED   from a vendor price list or published benchmark
    ESTIMATE    mine, defensible, unverified, and the thing to attack
    MEASURED    computed from the corpus in this repository
    DERIVED     arithmetic on the above
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Evidence(str, Enum):
    PUBLISHED = "published"
    MEASURED = "measured"
    ESTIMATE = "estimate"
    DERIVED = "derived"


@dataclass(frozen=True)
class Param:
    value: Any
    evidence: Evidence
    source: str
    note: str = ""

    def __float__(self) -> float:
        return float(self.value)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelPrice:
    """List price per million tokens."""

    name: str
    input_per_m: float
    output_per_m: float
    tier: str

    def cost(self, tokens_in: float, tokens_out: float) -> float:
        return (tokens_in / 1e6) * self.input_per_m + (tokens_out / 1e6) * self.output_per_m


# Representative list prices across the market as of early 2026. Exact vendor
# names are deliberately omitted: prices move monthly, and the finding below is
# that the choice between these tiers barely matters, which is robust to any of
# them changing.
PRICE_TIERS = [
    ModelPrice("frontier", 3.00, 15.00, "frontier"),
    ModelPrice("mid", 0.80, 4.00, "mid"),
    ModelPrice("small", 0.25, 1.25, "small"),
]

CHARS_PER_TOKEN = Param(
    value=4.0,
    evidence=Evidence.ESTIMATE,
    source="ESTIMATE — standard rule of thumb for English prose",
    note="Legal text runs slightly denser than this. The architectural "
         "conclusion holds across any plausible ratio because both "
         "architectures scale with the same constant.",
)

OUTPUT_TOKENS_PER_CLAUSE = Param(
    value=150.0,
    evidence=Evidence.ESTIMATE,
    source="ESTIMATE — a quoted span plus a short justification",
    note="The extracted span has a median length of 195 characters in the "
         "corpus; the rest is the model's reasoning and formatting.",
)


# ---------------------------------------------------------------------------
# Human review
# ---------------------------------------------------------------------------

PARALEGAL_RATE_PER_HOUR = Param(
    value=45.0,
    evidence=Evidence.ESTIMATE,
    source="ESTIMATE — fully loaded cost, REQUIRES VALIDATION",
    note="Salary, benefits, supervision and overhead for contract review "
         "staff. Offshored or outsourced review is materially cheaper and "
         "would lower this; a qualified lawyer doing the same work is several "
         "times higher.",
)

MINUTES_FULL_REVIEW = Param(
    value=45.0,
    evidence=Evidence.ESTIMATE,
    source="ESTIMATE — REQUIRES VALIDATION against a legal operations team",
    note="Time for a human to read a contract and extract all 41 clause types "
         "from scratch. The median contract in the corpus is about 6,400 "
         "tokens; the ninetieth percentile is four times that, so this is an "
         "average over a heavily skewed distribution.",
)

MINUTES_VERIFY = Param(
    value=12.0,
    evidence=Evidence.ESTIMATE,
    source="ESTIMATE — REQUIRES VALIDATION",
    note="Time to check an answer the agent has already proposed, with the "
         "span highlighted. Verification is faster than extraction because "
         "the reviewer is confirming a location rather than searching for "
         "one. The ratio to full review — here about 1:4 — is the single "
         "most load-bearing estimate in this model after the review rate "
         "itself.",
)


@dataclass
class Assumptions:
    """Mutable copy, so sensitivity runs cannot leak state between trials."""

    rate_per_hour: float = float(PARALEGAL_RATE_PER_HOUR)
    minutes_full_review: float = float(MINUTES_FULL_REVIEW)
    minutes_verify: float = float(MINUTES_VERIFY)
    chars_per_token: float = float(CHARS_PER_TOKEN)
    output_tokens_per_clause: float = float(OUTPUT_TOKENS_PER_CLAUSE)

    cost_per_escaped_error: float = 500.0
    """
    ESTIMATE, and the weakest number here.

    What one wrong or missed clause costs once it has passed review. Most cost
    nothing. A missed change-of-control or exclusivity clause in an M&A data
    room can cost a great deal more than this. It is an average standing in for
    a distribution nobody has published, and it is swept rather than defended.
    """

    @property
    def cost_full_review(self) -> float:
        return self.minutes_full_review / 60 * self.rate_per_hour

    @property
    def cost_verify(self) -> float:
        return self.minutes_verify / 60 * self.rate_per_hour


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

@dataclass
class Architecture:
    """
    How the work is sent to the model, which changes the token bill by a
    factor of thirty.

    PER_QUESTION resends the whole contract for each of the 41 clause types.
    It is the obvious implementation and the one most naive pipelines use.

    BATCHED sends the contract once and asks for all 41 clauses in a single
    pass. Same work, a thirtieth of the input tokens.
    """

    name: str
    resends_document: bool


PER_QUESTION = Architecture("per-question", True)
BATCHED = Architecture("batched", False)


@dataclass
class CostBreakdown:
    items: int
    inference: float
    review: float
    error: float

    review_rate: float
    architecture: str
    model: str

    @property
    def total(self) -> float:
        return self.inference + self.review + self.error

    @property
    def per_item(self) -> float:
        return self.total / self.items if self.items else 0.0

    @property
    def inference_share(self) -> float:
        return self.inference / self.total if self.total else 0.0

    def as_dict(self) -> dict:
        return {
            "items": self.items, "model": self.model,
            "architecture": self.architecture, "review_rate": self.review_rate,
            "inference_usd": self.inference, "review_usd": self.review,
            "error_usd": self.error, "total_usd": self.total,
            "per_item_usd": self.per_item,
            "inference_share_of_total": self.inference_share,
        }


def human_baseline(items: int, assumptions: Assumptions | None = None) -> float:
    """Cost with no agent at all: a human reads every document."""
    a = assumptions or Assumptions()
    return items * a.cost_full_review


def agent_cost(*, items: int, mean_tokens: float, questions_per_item: int,
               model: ModelPrice, architecture: Architecture,
               review_rate: float, error_rate_after_review: float = 0.0,
               assumptions: Assumptions | None = None) -> CostBreakdown:
    """
    Cost of running the process with an agent and a given review rate.

    `review_rate` is the fraction of items a human checks. It is an input here
    rather than an output because it is a policy choice — and the point of
    `calibration.py` is that the review rate needed to hit a given error
    tolerance depends on how well the agent ranks its own uncertainty, not on
    its accuracy.
    """
    a = assumptions or Assumptions()

    passes = questions_per_item if architecture.resends_document else 1
    tokens_in = items * mean_tokens * passes
    tokens_out = items * questions_per_item * a.output_tokens_per_clause

    inference = model.cost(tokens_in, tokens_out)
    review = items * review_rate * a.cost_verify
    error = items * questions_per_item * error_rate_after_review * a.cost_per_escaped_error

    return CostBreakdown(
        items=items, inference=inference, review=review, error=error,
        review_rate=review_rate, architecture=architecture.name,
        model=model.name,
    )


def break_even_review_rate(*, mean_tokens: float, questions_per_item: int,
                           model: ModelPrice, architecture: Architecture,
                           assumptions: Assumptions | None = None) -> float:
    """
    The review rate at which the agent stops saving anything.

    Above this, checking the agent's work costs more than doing the work.
    Because verification is faster than extraction, this sits well above zero
    — which is the useful part: an agent does not have to be trusted blindly
    to pay for itself.
    """
    a = assumptions or Assumptions()
    per_item_inference = agent_cost(
        items=1, mean_tokens=mean_tokens, questions_per_item=questions_per_item,
        model=model, architecture=architecture, review_rate=0.0,
        assumptions=a).inference

    headroom = a.cost_full_review - per_item_inference
    if headroom <= 0:
        return 0.0
    return min(1.0, headroom / a.cost_verify)


def registry() -> dict[str, Param]:
    return {k: v for k, v in globals().items() if isinstance(v, Param)}


def audit() -> str:
    """Parameter register, printed by `agentcost audit`."""
    lines = ["COST MODEL PARAMETERS", "=" * 74, ""]
    for cls in (Evidence.PUBLISHED, Evidence.MEASURED, Evidence.DERIVED,
                Evidence.ESTIMATE):
        group = {k: v for k, v in registry().items() if v.evidence == cls}
        if not group:
            continue
        lines.append(f"[{cls.value.upper()}]  {len(group)} parameter(s)")
        lines.append("-" * 74)
        for name, p in sorted(group.items()):
            lines.append(f"  {name} = {p.value}")
            lines.append(f"    source: {p.source}")
            if p.note:
                lines.append(f"    note  : {p.note}")
            lines.append("")
    a = Assumptions()
    lines += [
        "=" * 74,
        f"  full human review per contract : ${a.cost_full_review:,.2f}",
        f"  verifying agent output         : ${a.cost_verify:,.2f}",
        "",
        "  The ratio between those two, and the review rate, decide the",
        "  answer. Model price does not.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(audit())
