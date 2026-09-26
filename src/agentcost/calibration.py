"""
How much of the agent's work a human still has to check.

`economics.py` shows that the review rate decides the cost. This decides the
review rate, and the answer is not what the automation literature implies.

The usual question is "how accurate is the model". The operationally binding
question is different:

    Given an error tolerance, what fraction of items must a human look at?

Those come apart completely, and the gap between them is the whole argument.

    An agent that is 95% accurate but cannot tell which 5% is wrong needs
    100% review. It saves nothing.

    An agent that is 85% accurate and reliably flags its own uncertainty may
    need 20% review. It saves most of the budget.

The second agent is worse by every benchmark and better in production. What
separates them is not capability but **calibration**: whether confidence ranks
errors to the top of the queue.

The capture curve
-----------------
Rank every answer by the agent's confidence, ascending. Walk down the list and
record what share of all errors has been caught by each point. That curve is
the instrument:

    perfect ranking   every error sits below every correct answer, so
                      reviewing the error rate catches everything
    no ranking        errors are spread uniformly, so catching 90% of them
                      means reviewing 90% of the work
    real agents       somewhere between, and where exactly is the number
                      that decides the business case

This is a lift curve under another name. It is standard in credit and fraud
scoring and essentially absent from discussion of document-extraction agents,
which are sold on accuracy alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Answer:
    """One agent output, scored against expert ground truth."""

    item_id: str
    clause_type: str
    confidence: float
    """Agent's own confidence, 0–1. Any monotone score works; only the
    ranking matters."""
    correct: bool


@dataclass
class CaptureCurve:
    """Share of errors caught as review depth increases."""

    review_rates: list[float] = field(default_factory=list)
    errors_caught: list[float] = field(default_factory=list)
    n_answers: int = 0
    n_errors: int = 0

    @property
    def accuracy(self) -> float:
        if not self.n_answers:
            return 0.0
        return 1 - self.n_errors / self.n_answers

    def review_rate_for(self, tolerance: float) -> float:
        """
        Review depth needed to catch enough errors to leave at most
        `tolerance` of the original error mass unreviewed.

        Returns 1.0 when no depth short of everything achieves it, which is
        the honest answer for an uncalibrated agent rather than an
        extrapolation.
        """
        target = 1 - tolerance
        for rate, caught in zip(self.review_rates, self.errors_caught):
            if caught >= target:
                return rate
        return 1.0

    @property
    def lift(self) -> float:
        """
        How much better than chance the ranking is, at the depth equal to the
        error rate.

        A value of 1.0 means confidence carries no information about
        correctness — reviewing by confidence is no better than reviewing at
        random, and the agent cannot be trusted to triage its own work.
        """
        if not self.n_answers or not self.n_errors:
            return 1.0
        depth = self.n_errors / self.n_answers
        caught = next((c for r, c in zip(self.review_rates, self.errors_caught)
                       if r >= depth), 1.0)
        return caught / depth if depth else 1.0

    def as_dict(self) -> dict:
        return {
            "n_answers": self.n_answers, "n_errors": self.n_errors,
            "accuracy": self.accuracy, "lift_at_error_rate": self.lift,
            "review_rate_for_90pct_capture": self.review_rate_for(0.10),
            "review_rate_for_95pct_capture": self.review_rate_for(0.05),
            "review_rate_for_99pct_capture": self.review_rate_for(0.01),
            "curve": [{"review_rate": r, "errors_caught": c}
                      for r, c in zip(self.review_rates, self.errors_caught)],
        }


def capture_curve(answers: list[Answer], steps: int = 100) -> CaptureCurve:
    """
    Build the curve by ranking answers from least to most confident.

    Ties are broken by original order rather than shuffled, which is the
    pessimistic choice: an agent emitting one confidence value for everything
    gets no credit for accidental ordering.
    """
    if not answers:
        return CaptureCurve()

    ordered = sorted(enumerate(answers), key=lambda t: (t[1].confidence, t[0]))
    total = len(ordered)
    total_errors = sum(1 for _, a in ordered if not a.correct)

    curve = CaptureCurve(n_answers=total, n_errors=total_errors)
    if total_errors == 0:
        # Nothing to catch. Review rate zero suffices at any tolerance, and
        # saying so beats dividing by zero.
        curve.review_rates = [0.0, 1.0]
        curve.errors_caught = [1.0, 1.0]
        return curve

    caught = 0
    for i, (_, a) in enumerate(ordered, start=1):
        if not a.correct:
            caught += 1
        if i % max(1, total // steps) == 0 or i == total:
            curve.review_rates.append(i / total)
            curve.errors_caught.append(caught / total_errors)
    return curve


def random_baseline(error_rate: float, steps: int = 100) -> CaptureCurve:
    """
    The curve an agent with no useful confidence signal produces.

    Reviewing r of the work catches r of the errors. Comparing a measured
    curve against this is the test of whether confidence means anything, and
    it is the comparison vendors do not publish.
    """
    curve = CaptureCurve(n_answers=steps, n_errors=max(1, int(error_rate * steps)))
    for i in range(1, steps + 1):
        curve.review_rates.append(i / steps)
        curve.errors_caught.append(i / steps)
    return curve


@dataclass
class ReviewPolicy:
    """A review depth, costed."""

    tolerance: float
    review_rate: float
    errors_escaping: float
    cost_per_item: float
    saving_vs_human: float
    saving_pct: float

    def as_dict(self) -> dict:
        return {
            "error_tolerance": self.tolerance,
            "review_rate_required": self.review_rate,
            "errors_escaping_per_item": self.errors_escaping,
            "cost_per_item_usd": self.cost_per_item,
            "saving_vs_human_usd": self.saving_vs_human,
            "saving_pct": self.saving_pct,
        }


def policy_for(curve: CaptureCurve, tolerance: float, *,
               mean_tokens: float, questions_per_item: int,
               model, architecture, assumptions=None) -> ReviewPolicy:
    """
    What hitting an error tolerance costs, given this agent's calibration.

    Joins the two halves: the curve says how deep review must go, the cost
    model says what that depth costs.

    On the comparison being made
    ----------------------------
    The first version of this charged the agent for errors that escaped
    review while treating the human baseline as error-free. That is not a
    comparison — human contract review has a well-documented error rate of its
    own, and the resulting "savings" came out negative because the agent was
    being billed for a standard nobody actually meets.

    The tolerance is now a CONSTRAINT rather than a cost. Both paths are
    required to hit the same quality bar, so the error term is common to both
    and cancels; what differs is the inference and review needed to get there.
    That is the comparison a buyer is actually making, and it does not require
    an estimate of what a missed clause costs — which is the weakest number in
    the model and is now out of the headline entirely.
    """
    from .economics import Assumptions, agent_cost, human_baseline

    a = assumptions or Assumptions()
    rate = curve.review_rate_for(tolerance)
    escaping = (1 - curve.accuracy) * tolerance * questions_per_item

    # error_rate_after_review is deliberately zero: both paths are held to the
    # same tolerance, so escaped-error cost is common and cancels.
    cost = agent_cost(
        items=1, mean_tokens=mean_tokens, questions_per_item=questions_per_item,
        model=model, architecture=architecture, review_rate=rate,
        error_rate_after_review=0.0, assumptions=a,
    )
    baseline = human_baseline(1, a)

    return ReviewPolicy(
        tolerance=tolerance, review_rate=rate, errors_escaping=escaping,
        cost_per_item=cost.per_item,
        saving_vs_human=baseline - cost.per_item,
        saving_pct=(baseline - cost.per_item) / baseline if baseline else 0.0,
    )


def simulate(accuracy: float, calibration: float, n: int = 4182,
             seed: int = 0) -> list[Answer]:
    """
    Construct answers with known accuracy and known calibration quality.

    This characterises the instrument — it answers "would the curve detect
    this if it were true" — and is never reported as a finding. Numbers
    presented as findings come from `capture_curve` over real agent output.

    `calibration` runs 0 to 1: at 0 confidence is independent of correctness,
    at 1 every error is ranked below every correct answer.
    """
    import random

    rng = random.Random(seed)
    out = []
    for i in range(n):
        correct = rng.random() < accuracy
        base = rng.random()
        # Pull errors downward in proportion to the calibration parameter.
        conf = base * (1 - calibration) + (calibration * (0.9 if correct else 0.1))
        out.append(Answer(item_id=f"doc-{i // 41}", clause_type=f"c{i % 41}",
                          confidence=conf, correct=correct))
    return out
