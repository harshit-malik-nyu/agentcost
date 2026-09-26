"""
Choosing the review threshold with a distribution-free guarantee.

`calibration.py` builds an empirical capture curve and reads a review rate off
it. That is a lift chart, it has been standard in credit scoring since the
1990s, and it has a defect that matters here: **the threshold it selects
carries no guarantee on new data.** A threshold tuned to hit 5% error on a
validation set can exceed 5% in deployment, and nothing in the curve says by
how much or how often.

This module fixes that, using the machinery built for exactly this problem.

What is being controlled
------------------------
The quantity a buyer cares about is the **selective risk**: the error rate
among the items that are auto-accepted without human review. Items sent to
review are assumed corrected, so they contribute no error; the exposure is
entirely in what goes through untouched.

    R(λ) = P(answer is wrong | confidence ≥ λ)

The guarantee wanted is: with probability at least 1−δ over the calibration
draw, R(λ̂) ≤ α. That is a risk-controlling prediction set in the sense of
Bates et al. (2021), and the construction below is Learn-then-Test: compute a
distribution-free upper confidence bound on the risk at each candidate
threshold, and take the most permissive threshold whose bound clears α.

Why Clopper-Pearson rather than a normal approximation
------------------------------------------------------
The accepted set at a strict threshold can be small, and the error count within
it smaller still. A normal-approximation bound is unreliable exactly there —
near zero errors in a few dozen items — which is the regime a low error
tolerance puts you in. Clopper-Pearson inverts the exact binomial CDF and is
valid at any count, including zero. It is conservative, and conservative in the
right direction: it over-reviews rather than under-reviews.

Relation to prior work
----------------------
Selective prediction was formalised by Geifman and El-Yaniv (2017).
Distribution-free risk control is Bates et al. (2021), extended by Angelopoulos
et al. to conformal risk control (ICLR 2024) and Learn-then-Test (Annals of
Applied Statistics, 2025). Application to language models includes conformal
abstention for hallucination (Abbasi-Yadkori et al. 2024), prediction sets over
multiple-choice answers (Kumar et al. 2023), open-ended generation (Quach et
al. 2023), and selective conformal judging (SCOPE, 2026). Campos et al. (TACL
2024) survey the area.

**Nothing in the method here is novel.** What this module contributes is the
join: applying a standard risk-control construction to the *human-review
budget* rather than to prediction-set size, so the output is a staffing
decision with a guarantee attached rather than an accuracy number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Exact binomial upper confidence bound
# ---------------------------------------------------------------------------

def _log_binom_cdf(k: int, n: int, p: float) -> float:
    """log P(X <= k) for X ~ Binomial(n, p), computed in log space."""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return float("-inf") if k < n else 0.0

    log_p, log_q = math.log(p), math.log1p(-p)
    total = float("-inf")
    for i in range(k + 1):
        term = (math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                + i * log_p + (n - i) * log_q)
        # log-sum-exp
        if total == float("-inf"):
            total = term
        elif term > total:
            total = term + math.log1p(math.exp(total - term))
        else:
            total = total + math.log1p(math.exp(term - total))
    return total


def clopper_pearson_upper(k: int, n: int, delta: float = 0.05) -> float:
    """
    Exact upper confidence bound on a binomial rate.

    The largest p for which observing k or fewer errors in n trials would not
    be surprising at level delta. Valid at any n and any k including zero,
    which a normal approximation is not — and zero errors in a small accepted
    set is precisely the regime a strict threshold produces.
    """
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0

    target = math.log(delta)
    lo, hi = k / n, 1.0
    for _ in range(200):                      # bisection to ~1e-15
        mid = (lo + hi) / 2
        if _log_binom_cdf(k, n, mid) > target:
            lo = mid
        else:
            hi = mid
    return hi


# ---------------------------------------------------------------------------
# Risk-controlling threshold
# ---------------------------------------------------------------------------

@dataclass
class RiskControlledPolicy:
    """A review threshold with a distribution-free guarantee attached."""

    threshold: float
    alpha: float
    """Target selective risk: max error rate among auto-accepted items."""
    delta: float
    """The guarantee holds with probability at least 1 − delta over the
    calibration draw."""

    review_rate: float
    """Share of items sent to a human — what this costs."""

    empirical_selective_risk: float
    risk_upper_bound: float
    n_calibration: int
    n_accepted: int
    feasible: bool = True

    def as_dict(self) -> dict:
        return {
            "threshold": self.threshold, "alpha": self.alpha, "delta": self.delta,
            "review_rate": self.review_rate,
            "empirical_selective_risk": self.empirical_selective_risk,
            "risk_upper_bound": self.risk_upper_bound,
            "n_calibration": self.n_calibration, "n_accepted": self.n_accepted,
            "feasible": self.feasible,
        }


def control_risk(answers, alpha: float = 0.05, delta: float = 0.05,
                 grid: int = 200) -> RiskControlledPolicy:
    """
    Pick the most permissive review threshold whose selective risk is provably
    at most `alpha`.

    Sweeps candidate thresholds from strict to permissive and returns the one
    admitting the most auto-acceptance while its upper confidence bound still
    clears the target. Searching in that direction matters: taking the first
    threshold that clears empirically, rather than by bound, is what allows
    deployment violations.

    When no threshold is feasible — an agent whose confidence cannot isolate a
    low-risk subset at all — the policy returned reviews everything and is
    marked infeasible. That is the correct answer, and it is the one an
    empirical curve silently avoids giving.
    """
    if not answers:
        return RiskControlledPolicy(1.0, alpha, delta, 1.0, 0.0, 1.0, 0, 0, False)

    n = len(answers)

    # The candidate grid is FIXED IN ADVANCE, not read off the observed
    # confidences.
    #
    # Deriving it from the data was a real defect, caught by the coverage
    # experiment below: fixed-sequence testing only controls risk when the
    # order of hypotheses is pre-specified, and a grid built from observed
    # values is chosen after seeing the outcomes. Violations came out at 10%
    # against a 5% target — small, consistent, and exactly the signature of
    # selection on the calibration sample.
    candidates = [i / grid for i in range(grid + 1)]

    best: RiskControlledPolicy | None = None
    for lam in sorted(candidates, reverse=True):      # strict -> permissive
        accepted = [a for a in answers if a.confidence >= lam]
        if not accepted:
            continue
        errors = sum(1 for a in accepted if not a.correct)
        bound = clopper_pearson_upper(errors, len(accepted), delta)
        if bound <= alpha:
            best = RiskControlledPolicy(
                threshold=lam, alpha=alpha, delta=delta,
                review_rate=1 - len(accepted) / n,
                empirical_selective_risk=errors / len(accepted),
                risk_upper_bound=bound, n_calibration=n,
                n_accepted=len(accepted), feasible=True,
            )
        elif best is not None:
            break        # bound has crossed; the previous threshold is the pick

    if best is None:
        return RiskControlledPolicy(
            threshold=1.0 + 1e-9, alpha=alpha, delta=delta, review_rate=1.0,
            empirical_selective_risk=0.0, risk_upper_bound=0.0,
            n_calibration=n, n_accepted=0, feasible=False,
        )
    return best


def naive_threshold(answers, alpha: float = 0.05, grid: int = 200) -> float:
    """
    The threshold an empirical curve selects: the most permissive one whose
    *observed* selective risk is at or below the target, with no correction
    for having chosen it by looking at the same data.

    Included to be measured against, not recommended. `validate_coverage`
    below shows how often it fails on held-out data.
    """
    if not answers:
        return 1.0
    candidates = sorted({round(a.confidence, 6) for a in answers})
    best = 1.0 + 1e-9
    for lam in sorted(candidates, reverse=True):
        accepted = [a for a in answers if a.confidence >= lam]
        if not accepted:
            continue
        risk = sum(1 for a in accepted if not a.correct) / len(accepted)
        if risk <= alpha:
            best = lam
        else:
            break
    return best


# ---------------------------------------------------------------------------
# Coverage validation
# ---------------------------------------------------------------------------

@dataclass
class CoverageResult:
    """How often each method actually respects its target on fresh data."""

    trials: int
    alpha: float
    naive_violations: int
    conformal_violations: int
    naive_mean_risk: float
    conformal_mean_risk: float
    naive_mean_review: float
    conformal_mean_review: float
    infeasible: int = 0
    risks_naive: list[float] = field(default_factory=list)
    risks_conformal: list[float] = field(default_factory=list)

    @property
    def naive_violation_rate(self) -> float:
        return self.naive_violations / self.trials if self.trials else 0.0

    @property
    def conformal_violation_rate(self) -> float:
        return self.conformal_violations / self.trials if self.trials else 0.0

    def as_dict(self) -> dict:
        return {
            "trials": self.trials, "alpha": self.alpha,
            "naive_violation_rate": self.naive_violation_rate,
            "conformal_violation_rate": self.conformal_violation_rate,
            "naive_mean_selective_risk": self.naive_mean_risk,
            "conformal_mean_selective_risk": self.conformal_mean_risk,
            "naive_mean_review_rate": self.naive_mean_review,
            "conformal_mean_review_rate": self.conformal_mean_review,
            "infeasible_trials": self.infeasible,
        }


def validate_coverage(*, accuracy: float, calibration: float, alpha: float = 0.05,
                      delta: float = 0.05, n_cal: int = 500, n_test: int = 500,
                      trials: int = 200, seed: int = 0) -> CoverageResult:
    """
    Split-sample experiment: fit a threshold on one half, measure the risk it
    actually produces on the other.

    A note on using simulation here
    -------------------------------
    Elsewhere in this project simulation characterises the instrument and is
    never reported as a finding. This is the one place simulation IS the
    result, and legitimately so: the claim under test is a property of the
    statistical procedure — does the threshold respect its target on unseen
    data — not a claim about any model's behaviour. Validating a coverage
    guarantee by repeated sampling from a known distribution is the standard
    way to check such a procedure, because the ground truth has to be known
    for the question to be answerable at all.

    What this does NOT establish is that any real agent has useful confidence.
    That needs an evaluation run and remains open.
    """
    import random

    from .calibration import simulate

    rng = random.Random(seed)
    nv = cv = infeasible = 0
    nr, cr, nrev, crev = [], [], [], []

    for t in range(trials):
        pool = simulate(accuracy, calibration, n=n_cal + n_test,
                        seed=rng.randrange(1 << 30))
        rng.shuffle(pool)
        cal, test = pool[:n_cal], pool[n_cal:]

        def selective_risk(thr):
            acc = [a for a in test if a.confidence >= thr]
            if not acc:
                return 0.0, 1.0
            return (sum(1 for a in acc if not a.correct) / len(acc),
                    1 - len(acc) / len(test))

        n_thr = naive_threshold(cal, alpha)
        risk_n, rev_n = selective_risk(n_thr)
        nr.append(risk_n); nrev.append(rev_n)
        if risk_n > alpha:
            nv += 1

        pol = control_risk(cal, alpha, delta)
        if not pol.feasible:
            infeasible += 1
        risk_c, rev_c = selective_risk(pol.threshold)
        cr.append(risk_c); crev.append(rev_c)
        if risk_c > alpha:
            cv += 1

    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    return CoverageResult(
        trials=trials, alpha=alpha,
        naive_violations=nv, conformal_violations=cv,
        naive_mean_risk=mean(nr), conformal_mean_risk=mean(cr),
        naive_mean_review=mean(nrev), conformal_mean_review=mean(crev),
        infeasible=infeasible, risks_naive=nr, risks_conformal=cr,
    )
