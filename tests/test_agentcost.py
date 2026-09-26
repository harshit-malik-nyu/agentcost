"""
Tests for agentcost.

These pin the properties the conclusions rest on, and several pin an error
that was actually made. A cost model is unusually easy to get wrong in a way
that looks right: every version produces a plausible dollar figure.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentcost.calibration import (
    Answer, capture_curve, policy_for, random_baseline, simulate,
)
from agentcost.economics import (
    BATCHED, PER_QUESTION, PRICE_TIERS, Assumptions, Evidence, agent_cost,
    break_even_review_rate, human_baseline, registry,
)

ROOT = Path(__file__).resolve().parents[1]
MID = next(m for m in PRICE_TIERS if m.name == "mid")


@pytest.fixture(scope="module")
def corpus():
    path = ROOT / "data" / "cuad_stats.json"
    if not path.exists():
        pytest.skip("corpus statistics not vendored")
    return json.loads(path.read_text())


# ===========================================================================
# Corpus
# ===========================================================================

class TestCorpus:

    def test_expert_annotations_are_present(self, corpus):
        assert corpus["contracts"] == 102
        assert corpus["questions_per_contract"] == 41
        assert len(corpus["clause_prevalence"]) == 41

    def test_clause_prevalence_spans_the_full_range(self, corpus):
        """
        Prevalence runs from 0% to 100%, which is why an aggregate accuracy
        figure across all 41 clause types says almost nothing: a model that
        always answers 'not present' is perfect on some and useless on others.
        """
        rates = [v["rate"] for v in corpus["clause_prevalence"].values()]
        assert min(rates) < 0.02
        assert max(rates) > 0.98

    def test_majority_class_baseline_is_high(self, corpus):
        """
        Answering 'not present' to everything scores about 70%. Any accuracy
        claim has to be read against that floor, and most are not.
        """
        total = sum(v["asked"] for v in corpus["clause_prevalence"].values())
        present = sum(v["present"] for v in corpus["clause_prevalence"].values())
        assert 0.65 < 1 - present / total < 0.75


# ===========================================================================
# Cost structure
# ===========================================================================

class TestCostStructure:

    def test_inference_is_a_rounding_error(self, corpus):
        """
        The central finding. Vendor comparison is almost entirely about token
        price; token price is well under 2% of the cost of the work.
        """
        c = agent_cost(items=102, mean_tokens=11712, questions_per_item=41,
                       model=MID, architecture=BATCHED, review_rate=1.0)
        assert c.inference_share < 0.02

    def test_the_finding_holds_across_every_price_tier(self):
        """If it only held for cheap models it would be an artefact."""
        for m in PRICE_TIERS:
            c = agent_cost(items=102, mean_tokens=11712, questions_per_item=41,
                           model=m, architecture=BATCHED, review_rate=1.0)
            assert c.inference_share < 0.05, f"{m.name} breaks the finding"

    def test_batching_cuts_inference_by_an_order_of_magnitude(self):
        """
        Same work, same answers. Resending the contract for each of 41
        questions is the obvious implementation and costs thirty times more.
        """
        per_q = agent_cost(items=102, mean_tokens=11712, questions_per_item=41,
                           model=MID, architecture=PER_QUESTION, review_rate=0)
        batched = agent_cost(items=102, mean_tokens=11712, questions_per_item=41,
                             model=MID, architecture=BATCHED, review_rate=0)
        assert per_q.inference / batched.inference > 10

    def test_batching_does_not_change_the_conclusion(self):
        """
        Even the wasteful architecture leaves inference a minority of cost,
        which is why the headline is about review rather than engineering.
        """
        c = agent_cost(items=102, mean_tokens=11712, questions_per_item=41,
                       model=MID, architecture=PER_QUESTION, review_rate=1.0)
        assert c.inference_share < 0.10

    def test_review_scales_with_the_review_rate(self):
        low = agent_cost(items=100, mean_tokens=10000, questions_per_item=41,
                         model=MID, architecture=BATCHED, review_rate=0.10)
        high = agent_cost(items=100, mean_tokens=10000, questions_per_item=41,
                          model=MID, architecture=BATCHED, review_rate=0.90)
        assert high.review == pytest.approx(9 * low.review)

    def test_break_even_exceeds_zero_because_verifying_beats_extracting(self):
        """
        An agent does not have to be trusted blindly to pay for itself. This
        is the number that makes 'human in the loop' economically coherent
        rather than a hedge.
        """
        be = break_even_review_rate(mean_tokens=11712, questions_per_item=41,
                                    model=MID, architecture=BATCHED)
        assert be > 0.5

    def test_zero_items_does_not_divide_by_zero(self):
        c = agent_cost(items=0, mean_tokens=10000, questions_per_item=41,
                       model=MID, architecture=BATCHED, review_rate=0.5)
        assert c.per_item == 0.0


class TestAssumptions:

    def test_sensitivity_runs_cannot_leak_state(self):
        base = Assumptions()
        trial = Assumptions(minutes_verify=999)
        assert base.minutes_verify != 999
        assert trial.minutes_verify == 999

    def test_verification_is_modelled_as_faster_than_extraction(self):
        a = Assumptions()
        assert a.cost_verify < a.cost_full_review

    def test_every_parameter_is_sourced(self):
        for name, p in registry().items():
            assert p.source, f"{name} has no source"

    def test_estimates_declare_themselves(self):
        """
        An unvalidated number must be visibly unvalidated where a reader meets
        it, not only in a summary elsewhere.
        """
        for name, p in registry().items():
            if p.evidence is Evidence.ESTIMATE:
                assert "ESTIMATE" in p.source.upper(), f"{name} hides it"


# ===========================================================================
# Calibration
# ===========================================================================

class TestCaptureCurve:

    def test_perfect_ranking_catches_everything_at_the_error_rate(self):
        answers = ([Answer(f"d{i}", "c", 0.1, False) for i in range(10)]
                   + [Answer(f"d{i}", "c", 0.9, True) for i in range(90)])
        curve = capture_curve(answers)
        assert curve.review_rate_for(0.0) <= 0.11

    def test_no_ranking_requires_reviewing_nearly_everything(self):
        """
        The case the whole project turns on: an accurate agent that cannot
        rank its own uncertainty forces near-total review and saves little.
        """
        answers = [Answer(f"d{i}", "c", 0.5, i % 10 != 0) for i in range(1000)]
        curve = capture_curve(answers)
        assert curve.review_rate_for(0.05) > 0.85

    def test_lift_is_one_when_confidence_carries_no_information(self):
        answers = [Answer(f"d{i}", "c", 0.5, i % 10 != 0) for i in range(1000)]
        assert capture_curve(answers).lift == pytest.approx(1.0, abs=0.15)

    def test_lift_exceeds_one_when_confidence_ranks_errors_down(self):
        curve = capture_curve(simulate(0.85, calibration=0.85, seed=3))
        assert curve.lift > 3

    def test_ties_are_broken_pessimistically(self):
        """
        An agent emitting one confidence value for everything gets no credit
        for accidental ordering, which would otherwise flatter it.
        """
        answers = [Answer(f"d{i}", "c", 0.5, i >= 50) for i in range(100)]
        curve = capture_curve(answers)
        assert curve.review_rate_for(0.05) > 0.4

    def test_no_errors_needs_no_review(self):
        curve = capture_curve([Answer(f"d{i}", "c", 0.9, True) for i in range(50)])
        assert curve.n_errors == 0
        assert curve.review_rate_for(0.05) == 0.0

    def test_empty_input_is_safe(self):
        assert capture_curve([]).n_answers == 0

    def test_unreachable_tolerance_returns_full_review(self):
        """
        Honest rather than extrapolated: if no depth short of everything hits
        the tolerance, the answer is everything.
        """
        answers = [Answer(f"d{i}", "c", 0.5, i % 2 == 0) for i in range(100)]
        assert capture_curve(answers).review_rate_for(0.0) <= 1.0

    def test_random_baseline_is_the_diagonal(self):
        base = random_baseline(0.15)
        for r, c in zip(base.review_rates, base.errors_caught):
            assert r == pytest.approx(c, abs=1e-9)


class TestPolicy:

    ARGS = dict(mean_tokens=11712, questions_per_item=41,
                model=MID, architecture=BATCHED)

    def test_calibration_beats_accuracy(self):
        """
        THE FINDING. A worse model that knows when it is wrong costs less to
        run than a better model that does not.
        """
        blind = policy_for(capture_curve(simulate(0.95, 0.0, seed=7)), 0.05, **self.ARGS)
        calibrated = policy_for(capture_curve(simulate(0.85, 0.85, seed=7)), 0.05, **self.ARGS)
        assert calibrated.review_rate < blind.review_rate
        assert calibrated.cost_per_item < blind.cost_per_item

    def test_tolerance_is_a_constraint_not_a_cost(self):
        """
        REGRESSION. The first version charged the agent for escaped errors
        while treating the human baseline as error-free, which is not a
        comparison — it produced negative savings for every agent, because it
        was billing against a standard nobody meets.

        Both paths are now held to the same tolerance, so the error term is
        common and cancels.
        """
        pol = policy_for(capture_curve(simulate(0.90, 0.7, seed=1)), 0.05, **self.ARGS)
        assert pol.saving_pct > 0
        assert pol.cost_per_item < human_baseline(1)

    def test_a_tighter_tolerance_costs_more(self):
        curve = capture_curve(simulate(0.90, 0.7, seed=1))
        loose = policy_for(curve, 0.20, **self.ARGS)
        tight = policy_for(curve, 0.01, **self.ARGS)
        assert tight.review_rate >= loose.review_rate
        assert tight.cost_per_item >= loose.cost_per_item

    def test_savings_are_reported_against_the_human_baseline(self):
        pol = policy_for(capture_curve(simulate(0.90, 0.8, seed=2)), 0.05, **self.ARGS)
        assert pol.saving_vs_human == pytest.approx(
            human_baseline(1) - pol.cost_per_item)


class TestSimulation:

    def test_simulation_hits_the_requested_accuracy(self):
        answers = simulate(0.85, 0.5, n=4000, seed=11)
        acc = sum(a.correct for a in answers) / len(answers)
        assert 0.82 < acc < 0.88

    def test_calibration_parameter_moves_the_lift(self):
        low = capture_curve(simulate(0.85, 0.0, n=4000, seed=5)).lift
        high = capture_curve(simulate(0.85, 0.9, n=4000, seed=5)).lift
        assert high > low * 2

    def test_simulation_is_documented_as_not_a_finding(self):
        """
        It characterises the instrument. Numbers reported as findings come
        from real agent output, and the docstring has to say so.
        """
        import agentcost.calibration as cal
        doc = " ".join(cal.simulate.__doc__.split())
        assert "never reported as a finding" in doc
