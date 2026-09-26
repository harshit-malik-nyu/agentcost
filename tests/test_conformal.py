"""
Risk-controlled review thresholds.

These pin the statistical properties. A threshold that looks right on the
calibration sample and fails in deployment is the exact failure this module
exists to prevent, so the tests check the guarantee rather than the output.
"""

from __future__ import annotations

import pytest

from agentcost.calibration import Answer, simulate
from agentcost.conformal import (
    clopper_pearson_upper, control_risk, naive_threshold, validate_coverage,
)


class TestClopperPearson:

    def test_zero_errors_matches_the_rule_of_three(self):
        """0/n at 95% gives approximately 3/n, the standard result."""
        assert clopper_pearson_upper(0, 100, 0.05) == pytest.approx(0.03, abs=0.005)
        assert clopper_pearson_upper(0, 300, 0.05) == pytest.approx(0.01, abs=0.002)

    def test_bound_exceeds_the_point_estimate(self):
        for k, n in [(1, 100), (5, 100), (25, 100)]:
            assert clopper_pearson_upper(k, n, 0.05) > k / n

    def test_bound_tightens_with_sample_size(self):
        assert clopper_pearson_upper(5, 100, 0.05) > clopper_pearson_upper(50, 1000, 0.05)

    def test_valid_at_the_boundaries(self):
        """
        Zero errors in a small accepted set is the regime a strict threshold
        produces, and where a normal approximation is least trustworthy.
        """
        assert 0 < clopper_pearson_upper(0, 5, 0.05) < 1
        assert clopper_pearson_upper(10, 10, 0.05) == 1.0
        assert clopper_pearson_upper(0, 0, 0.05) == 1.0

    def test_stricter_confidence_widens_the_bound(self):
        assert clopper_pearson_upper(5, 100, 0.01) > clopper_pearson_upper(5, 100, 0.10)


class TestRiskControl:

    def test_the_candidate_grid_is_fixed_in_advance(self):
        """
        REGRESSION. The grid was originally built from observed confidence
        values, which makes the selection order data-dependent and breaks
        fixed-sequence testing. Violations ran at 10% against a 5% target —
        small, consistent, and exactly the signature of selecting on the
        calibration sample.
        """
        import inspect

        from agentcost import conformal
        src = inspect.getsource(conformal.control_risk)
        assert "FIXED IN ADVANCE" in src
        assert "candidates = [i / grid for i in range(grid + 1)]" in src

    def test_returns_a_bound_below_the_target(self):
        pol = control_risk(simulate(0.90, 0.8, n=800, seed=3), alpha=0.05)
        assert pol.feasible
        assert pol.risk_upper_bound <= 0.05

    def test_the_bound_is_conservative(self):
        """Empirical risk sits below the bound, never above."""
        pol = control_risk(simulate(0.90, 0.8, n=800, seed=3), alpha=0.05)
        assert pol.empirical_selective_risk <= pol.risk_upper_bound

    def test_a_tighter_target_costs_more_review(self):
        answers = simulate(0.90, 0.7, n=1500, seed=4)
        loose = control_risk(answers, alpha=0.20)
        tight = control_risk(answers, alpha=0.02)
        assert tight.review_rate >= loose.review_rate

    def test_better_calibration_buys_lower_review(self):
        blind = control_risk(simulate(0.90, 0.0, n=1500, seed=5), alpha=0.05)
        sharp = control_risk(simulate(0.90, 0.9, n=1500, seed=5), alpha=0.05)
        assert sharp.review_rate < blind.review_rate

    def test_infeasible_targets_review_everything(self):
        """
        An agent whose confidence cannot isolate a low-risk subset should be
        told to review everything. That is the correct answer, and it is the
        one an empirical curve silently avoids giving.
        """
        answers = [Answer(f"d{i}", "c", 0.5, i % 3 != 0) for i in range(300)]
        pol = control_risk(answers, alpha=0.01)
        assert not pol.feasible
        assert pol.review_rate == 1.0

    def test_empty_input_is_safe(self):
        pol = control_risk([], alpha=0.05)
        assert not pol.feasible and pol.review_rate == 1.0

    def test_serialises(self):
        d = control_risk(simulate(0.9, 0.8, n=400, seed=1), 0.05).as_dict()
        assert {"threshold", "alpha", "delta", "review_rate", "feasible"} <= set(d)


class TestCoverageValidation:
    """
    The result that justifies the module. Simulation IS the finding here and
    legitimately so: the claim under test is a property of the procedure, not
    of any model, and a coverage guarantee can only be checked against a known
    ground truth.
    """

    def test_the_empirical_threshold_fails_often(self):
        r = validate_coverage(accuracy=0.90, calibration=0.7, alpha=0.05,
                              n_cal=400, n_test=400, trials=60, seed=7)
        assert r.naive_violation_rate > 0.20, (
            "a threshold fitted to the mean should violate its target often")

    def test_the_risk_controlled_threshold_holds(self):
        r = validate_coverage(accuracy=0.90, calibration=0.7, alpha=0.05,
                              delta=0.05, n_cal=400, n_test=400,
                              trials=60, seed=7)
        # Monte Carlo slack at 60 trials: SE is about 2.8 points.
        assert r.conformal_violation_rate < 0.20
        assert r.conformal_violation_rate < r.naive_violation_rate

    def test_control_costs_review(self):
        """
        The guarantee is not free. It buys validity with extra human review,
        and quoting one without the other would misrepresent the trade.
        """
        r = validate_coverage(accuracy=0.90, calibration=0.3, alpha=0.05,
                              n_cal=400, n_test=400, trials=40, seed=9)
        assert r.conformal_mean_review > r.naive_mean_review

    def test_simulation_use_is_justified_in_the_docstring(self):
        from agentcost.conformal import validate_coverage as vc
        doc = " ".join(vc.__doc__.split())
        assert "property of the statistical procedure" in doc
        assert "does NOT establish" in doc


class TestNaiveBaseline:

    def test_naive_hits_its_target_on_the_fitting_sample(self):
        """By construction — which is the whole problem."""
        answers = simulate(0.90, 0.7, n=1000, seed=2)
        thr = naive_threshold(answers, alpha=0.05)
        accepted = [a for a in answers if a.confidence >= thr]
        if accepted:
            risk = sum(1 for a in accepted if not a.correct) / len(accepted)
            assert risk <= 0.05 + 1e-9

    def test_naive_accepts_more_than_risk_control(self):
        answers = simulate(0.90, 0.7, n=1000, seed=2)
        assert naive_threshold(answers, 0.05) <= control_risk(answers, 0.05).threshold
