"""
Command line interface.

The CLI is what someone else would actually run, and the breakdown output is
the argument the project makes. Both are asserted rather than assumed.
"""

from __future__ import annotations

import json

import pytest

from agentcost.cli import main


class TestBreakdown:

    def test_reports_inference_as_a_minority_of_cost(self, capsys):
        assert main(["breakdown"]) == 0
        out = capsys.readouterr().out
        assert "inference" in out and "human review" in out
        assert "rounding error" in out

    def test_json_is_parseable_and_carries_the_share(self, capsys):
        assert main(["--json", "breakdown", "--review", "0.2"]) == 0
        d = json.loads(capsys.readouterr().out)
        assert d["inference_share_of_total"] < 0.05
        assert d["review_rate"] == 0.2
        assert d["human_baseline_usd"] > d["total_usd"]

    def test_lower_review_rate_costs_less(self, capsys):
        main(["--json", "breakdown", "--review", "1.0"])
        full = json.loads(capsys.readouterr().out)
        main(["--json", "breakdown", "--review", "0.1"])
        light = json.loads(capsys.readouterr().out)
        assert light["total_usd"] < full["total_usd"]

    def test_per_question_architecture_costs_more_inference(self, capsys):
        main(["--json", "breakdown", "--per-question"])
        a = json.loads(capsys.readouterr().out)
        main(["--json", "breakdown"])
        b = json.loads(capsys.readouterr().out)
        assert a["inference_usd"] > 5 * b["inference_usd"]

    def test_unknown_model_tier_is_refused(self):
        with pytest.raises(SystemExit):
            main(["breakdown", "--model", "nonsense"])


class TestPolicy:

    def test_blind_agent_is_flagged_as_unable_to_triage(self, capsys):
        """
        The warning that matters: an agent whose confidence carries no signal
        cannot triage its own work, whatever its accuracy.
        """
        main(["policy", "--accuracy", "0.95", "--calibration", "0.0"])
        out = capsys.readouterr().out
        assert "cannot triage its own work" in out

    def test_calibrated_agent_needs_less_review(self, capsys):
        main(["--json", "policy", "--accuracy", "0.85", "--calibration", "0.85"])
        good = json.loads(capsys.readouterr().out)
        main(["--json", "policy", "--accuracy", "0.95", "--calibration", "0.0"])
        blind = json.loads(capsys.readouterr().out)
        assert good["review_rate_required"] < blind["review_rate_required"]
        assert good["cost_per_item_usd"] < blind["cost_per_item_usd"]

    def test_lift_is_reported(self, capsys):
        main(["--json", "policy", "--calibration", "0.8"])
        assert json.loads(capsys.readouterr().out)["lift"] > 1


class TestAudit:

    def test_lists_parameters_and_separates_estimates(self, capsys):
        assert main(["audit"]) == 0
        out = capsys.readouterr().out
        assert "[ESTIMATE]" in out
        assert "Model price does not" in out


class TestArguments:

    def test_a_subcommand_is_required(self):
        with pytest.raises(SystemExit):
            main([])
