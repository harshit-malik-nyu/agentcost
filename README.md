# AgentCost: evaluate automation after review cost and risk

**Decision:** should a workflow retain review-all, evaluate calibrated triage, or hold automation?

**Recommendation:** run a bounded evaluation of review time and accepted-error risk before committing integration spend. The available evidence does not establish production ROI or deployment safety.

[Executive decision](analysis/output/decision_lab/executive-summary.md) · [Review-time decision boundary](analysis/output/review_boundary/decision.md) · [Evidence and methods](analysis/README.md) · [Claim corrections](docs/claim-corrections.md)

## What is demonstrated

| Evidence class | Result | Interpretation |
|---|---|---|
| Frozen model outputs | 820 unique model/contract/clause records; 38 errors across 10 distinct contracts | Retrospective extractive-QA evidence, not a sample of independent customer deployments |
| Contract-grouped validation | Mean trained lift 8.45; raw-confidence lift 0.00 across 5 folds | Small-sample ranking result; fold lifts range 0.00–18.22, including zero-lift folds |
| Business model | Activity-based costs, cash outlay, capacity and assumed incident losses | Conditional planning arithmetic; review times, rates and risk are assumptions |
| Product analysis | Synthetic assigned-unit funnels, mature retention and guarded experiment decisions | QA of analytical mechanics; no observed adoption or causal product effect |
| Decision boundary | Separate economic break-even from operational feasibility | A cheaper or equal-cost option can still fail capacity or risk |

The stored records and [source hashes](analysis/output/manifest.json) define the observation set. Duplicate task aliases are excluded. Both model outputs for a contract stay in the same validation fold. The grouped result does not establish general calibration or a safe production threshold.

## The planning decision

The current specification assumes 20,000 cases in one planning year, 2,400 reviewer hours and a 0.6% incident-rate ceiling. These quantities and incident rates are illustrative, not measured operations.

| Alternative | Economic cost | Cash outlay | Reviewer hours | Passes assumed risk and capacity? |
|---|---:|---:|---:|---|
| Review every clause | $260,000 | $0 | 2,666.7 | No |
| Calibrated triage with exception review | $163,100 | $19,600 | 1,150.0 | Yes |
| Aggressive auto-acceptance | $214,600 | $10,600 | 266.7 | No |

Economic cost values staff capacity and includes scenario incident losses. Cash outlay excludes salaried capacity and contingent losses. A $0 cash line for review-all does **not** mean the people or work are free; it follows the model's existing-salaried-capacity classification. No modeled difference is a realized saving.

**Review-time challenge:** at a fixed 35% exception share, triage crosses the theoretical review-all cost at 18.23 minutes per exception and hits the capacity limit at 19.71 minutes. Review-all is already over capacity. Therefore the cost crossing alone does not make review-all an acceptable recommendation. At 20 minutes no option passes the planning gates. [Inspect the calculation, alternatives and tests](analysis/output/review_boundary/decision.md).

The broader [27-scenario review](analysis/output/assurance/decision-assurance.md) has no option feasible across every case. Scenario counts are not probabilities. Accepted-error bounds and domain-quality gates remain separate from cost, workflow activation and finite synthetic tests.

## What changed in the evidence

Earlier public prose mixed historical runs, illustrative economics and superseded validation. The current entry point replaces those claims with the evidence table above. The [correction record](docs/claim-corrections.md) identifies the numerical inconsistency, stale validation descriptions and conditional claims that were corrected. Git history retains the earlier versions.

Threshold-selection guarantees require the documented sampling, policy, calibration and multiple-testing assumptions; they are not unconditional assurances about arbitrary future data. See [the current decision brief](docs/decision-brief.md) and [the case against the analysis](docs/against.md).

## Reproduce without external services

Use committed inputs and Python's standard library:

```sh
python analysis/build.py
python analysis/test_decision_lab.py
python analysis/decision_lab.py
python analysis/test_review_boundary.py
python analysis/review_boundary.py
python analysis/build_readme.py
```

For the package's wider test suite, install its development dependencies with `pip install -e ".[dev]"` and run `pytest -q`. That command installs packages; it is not dependency-free. Model remeasurement and corpus downloads are separate historical-reproduction paths and are not required to rebuild the committed offline analysis.

The installed CLI also exposes `agentcost breakdown`, `agentcost policy` and `agentcost audit`; their outputs must be read with the assumptions and model scope, not as verified vendor prices or observed savings.

## Boundaries and ownership

This is an independent, AI-assisted analytical work sample. No client engagement, stakeholder approval, legal advice, production integration or realized business impact is claimed. The contract sample is small, contains only 38 errors, and uses truncated contexts. Historical model output is not a substitute for an independent production calibration set.

Proposed ownership: the workflow owner defines capacity and acceptable risk; the analyst maintains denominators and the assumption register; an appropriately qualified domain reviewer adjudicates errors. These are proposed responsibilities, not participants who accepted this project.

Source corpus: [CUAD / The Atticus Project](https://github.com/TheAtticusProject/cuad). Repository license: MIT. Historical source data retain their own license and attribution requirements.
