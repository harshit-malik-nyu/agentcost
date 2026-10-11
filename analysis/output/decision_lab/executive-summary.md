# Buy automation only when review-adjusted cost and risk both improve

Prefer a bounded triage evaluation over blanket auto-acceptance. Measure exception-review time and error exposure at contract level before paying fixed integration costs.

**Observed evidence:** 820 deduplicated CUAD outputs contain 38 errors. Contract-grouped retrospective validation yields mean trained lift 8.45 versus raw lift 0.00; this small sample cannot establish production ROI.

**Decision owner:** Legal operations lead (proposed role, not a client).

**Base planning case:** Calibrated triage with exception review. This is conditional on the assumptions, not deployment approval.

| Alternative | Economic cost | Cash outlay | Reviewer hours | Passes risk and capacity |
|---|---:|---:|---:|---|
| Review every clause | $260,000 | $0 | 2,666.7 | False |
| Calibrated triage with exception review | $163,100 | $19,600 | 1,150.0 | True |
| Aggressive auto-acceptance | $214,600 | $10,600 | 266.7 | False |

Economic cost includes valued staff capacity and scenario incident losses. Cash outlay excludes salaried capacity and contingent losses. These are not realized savings.

**Sensitivity across 27 equally enumerated scenarios:** Calibrated triage with exception review: 9, Review every clause: 3, NO FEASIBLE OPTION: 15. Grid counts are not probabilities.

**Synthetic product experiment:** HOLD. Primary effect 1.4%; 95% interval [-2.4%, 5.2%]. Guardrail effect 1.6%; interval [0.8%, 2.5%]. These numbers test the decision logic, not product performance.

**Priority order:**

1. Block auto-acceptance when no simultaneous risk bound is feasible.
2. Compare review-all and calibrated triage at the same contract workload, including QA and integration costs.
3. Expand automation only if observed review time and residual error stay below the predeclared gates.

**What would reverse the recommendation:** Review-all wins at low volume or if exception review consumes the apparent savings; no option should ship if the risk ceiling fails.

**Remaining uncertainty:** Clause correctness is not legal-loss severity; assumed incident rates and dollar losses are not estimates from the 38 errors.
