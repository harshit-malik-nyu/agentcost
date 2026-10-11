# Decision assurance: agentcost

Prefer a bounded triage evaluation over blanket auto-acceptance. Measure exception-review time and error exposure at contract level before paying fixed integration costs.

**Evidence:** 820 deduplicated CUAD outputs contain 38 errors. Contract-grouped retrospective validation yields mean trained lift 8.45 versus raw lift 0.00; this small sample cannot establish production ROI.

**Uncertainty:** Clause correctness is not legal-loss severity; assumed incident rates and dollar losses are not estimates from the 38 errors.

## Robustness of the assumed business case

| Alternative | Feasible grid cases / 27 | Worst regret when feasible | Maximum volume at fixed review capacity |
|---|---:|---:|---:|
| Review every clause | 6 | $73,175.00 | 18000 |
| Calibrated triage with exception review | 9 | $0.00 | 41739 |
| Aggressive auto-acceptance | 0 | No feasible case | 180000 |

**Robust minimax choice:** None. No option is feasible across the entire stress grid.

Finite assumed grid, not a probability distribution or protection outside the grid. Conditional regret never overrides feasibility. Null robust option means no unconditional recommendation. Cost regret is relative to the cheapest feasible option in that same scenario. Failure counts can overlap.

## Missing-outcome sensitivity

| Outcomes masked per arm | Primary effect bounds | Incident effect bounds | Every completion clears margins? |
|---|---:|---:|---|
| 0% | [1.4%, 1.4%] | [1.6%, 1.6%] | False |
| 5% | [-3.8%, 6.2%] | [-3.5%, 6.5%] | False |
| 10% | [-8.4%, 11.6%] | [-8.5%, 11.5%] | False |
| 20% | [-18.2%, 21.8%] | [-18.9%, 21.1%] | False |

These are identification bounds on deliberately masked synthetic outcomes, not confidence intervals or measured missingness. Failures and non-starters remain in the assigned denominator. Passing this sensitivity check would still require statistical and domain-quality gates.

## Domain measurement contract

**Decision:** Choose review-all or calibrated triage for the same eligible contract workload.

**Primary outcome:** Total inference, review, QA and rework cost per eligible contract, with accepted-error risk reported independently.

**Denominator:** All eligible assigned contracts, including failures and retries; report clause-level errors separately from contract cost.

**Independent unit:** Contract for retrospective validation; reviewer team for a workflow trial if teammates share routing or training.

**Quality guardrail:** Accepted-error risk bound, review capacity and material-error rate. Do not infer incident dollar severity from clause labels.

**Minimum required data:** Contract and clause IDs, assignment, model/policy version, review minutes, inference charges, QA/rework events, truth provenance and material-error adjudication.

**Missingness and failure handling:** Keep failed contracts and retry costs. Unadjudicated outputs remain unknown; compare conservative error bounds before auto-acceptance.

**Release rule:** Require quality gates and lower review-adjusted cost under measured workload; the assumed economic model alone does not approve automation.

**Reversal trigger:** Exception time, residual error or integration cost eliminates the advantage, or no threshold meets the risk ceiling.

## Interview defense

**Is 8.45 lift a financial return?**

No. It is a retrospective model-validation result. Financial return additionally needs measured review effort, deployment cost and error exposure.

**Why split by contract?**

Clauses from one contract share context. Splitting those clauses across training and validation can leak information and overstate generalization.

## Implementation handoff

1. Freeze the eligible population, denominator, outcome horizon and proposed decision owner before comparison.
2. Reconcile assignment and outcome records at the independence unit; classify missingness without deleting cases.
3. Estimate the domain outcome and its uncertainty separately from activation. Apply risk and capacity vetoes before comparing costs.
4. Record approve, hold, or reverse with the evidence version and violated gate. An unmeasured required domain gate means HOLD.

The supplied cluster-budget grid is a planning sensitivity check with assumed ICC. It is not an estimate from real clusters and does not guarantee joint power. No stakeholder approval or real experiment is asserted.
