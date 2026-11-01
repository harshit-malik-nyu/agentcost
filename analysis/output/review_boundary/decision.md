# When does extra review reverse the decision?

Illustrative planning assumptions only. No measured review time, incident risk or cash savings.

At the fixed 35% review share, triage ties the theoretical review-all economic cost at **18.23 minutes per exception**. It reaches the 2,400-hour capacity limit at **19.71 minutes**.

**Important:** review-all already needs 2,666.7 hours at the assumed 20,000-case volume. It is infeasible at the declared capacity. Crossing its cost line is not permission to recommend it. At 19 minutes triage remains the only feasible option in this planning model; at 20 minutes no option is feasible.

| Exception minutes | Triage economic cost | Cash outlay | Review hours | Model selection |
|---:|---:|---:|---:|---|
| 9 | $163,100.00 | $19,600.00 | 1,150.0 | Calibrated triage with exception review |
| 18 | $257,600.00 | $19,600.00 | 2,200.0 | Calibrated triage with exception review |
| 19 | $268,100.00 | $19,600.00 | 2,316.7 | Calibrated triage with exception review |
| 20 | $278,600.00 | $19,600.00 | 2,433.3 | NO FEASIBLE OPTION |

One-way review-time sensitivity at fixed 35% exception share, volume, costs, QA and assumed risk. Does not optimize or validate the relationship between review effort and safety.

## Decision and reversal conditions

- Base planning case: request a bounded evaluation; do not authorize deployment from assumed incident rates.
- If review time rises, update both the activity cost and workload. Do not compare only model API spending.
- If no option passes capacity and risk, hold or revise the operating model with explicit new assumptions; do not choose the least expensive failing option.
- Salaried review hours are valued capacity, not automatically avoidable cash expenditure. Releasing time is not equivalent to a headcount saving.
- The actual risk bound, workload mix and review-time distribution must be measured separately; averages and annual capacity do not establish peak staffing or queue SLAs.

## Reproduce

`python analysis/test_review_boundary.py` then `python analysis/review_boundary.py`. Python standard library only. The tests compare Decimal-derived boundaries with the existing cost engine on both sides of each boundary and retain risk/capacity vetoes.
