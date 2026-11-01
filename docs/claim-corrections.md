# Public claim reconciliation

This revision corrects the public entry point; it does not change the frozen model observations, retrain a model or assert new business outcomes. Earlier content remains available in Git history.

| Earlier claim or ambiguity | Correction | Basis |
|---|---|---|
| “30× difference” alongside $41.69 and $3.46 | The displayed numbers imply about 12.05×, not 30×. The current front page omits this legacy architecture headline. | Direct arithmetic; no new vendor-price or architecture-equivalence claim |
| Model inference and review figures framed as the cost of running the process | Separate illustrative assumptions from stored observations; no measured operational savings | `decision_spec.json` and activity-based economic output |
| Trained-lift “direction is consistent across folds” | Explicitly expose zero-lift folds and the 0–18.22 range | `evidence/mechanism_grouped.json` |
| “A trained calibrator ... was not tried” after discussing trained validation | Remove the stale statement and distinguish the performed retrospective ranker from unperformed production calibration | Grouped mechanism artifact and current decision brief |
| Guarantee language independent of model quality could read as unconditional deployment safety | Explicitly retain sampling, fixed-policy, calibration and selection assumptions | Current corrected threshold implementation and decision brief |
| “No dependencies” immediately after a developer installation command | Standard-library offline analysis is separate from package development and model remeasurement dependencies | `pyproject.toml` and analysis imports |
| Cost crossing could imply switching to review-all | Evaluate feasibility separately: review-all exceeds the declared annual capacity at base volume | New review-time boundary tests against the existing economic model |

## Prevent recurrence

`analysis/build_readme.py` renders the entry point from committed observations and current economic output. CI rebuilds it and fails on drift. The script verifies unique model/contract/clause keys and reconciles counts with the mechanism artifact. This is protection against these specific inconsistencies, not an automated fact-checker for every sentence.

## Business acceptance for the boundary analysis

| ID | Requirement | Observable acceptance |
|---|---|---|
| AC-B1 | Base boundary case agrees with current economics | 9-minute case: $163,100 economic cost, $19,600 cash outlay, 1,150 review hours |
| AC-B2 | Economic crossing and feasibility stay distinct | On either side of cost equality, over-capacity review-all remains unselectable |
| AC-B3 | Capacity failure produces no feasible option | A value just above the capacity boundary blocks every option |
| AC-B4 | Lower review cost cannot override risk | Setting triage incident risk above the ceiling blocks it even at zero review minutes |
| AC-B5 | Capacity value is not automatically cash savings | Changing review time changes valued cost but not the salaried-capacity cash classification |

Each requirement maps to a named test in `analysis/test_review_boundary.py`. Acceptance is technical and bounded. No stakeholder sign-off, measured review-time distribution or approved risk threshold is implied.
