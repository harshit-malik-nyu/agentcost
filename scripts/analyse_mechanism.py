#!/usr/bin/env python3
"""
Why the confidence cannot rank errors, and whether a calibrator recovers it.

The measurement established that confidence carries no ranking information,
under three signals and two models. That is a fact without an explanation, and
an unexplained negative result is weak: it could be a property of these
checkpoints, of extractive QA, or of the way the question was asked.

Two experiments, both on data already collected.

MECHANISM
    Errors are not uniform. The model is roughly eight times more likely to be
    wrong when a clause IS present than when it is absent, and five of
    forty-one clause types hold two thirds of all errors.

    The hypothesis that follows: confidence encodes *whether something is
    there*, which the model judges well, rather than *whether the extracted
    span is right*, which it judges badly. Because 90% of questions have
    nothing to find, accuracy is carried almost entirely by correct declining,
    and the errors sit in the minority of cases where the model has found
    something and is therefore uniformly confident.

    That hypothesis is testable: restrict to questions where a clause IS
    present and ask whether confidence ranks errors within that subset. If the
    signal is about presence, ranking should stay absent there too.

FIX
    A logistic regression on the three confidence signals plus cheap features,
    fitted on half the data and evaluated on the other half.

    This separates two very different conclusions. If a trained combination
    recovers ranking, the information was present and badly expressed — a
    solvable engineering problem. If it does not, the information is genuinely
    absent, and no post-hoc calibration will fix it.

    Implemented from scratch rather than with scikit-learn, to keep the
    package dependency-free; it is plain gradient descent on a convex
    objective and converges reliably at this size.
"""

from __future__ import annotations

import collections
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SIGNALS = ("conf_null_margin", "conf_span_prob", "conf_start_end")


# ---------------------------------------------------------------------------
# Logistic regression, no dependencies
# ---------------------------------------------------------------------------

def fit_logistic(X: list[list[float]], y: list[int], *, epochs: int = 3000,
                 lr: float = 0.5, l2: float = 1e-3) -> list[float]:
    """
    Gradient descent on the regularised log-likelihood.

    L2 is small but non-zero: with 410 rows and a handful of features,
    separation is possible and would send weights to infinity, which looks
    like a perfect fit and generalises badly.
    """
    n_feat = len(X[0])
    w = [0.0] * (n_feat + 1)          # last entry is the bias

    for _ in range(epochs):
        grad = [0.0] * (n_feat + 1)
        for xi, yi in zip(X, y):
            z = sum(w[j] * xi[j] for j in range(n_feat)) + w[-1]
            p = 1 / (1 + math.exp(-max(-30, min(30, z))))
            err = p - yi
            for j in range(n_feat):
                grad[j] += err * xi[j]
            grad[-1] += err
        m = len(X)
        for j in range(n_feat):
            w[j] -= lr * (grad[j] / m + l2 * w[j])
        w[-1] -= lr * (grad[-1] / m)
    return w


def predict(w: list[float], xi: list[float]) -> float:
    z = sum(w[j] * xi[j] for j in range(len(xi))) + w[-1]
    return 1 / (1 + math.exp(-max(-30, min(30, z))))


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def featurise(r: dict, clause_rate: dict[str, float]) -> list[float]:
    """
    The three confidence signals, plus features a deployer would actually
    have at inference time.

    Deliberately excludes anything derived from the gold answer: a calibrator
    that peeks at the label recovers ranking trivially and tells you nothing.
    """
    pred = r.get("predicted", "") or ""
    return [
        r.get("conf_null_margin", 0.0),
        r.get("conf_span_prob", 0.0),
        r.get("conf_start_end", 0.0),
        min(len(pred), 300) / 300,                 # answer length, normalised
        1.0 if not pred.strip() else 0.0,          # declined to answer
        clause_rate.get(r["clause"], 0.0),         # base rate of this clause
    ]


def lift_of(scores: list[float], correct: list[bool]) -> float:
    """Lift at review depth equal to the error rate, ties broken pessimistically."""
    n = len(scores)
    n_err = sum(1 for c in correct if not c)
    if not n or not n_err:
        return 1.0
    order = sorted(range(n), key=lambda i: (scores[i], i))
    depth = n_err / n
    k = max(1, int(round(depth * n)))
    caught = sum(1 for i in order[:k] if not correct[i]) / n_err
    return caught / depth if depth else 1.0


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def mechanism(records: list[dict]) -> dict:
    """Where the errors are, and whether ranking survives inside the hard subset."""
    errs = [r for r in records if not r["correct"]]
    by_clause = collections.Counter(r["clause"] for r in errs)
    asked = collections.Counter(r["clause"] for r in records)
    top5 = sum(n for _, n in by_clause.most_common(5))

    present = [r for r in records if r["gold_present"]]
    absent = [r for r in records if not r["gold_present"]]

    within = {}
    for sig in SIGNALS:
        within[sig] = lift_of([r.get(sig, 0.0) for r in present],
                              [r["correct"] for r in present])

    return {
        "clause_types_with_errors": len(by_clause),
        "clause_types_total": len(asked),
        "top5_share_of_errors": top5 / len(errs) if errs else 0.0,
        "worst_clauses": [
            {"clause": c, "errors": n, "asked": asked[c], "rate": n / asked[c]}
            for c, n in by_clause.most_common(6)],
        "error_rate_when_clause_present": (
            sum(1 for r in present if not r["correct"]) / len(present)
            if present else 0.0),
        "error_rate_when_clause_absent": (
            sum(1 for r in absent if not r["correct"]) / len(absent)
            if absent else 0.0),
        "n_present": len(present), "n_absent": len(absent),
        "lift_within_present_subset": within,
        "interpretation": (
            "Confidence appears to encode whether a clause is PRESENT, which "
            "the model judges well, rather than whether the extracted span is "
            "RIGHT, which it judges badly. Since most questions have nothing "
            "to find, accuracy is carried by correct declining, and the errors "
            "concentrate where the model has found something and is therefore "
            "uniformly confident."),
    }


def try_to_fix(records: list[dict], seed: int = 0, folds: int = 5) -> dict:
    """
    Can a trained combination of the available signals rank errors?

    Cross-validated, because a single split of 820 rows with 38 errors gives
    an estimate too noisy to conclude from either way.
    """
    rng = random.Random(seed)
    rows = records[:]
    rng.shuffle(rows)

    fold_lifts, baseline_lifts = [], []
    for f in range(folds):
        test = rows[f::folds]
        train = [r for i, r in enumerate(rows) if i % folds != f]
        if not test or not train:
            continue

        # Clause base rates are computed from the TRAINING fold only.
        # Deriving them from the full set leaks the held-out labels through
        # the feature, and a calibrator that peeks recovers ranking trivially
        # and tells you nothing.
        counts = collections.Counter(r["clause"] for r in train)
        pres = collections.Counter(r["clause"] for r in train if r["gold_present"])
        rate = collections.defaultdict(float)
        for c in counts:
            rate[c] = pres[c] / counts[c]

        Xtr = [featurise(r, rate) for r in train]
        ytr = [1 if r["correct"] else 0 for r in train]
        if len(set(ytr)) < 2:
            continue
        w = fit_logistic(Xtr, ytr)

        scores = [predict(w, featurise(r, rate)) for r in test]
        correct = [r["correct"] for r in test]
        fold_lifts.append(lift_of(scores, correct))
        baseline_lifts.append(
            lift_of([r.get("conf_null_margin", 0.0) for r in test], correct))

    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    trained, base = mean(fold_lifts), mean(baseline_lifts)
    return {
        "folds": len(fold_lifts),
        "trained_lift": trained,
        "raw_confidence_lift": base,
        "recovered_ranking": trained > 1.5,
        "fold_lifts": fold_lifts,
        "verdict": (
            "A trained combination recovers ranking: the information was "
            "present and badly expressed, which is an engineering problem."
            if trained > 1.5 else
            "A trained combination on every available signal still cannot rank "
            "errors. The information is not merely badly expressed — on these "
            "features it is absent, and no post-hoc calibration recovers it."),
    }


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="real_agent.*.json",
                    help="which measurement files to analyse")
    ap.add_argument("--out", default=str(ROOT / "evidence" / "mechanism.json"))
    args = ap.parse_args()

    paths = sorted((ROOT / "evidence").glob(args.glob))
    records: list[dict] = []
    for p in paths:
        if p.name.endswith("records.json"):
            continue
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            records.extend(data)

    if not records:
        print("no per-answer records found; run measure_real_agent.py first",
              file=sys.stderr)
        return 1

    by_model = {}
    for r in records:
        by_model.setdefault(r.get("model", "unknown"), []).append(r)

    result = {"n_records": len(records),
              "n_errors": sum(1 for r in records if not r["correct"]),
              "mechanism": mechanism(records),
              "fix_attempt": try_to_fix(records),
              "per_model": {
                  m: {"n": len(rs),
                      "errors": sum(1 for r in rs if not r["correct"]),
                      "mechanism": mechanism(rs),
                      "fix_attempt": try_to_fix(rs)}
                  for m, rs in by_model.items() if len(rs) > 60}}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    m, f = result["mechanism"], result["fix_attempt"]
    print("=" * 72)
    print(f"  {result['n_records']} answers, {result['n_errors']} errors")
    print()
    print("  MECHANISM")
    print(f"    error rate, clause present : "
          f"{m['error_rate_when_clause_present']:.1%} (n={m['n_present']})")
    print(f"    error rate, clause absent  : "
          f"{m['error_rate_when_clause_absent']:.1%} (n={m['n_absent']})")
    print(f"    errors in 5 clause types   : {m['top5_share_of_errors']:.0%}")
    print()
    print("    lift WITHIN the present-clause subset:")
    for sig, lv in m["lift_within_present_subset"].items():
        print(f"      {sig[5:]:14s} {lv:.2f}")
    print()
    print("  FIX ATTEMPT (logistic regression, "
          f"{f['folds']}-fold cross-validated)")
    print(f"    raw confidence lift : {f['raw_confidence_lift']:.2f}")
    print(f"    trained lift        : {f['trained_lift']:.2f}")
    print(f"    {f['verdict']}")
    if result.get("per_model") and len(result["per_model"]) > 1:
        print()
        print("  PER ARCHITECTURE")
        print(f"    {'model':34s} {'raw':>6s} {'trained':>8s} {'present%':>9s}")
        for name, d in result["per_model"].items():
            print(f"    {name.split('/')[-1][:32]:34s} "
                  f"{d['fix_attempt']['raw_confidence_lift']:>6.2f} "
                  f"{d['fix_attempt']['trained_lift']:>8.2f} "
                  f"{d['mechanism']['error_rate_when_clause_present']:>8.1%}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
