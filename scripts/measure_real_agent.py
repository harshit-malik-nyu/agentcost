#!/usr/bin/env python3
"""
Measure a real extraction model's calibration on CUAD.

This is the piece the rest of the project assumes. `calibration.py` shows that
the review rate, and therefore the cost, is decided by whether an agent's
confidence ranks its errors. Everything reported so far about that used
constructed answers with a calibration knob, which demonstrates a mechanism and
establishes nothing about any real system.

This runs an actual model — RoBERTa fine-tuned on CUAD — over actual contracts
and records, per question, what it answered, how confident it was, and whether
it was right. That produces a real capture curve and a real risk-controlled
threshold.

Design notes
------------
**Time budget with checkpointing.** CI runners are time-limited and a job that
overruns produces nothing at all. Results are flushed after every contract, so
a truncated run still yields a usable sample and says how far it got. A partial
measurement is worth far more than a failed one.

**Correctness is span overlap, not exact match.** CUAD answers are highlighted
passages; a prediction that captures the operative clause but trims a leading
article is right in every sense that matters to a reviewer. Exact string match
would score the model far below its usefulness and make the calibration curve
measure tokenisation rather than comprehension.

**Impossible answers are the majority class.** 70.3% of questions have no
answer in the contract, so the model must be allowed to decline, and declining
correctly counts as correct. Scoring only the answerable subset would throw
away the majority of the task.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
DATA_URL = "https://raw.githubusercontent.com/The-Atticus-Project/cuad/main/data.zip"
DEFAULT_MODEL = "akdeniz27/roberta-base-cuad"


def fetch_corpus(cache: Path) -> list:
    """Download the CUAD test split, once."""
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "data.zip"
    if not archive.exists():
        print(f"fetching {DATA_URL} ...", flush=True)
        archive.write_bytes(urlopen(DATA_URL, timeout=300).read())
    with zipfile.ZipFile(archive) as z:
        return json.loads(z.read("test.json"))["data"]


def normalise(text: str) -> str:
    return " ".join(text.lower().split())


def overlaps(pred: str, gold: str, threshold: float = 0.5) -> bool:
    """
    Token-level Jaccard overlap.

    A reviewer checking a highlighted span cares whether the operative text is
    captured, not whether the boundaries match to the character. Exact match
    would measure tokenisation.
    """
    p, g = set(normalise(pred).split()), set(normalise(gold).split())
    if not p or not g:
        return False
    return len(p & g) / len(p | g) >= threshold


def score_prediction(pred_text: str, pred_score: float, gold_texts: list[str],
                     min_conf: float) -> bool:
    """
    Was the model right?

    Declining when there is nothing to find counts as correct — it is 70% of
    the task and a reviewer would call it the right answer.
    """
    declined = (not pred_text.strip()) or pred_score < min_conf
    if not gold_texts:
        return declined
    if declined:
        return False
    return any(overlaps(pred_text, g) for g in gold_texts)


def run(model_name: str, limit_contracts: int, budget_s: float,
        out_path: Path, cache: Path, min_conf: float) -> dict:
    from transformers import pipeline

    docs = fetch_corpus(cache)
    print(f"corpus: {len(docs)} contracts", flush=True)

    print(f"loading {model_name} (CPU) ...", flush=True)
    qa = pipeline("question-answering", model=model_name, device=-1)

    started = time.time()
    records: list[dict] = []
    contracts_done = 0
    truncated = False

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"budget {budget_s:.0f}s for up to {limit_contracts} contracts",
          flush=True)
    for doc in docs[:limit_contracts]:
        if time.time() - started > budget_s:
            truncated = True
            print(f"time budget reached after {contracts_done} contracts",
                  flush=True)
            break

        for para in doc["paragraphs"]:
            context = para["context"]
            print(f"  contract {contracts_done + 1}: {len(context):,} chars, "
                  f"{len(para['qas'])} questions", flush=True)
            for qa_item in para["qas"]:
                if time.time() - started > budget_s:
                    truncated = True
                    break
                clause = qa_item["id"].split("__")[-1]
                gold = [a["text"] for a in qa_item.get("answers", [])]
                try:
                    res = qa(question=qa_item["question"], context=context,
                             handle_impossible_answer=True, max_answer_len=200,
                             max_seq_len=384, doc_stride=128)
                except Exception as exc:          # noqa: BLE001
                    print(f"  skipped {clause}: {type(exc).__name__}", flush=True)
                    continue

                text, conf = res.get("answer", ""), float(res.get("score", 0.0))
                records.append({
                    "contract": doc.get("title", f"doc-{contracts_done}"),
                    "clause": clause,
                    "confidence": conf,
                    "predicted": text[:300],
                    "gold_present": bool(gold),
                    "correct": score_prediction(text, conf, gold, min_conf),
                })

        contracts_done += 1
        # Flush after every contract: a truncated run must still leave data.
        out_path.write_text(json.dumps(records))
        print(f"  {contracts_done} contracts, {len(records)} questions, "
              f"{time.time() - started:.0f}s", flush=True)

    return {
        "model": model_name,
        "contracts_evaluated": contracts_done,
        "questions": len(records),
        "truncated": truncated,
        "seconds": time.time() - started,
        "min_confidence_to_answer": min_conf,
    }


def analyse(records: list[dict]) -> dict:
    """Capture curve and risk-controlled threshold over the real answers."""
    sys.path.insert(0, str(ROOT / "src"))
    from agentcost.calibration import Answer, capture_curve
    from agentcost.conformal import control_risk

    answers = [Answer(item_id=r["contract"], clause_type=r["clause"],
                      confidence=r["confidence"], correct=r["correct"])
               for r in records]
    curve = capture_curve(answers)

    half = len(answers) // 2
    policies = {}
    for alpha in (0.01, 0.05, 0.10):
        pol = control_risk(answers[:half], alpha=alpha, delta=0.05)
        held = answers[half:]
        accepted = [a for a in held if a.confidence >= pol.threshold]
        realised = (sum(1 for a in accepted if not a.correct) / len(accepted)
                    if accepted else 0.0)
        policies[f"alpha={alpha}"] = {
            **pol.as_dict(),
            "realised_risk_on_heldout": realised,
            "held_out_accepted": len(accepted),
            "respected_target": realised <= alpha,
        }

    majority = sum(1 for r in records if not r["gold_present"]) / max(1, len(records))
    return {
        "accuracy": curve.accuracy,
        "majority_class_baseline": majority,
        "lift_at_error_rate": curve.lift,
        "review_for_90pct_capture": curve.review_rate_for(0.10),
        "review_for_95pct_capture": curve.review_rate_for(0.05),
        "capture_curve": curve.as_dict()["curve"],
        "risk_controlled": policies,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--contracts", type=int, default=25)
    ap.add_argument("--budget", type=float, default=2400,
                    help="seconds before checkpointing and stopping")
    ap.add_argument("--min-conf", type=float, default=0.05,
                    help="below this the model is treated as declining")
    ap.add_argument("--out", default=str(ROOT / "evidence" / "real_agent.json"))
    ap.add_argument("--cache", default=str(ROOT / ".cache"))
    args = ap.parse_args()

    raw = Path(args.out).with_suffix(".records.json")
    meta = run(args.model, args.contracts, args.budget, raw,
               Path(args.cache), args.min_conf)

    records = json.loads(raw.read_text()) if raw.exists() else []
    if not records:
        print("no records produced; refusing to write an empty result",
              file=sys.stderr)
        return 1

    result = {"method": meta, **analyse(records)}
    Path(args.out).write_text(json.dumps(result, indent=2))

    print()
    print("=" * 66)
    print(f"  model              : {meta['model']}")
    print(f"  questions scored   : {meta['questions']:,} "
          f"over {meta['contracts_evaluated']} contracts"
          f"{' (truncated)' if meta['truncated'] else ''}")
    print(f"  accuracy           : {result['accuracy']:.1%}")
    print(f"  majority baseline  : {result['majority_class_baseline']:.1%}")
    print(f"  lift over random   : {result['lift_at_error_rate']:.2f}x")
    print(f"  review for 95% capture: {result['review_for_95pct_capture']:.1%}")
    print()
    for name, pol in result["risk_controlled"].items():
        print(f"  {name}: review {pol['review_rate']:.1%}, "
              f"held-out risk {pol['realised_risk_on_heldout']:.3f}, "
              f"target respected: {pol['respected_target']}")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
