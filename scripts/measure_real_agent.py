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

**Contexts are truncated, and that is a real limitation.**
Two earlier attempts ran full contracts with sliding windows and both died at
the CI timeout having committed nothing. The arithmetic: a 25,000-character
contract at stride 128 is roughly 25 windows per question, times 41 questions,
times a CPU forward pass each — minutes per contract before any of it is
useful.

Contexts are therefore truncated to the opening section. What that costs is
specific and should be read with the result: clause types that appear late in
a contract become unfindable, so measured accuracy is lower than the model
would achieve with the whole document, and the mix of clause types is skewed
toward those drafted near the front — parties, dates, term, governing law.

What it does NOT distort is the quantity this exists to measure. The capture
curve asks whether the model's confidence ranks its errors, and that is a
property of the confidence signal on whatever inputs it sees. A model with
useful calibration on truncated text has useful calibration; one without does
not acquire it from longer inputs.
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


class Extractor:
    """
    Extractive QA run directly, without the pipeline helper.

    The `question-answering` pipeline was removed from recent transformers, so
    the forward pass is done here. That turns out to be the better choice
    regardless: the null score — the model's own estimate that the clause is
    absent — is the signal this whole project is about, and running the model
    directly makes it explicit rather than leaving it to a helper's internal
    thresholding.

    Confidence is the softmax margin between the best answer span and the null
    span. It is a probability over that two-way choice, which is exactly the
    quantity a review-triage decision needs.
    """

    def __init__(self, model_name: str):
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.model.eval()

    def __call__(self, question: str, context: str, max_len: int = 384,
                 max_answer_tokens: int = 60) -> tuple[str, float]:
        torch = self.torch
        enc = self.tok(question, context, truncation="only_second",
                       max_length=max_len, return_tensors="pt",
                       return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")[0]
        seq_ids = enc.sequence_ids(0)

        with torch.no_grad():
            out = self.model(**enc)
        start, end = out.start_logits[0], out.end_logits[0]

        # Only spans inside the context are admissible answers; position 0 is
        # the CLS token and carries the null score.
        valid = [i for i, sid in enumerate(seq_ids) if sid == 1]
        if not valid:
            return "", 0.0

        null = (start[0] + end[0]).item()

        best, best_span = float("-inf"), None
        for i in valid:
            for j in valid:
                if j < i or j - i + 1 > max_answer_tokens:
                    continue
                score = (start[i] + end[j]).item()
                if score > best:
                    best, best_span = score, (i, j)

        if best_span is None:
            return "", 0.0

        # Two-way softmax between answering and declining.
        m = max(best, null)
        p_answer = pow(2.718281828, best - m)
        p_null = pow(2.718281828, null - m)
        conf = p_answer / (p_answer + p_null)

        i, j = best_span
        text = context[int(offsets[i][0]):int(offsets[j][1])]
        return text, float(conf)


def run(model_name: str, limit_contracts: int, budget_s: float,
        out_path: Path, cache: Path, min_conf: float,
        max_chars: int = 4000) -> dict:
    docs = fetch_corpus(cache)
    print(f"corpus: {len(docs)} contracts", flush=True)

    print(f"loading {model_name} (CPU) ...", flush=True)
    qa = Extractor(model_name)

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
            context = para["context"][:max_chars]
            print(f"  contract {contracts_done + 1}: {len(context):,} chars "
                  f"(of {len(para['context']):,}), {len(para['qas'])} questions",
                  flush=True)
            for qa_item in para["qas"]:
                if time.time() - started > budget_s:
                    truncated = True
                    break
                clause = qa_item["id"].split("__")[-1]
                # Gold answers outside the truncated window are unfindable by
                # construction, so they are dropped rather than counted as
                # misses the model had no chance at.
                gold = [a["text"] for a in qa_item.get("answers", [])
                        if a["text"] in context]
                try:
                    text, conf = qa(qa_item["question"], context)
                except Exception as exc:          # noqa: BLE001
                    print(f"  skipped {clause}: {type(exc).__name__}: {exc}",
                          flush=True)
                    continue
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
        "context_chars_used": max_chars,
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
    ap.add_argument("--max-chars", type=int, default=4000,
                    help="truncate each contract to this many characters")
    ap.add_argument("--min-conf", type=float, default=0.05,
                    help="below this the model is treated as declining")
    ap.add_argument("--out", default=str(ROOT / "evidence" / "real_agent.json"))
    ap.add_argument("--cache", default=str(ROOT / ".cache"))
    args = ap.parse_args()

    raw = Path(args.out).with_suffix(".records.json")
    meta = run(args.model, args.contracts, args.budget, raw,
               Path(args.cache), args.min_conf, args.max_chars)

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
    print(f"  context per doc    : {meta['context_chars_used']:,} chars "
          f"(truncated)")
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
