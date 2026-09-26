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
SQUAD_URL = ("https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/"
             "master/dataset/dev-v2.0.json")
DEFAULT_MODEL = "akdeniz27/roberta-base-cuad"

# A second task, for the generalisation test.
#
# CUAD is 90% unanswerable; SQuAD v2 is 50%. The mechanism proposed in
# analyse_mechanism.py — that confidence encodes PRESENCE rather than
# correctness, and pooled ranking collapses because presence dominates — makes
# a falsifiable prediction about that difference: pooled lift should be
# materially higher on the balanced task. If it is still zero there, the
# explanation is wrong.
#
# SQuAD v2 also runs in a fraction of the time (712-character contexts against
# 25,000), which is what makes testing several architectures affordable.
SQUAD_MODELS = [
    "deepset/roberta-base-squad2",        # same architecture, different task
    "deepset/bert-base-cased-squad2",     # different architecture
    "deepset/electra-base-squad2",        # different pre-training objective
]


def fetch_squad(cache: Path, limit_paragraphs: int = 400) -> list:
    """
    SQuAD v2 dev set, reshaped into the same structure as CUAD.

    Both are SQuAD-format already, so the harness needs no special casing —
    which is the point: the same measurement code runs on both tasks, so a
    difference in result is a difference in the data rather than in the
    instrument.
    """
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / "squad_dev_v2.json"
    if not f.exists():
        print(f"fetching {SQUAD_URL} ...", flush=True)
        f.write_bytes(urlopen(SQUAD_URL, timeout=300).read())

    raw = json.loads(f.read_text())
    docs = []
    for article in raw["data"]:
        for para in article["paragraphs"]:
            docs.append({"title": article.get("title", "squad"),
                         "paragraphs": [para]})
            if len(docs) >= limit_paragraphs:
                return docs
    return docs


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
                 max_answer_tokens: int = 60) -> tuple[str, dict[str, float]]:
        """
        Returns the predicted span and THREE confidence estimates.

        Reporting one number would conflate two questions: whether the model
        is uncalibrated, and whether the estimator is. The first measurement
        used only the null margin, which saturates by construction on a model
        trained with a null class — so "this model has no usable confidence"
        was partly a claim about the estimator.

            null_margin   softmax between the best span and the null span.
                          The practitioner default for this architecture.
            span_prob     softmax over all admissible spans, probability mass
                          on the chosen one. Measures how peaked the span
                          distribution is, independent of the null class.
            start_end     geometric mean of the independent start and end
                          token probabilities. A cruder signal that does not
                          depend on span enumeration at all.

        If ranking is absent under all three, the finding is about the model.
        If one of them ranks errors, the finding was about the estimator.
        """
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

        valid_t = torch.tensor(valid)
        best, best_span = float("-inf"), None
        span_scores: list[float] = []
        for i in valid:
            for j in valid:
                if j < i or j - i + 1 > max_answer_tokens:
                    continue
                score = (start[i] + end[j]).item()
                span_scores.append(score)
                if score > best:
                    best, best_span = score, (i, j)

        if best_span is None:
            return "", {"null_margin": 0.0, "span_prob": 0.0, "start_end": 0.0}

        import math

        # 1. Two-way softmax between answering and declining.
        m = max(best, null)
        p_answer = math.exp(best - m)
        p_null = math.exp(null - m)
        null_margin = p_answer / (p_answer + p_null)

        # 2. Softmax over all admissible spans: how peaked is the choice?
        mx = max(span_scores)
        denom = sum(math.exp(sc - mx) for sc in span_scores)
        span_prob = math.exp(best - mx) / denom if denom else 0.0

        # 3. Independent start and end token probabilities, geometric mean.
        i, j = best_span
        s_probs = torch.softmax(start[valid_t], dim=0)
        e_probs = torch.softmax(end[valid_t], dim=0)
        pos = {v: k for k, v in enumerate(valid)}
        start_end = float((s_probs[pos[i]] * e_probs[pos[j]]) ** 0.5)

        text = context[int(offsets[i][0]):int(offsets[j][1])]
        return text, {"null_margin": float(null_margin),
                      "span_prob": float(span_prob),
                      "start_end": start_end}


def run(model_name: str, limit_contracts: int, budget_s: float,
        out_path: Path, cache: Path, min_conf: float,
        max_chars: int = 4000, task: str = "cuad") -> dict:
    docs = (fetch_squad(cache, limit_paragraphs=limit_contracts)
            if task == "squad" else fetch_corpus(cache))
    print(f"corpus [{task}]: {len(docs)} documents", flush=True)

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
                    text, confs = qa(qa_item["question"], context)
                except Exception as exc:          # noqa: BLE001
                    print(f"  skipped {clause}: {type(exc).__name__}: {exc}",
                          flush=True)
                    continue
                conf = confs["null_margin"]
                records.append({
                    "task": task,
                    "model": model_name,
                    "contract": doc.get("title", f"doc-{contracts_done}"),
                    "clause": clause,
                    "confidence": conf,
                    "conf_null_margin": confs["null_margin"],
                    "conf_span_prob": confs["span_prob"],
                    "conf_start_end": confs["start_end"],
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
        "task": task,
        "model": model_name,
        "context_chars_used": max_chars,
        "contracts_evaluated": contracts_done,
        "questions": len(records),
        "truncated": truncated,
        "seconds": time.time() - started,
        "min_confidence_to_answer": min_conf,
    }


ESTIMATORS = ("conf_null_margin", "conf_span_prob", "conf_start_end")


def analyse_estimator(records: list[dict], field: str) -> dict:
    """Capture behaviour under one confidence estimator."""
    sys.path.insert(0, str(ROOT / "src"))
    from agentcost.calibration import Answer, capture_curve

    answers = [Answer(item_id=r["contract"], clause_type=r["clause"],
                      confidence=r.get(field, r["confidence"]),
                      correct=r["correct"])
               for r in records]
    curve = capture_curve(answers)

    ordered = sorted(answers, key=lambda a: a.confidence)
    n_err = sum(1 for a in ordered if not a.correct)
    caught = {}
    for frac in (0.05, 0.10, 0.25, 0.50):
        k = int(len(ordered) * frac)
        caught[f"caught_at_{int(frac*100)}pct"] = (
            sum(1 for a in ordered[:k] if not a.correct) / n_err
            if n_err else 1.0)

    distinct = len({round(a.confidence, 3) for a in answers})
    return {
        "estimator": field,
        "lift": curve.lift,
        "accuracy": curve.accuracy,
        "errors": n_err,
        "distinct_values": distinct,
        "saturation": 1 - distinct / max(1, len(answers)),
        **caught,
        "review_for_95pct_capture": curve.review_rate_for(0.05),
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
    by_estimator = [analyse_estimator(records, f) for f in ESTIMATORS
                    if any(f in r for r in records)]
    return {
        "by_estimator": by_estimator,
        "accuracy": curve.accuracy,
        "majority_class_baseline": majority,
        "lift_at_error_rate": curve.lift,
        "review_for_90pct_capture": curve.review_rate_for(0.10),
        "review_for_95pct_capture": curve.review_rate_for(0.05),
        "capture_curve": curve.as_dict()["curve"],
        "risk_controlled": policies,
    }


MODELS = [
    "akdeniz27/roberta-base-cuad",
    "Rakib/roberta-base-on-cuad",
    "marshmellow77/roberta-base-cuad",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--task", default="cuad", choices=["cuad", "squad"])
    ap.add_argument("--all-models", action="store_true",
                    help="run every known CUAD checkpoint, so the finding is "
                         "about the method rather than one snapshot")
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

    if args.all_models:
        models = SQUAD_MODELS if args.task == "squad" else MODELS
    else:
        models = [args.model]
    per_model, all_records = [], []

    for name in models:
        raw = Path(args.out).with_suffix(
            f".{args.task}.{name.split('/')[-1]}.json")
        try:
            meta = run(name, args.contracts, args.budget / len(models), raw,
                       Path(args.cache), args.min_conf, args.max_chars,
                       task=args.task)
        except Exception as exc:          # noqa: BLE001
            print(f"!! {name} unavailable: {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
            continue
        recs = json.loads(raw.read_text()) if raw.exists() else []
        if not recs:
            continue
        all_records.extend(recs)
        per_model.append({"method": meta, **analyse(recs)})

    if not per_model:
        print("no model produced records; refusing to write an empty result",
              file=sys.stderr)
        return 1

    result = {"models": per_model,
              "pooled": analyse(all_records) if len(per_model) > 1 else None}
    Path(args.out).write_text(json.dumps(result, indent=2))

    print()
    print("=" * 74)
    print("  CONFIDENCE ESTIMATOR COMPARISON  (lift > 1 means errors rank low)")
    print("=" * 74)
    print(f"  {'model':26s} {'estimator':18s} {'lift':>6s} {'sat':>6s} "
          f"{'err@25%':>8s}")
    for m in per_model:
        short = m["method"]["model"].split("/")[-1][:24]
        for e in m["by_estimator"]:
            print(f"  {short:26s} {e['estimator'][5:]:18s} {e['lift']:>6.2f} "
                  f"{e['saturation']:>6.1%} {e['caught_at_25pct']:>7.0%}")
    print("=" * 74)

    meta = per_model[0]["method"]
    result_first = per_model[0]

    print()
    print("=" * 66)
    print(f"  model              : {meta['model']}")
    print(f"  questions scored   : {meta['questions']:,} "
          f"over {meta['contracts_evaluated']} contracts"
          f"{' (truncated)' if meta['truncated'] else ''}")
    print(f"  context per doc    : {meta['context_chars_used']:,} chars "
          f"(truncated)")
    print(f"  accuracy           : {result_first['accuracy']:.1%}")
    print(f"  majority baseline  : {result_first['majority_class_baseline']:.1%}")
    print(f"  lift over random   : {result_first['lift_at_error_rate']:.2f}x")
    print(f"  review for 95% capture: "
          f"{result_first['review_for_95pct_capture']:.1%}")
    print()
    for name, pol in result_first["risk_controlled"].items():
        print(f"  {name}: review {pol['review_rate']:.1%}, "
              f"held-out risk {pol['realised_risk_on_heldout']:.3f}, "
              f"target respected: {pol['respected_target']}")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
