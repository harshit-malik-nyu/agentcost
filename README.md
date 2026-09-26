# What does an AI agent actually cost to run a process?

**Not what the vendor comparison measures.**

Built on CUAD — 510 commercial contracts annotated by practising lawyers
across 41 clause types. Real documents, real expert ground truth, real token
counts. Every figure regenerates from the code.

---

## The finding

### Inference is a rounding error

Running contract extraction over the corpus, batched, on a mid-tier model:

| | |
|---|---:|
| Inference | **$3.46** |
| Human review | **$918.00** |
| Inference share of total | **0.38%** |

The entire vendor conversation — price per million tokens, which model, which
provider — moves **under one percent** of the cost of the work. It holds across
every price tier: even a frontier model at four times the price leaves
inference at 1.4%.

**The review term dominates, and nothing else comes close.**

### So the review rate is the whole decision

| Review rate | Cost per contract | Saving vs human-only |
|---:|---:|---:|
| 100% | $9.02 | 73% |
| 50% | $4.51 | 87% |
| 15% | $1.38 | 96% |

Human-only extraction costs $33.75 per contract. Note the top row: **even
checking every single output still saves 73%**, because verifying a highlighted
answer is roughly four times faster than finding it from scratch. An agent does
not have to be trusted blindly to pay for itself.

### And the review rate is set by calibration, not accuracy

This is the part that inverts the usual pitch.

| Agent | Review needed | Cost/contract | Saving |
|---|---:|---:|---:|
| 95% accurate, **cannot rank its errors** | 95% | $8.59 | 75% |
| **85% accurate, ranks errors well** | **15%** | **$1.36** | **96%** |
| 95% accurate and ranks errors well | 6% | $0.56 | 98% |

*(These three rows are constructed, to isolate the mechanism. The measured row
— a real model, with real confidence scores — is below, and it lands in the
first category.)*

**A worse model that knows when it is wrong is six times cheaper to run than a
better model that does not.**

The mechanism: if confidence does not predict error, you have to check
everything, and checking everything is where the money goes. If confidence
ranks errors to the top of the queue, you check a fraction and catch most of
them. The measured quantity is **lift** — how much better than random the
ranking is:

| Agent | Lift over random |
|---|---:|
| 95% accurate, blind | 1.4x |
| 85% accurate, calibrated | 6.9x |
| 95% accurate, calibrated | 19.0x |

Lift is standard in credit and fraud scoring. It is essentially absent from how
document-extraction agents are sold, which is on accuracy alone.

### Measured: the confidence cannot rank errors, and it is not the estimator's fault

Everything above about calibration was a mechanism. Here is a measurement.

Two independently fine-tuned CUAD checkpoints, 410 questions each from real
contracts, **three different confidence estimators computed per answer**:

| Estimator | What it measures | Distinct values | Lift |
|---|---|---:|---:|
| `null_margin` | softmax between best span and the null span — the practitioner default | 15 | **0.00** |
| `span_prob` | softmax over all admissible spans | **187** | **0.00** |
| `start_end` | geometric mean of independent start and end token probabilities | **222** | **0.00** |

**Zero lift in all six model-estimator combinations.**

The three estimators were computed precisely to separate two questions that the
first measurement conflated: is the *model* uncalibrated, or is the *estimator*?
Null-margin softmax saturates by construction on a model trained with a null
class, so a finding based on it alone is partly a claim about my own choice of
signal.

The answer is unambiguous. `span_prob` and `start_end` are **not degenerate** —
187 and 222 distinct values across 410 answers — and they rank errors no better
than the saturated one. Reviewing the least-confident quarter catches **0–4%**
of errors under every signal.

| Model | Questions | Accuracy | Errors | Lift (all three estimators) |
|---|---:|---:|---:|---:|
| `akdeniz27/roberta-base-cuad` | 410 | 96.6% | 14 | 0.00 |
| `Rakib/roberta-base-on-cuad` | 410 | 94.1% | 24 | 0.00 |

And the errors are the kind that matter — high-confidence fabrications:

| Confidence | Clause present? | Predicted |
|---:|---|---|
| 1.0000 | no | `6th day of April, 1999` |
| 1.0000 | yes | `OCTOBER 15, 2009` |
| 0.9999 | no | `, 2013,` |

**What this means for the economics.** These agents are the "blind" row of the
table above. They cannot triage their own work, so the review rate to hit any
error tolerance is near 100%, and the saving collapses from the 96% a
calibrated agent would deliver to the 73% available from verification speed
alone.

### Why the signal is absent — and it is recoverable

A negative result without a mechanism is weak. Two further experiments, on the
same data.

**Where the errors sit.** They are not uniform. The model is **eight times more
likely to be wrong when a clause is actually present**:

| | Error rate | n |
|---|---:|---:|
| Clause present | **21.8%** | 78 |
| Clause absent | 2.8% | 742 |

And five of forty-one clause types hold **66%** of all errors — led by
Effective Date (45% wrong), Parties (30%) and Expiration Date (20%).

**The mechanism.** Confidence encodes *whether something is there*, which the
model judges well, not *whether the extracted span is right*, which it judges
badly. Because 90% of questions have nothing to find, accuracy is carried
almost entirely by correct declining, and the errors concentrate in the
minority of cases where the model has found something and is therefore
uniformly confident.

That hypothesis predicts something testable: ranking should reappear inside the
present-clause subset, where the presence signal no longer dominates. It does.

| Signal | Lift, pooled | Lift within present-clause subset |
|---|---:|---:|
| `null_margin` | 0.00 | **3.51** |
| `span_prob` | 0.00 | **1.89** |
| `start_end` | 0.00 | **2.43** |

**The fix.** If the information is present but badly expressed, a trained
combination should recover it. A logistic regression over the same signals,
five-fold cross-validated, with clause base rates computed **per training fold**
so no held-out label leaks through a feature:

| | Lift |
|---|---:|
| Raw `null_margin` | **0.00** |
| Trained on the three confidence signals alone | **4.56** |
| Trained with cheap deployment features | **7.21** |

**The information was there the whole time.** Raw confidence cannot rank
errors; a linear combination of exactly the same numbers ranks them well. This
is an engineering problem, not a capability ceiling — and it moves the agent
from the "blind" row of the cost table to the "calibrated" row, from a 75%
saving to something near 96%.

**What a buyer should take from this.** Do not ask a vendor for an accuracy
figure, and do not trust a raw confidence score for triage. Ask for the capture
curve, and if it is flat, ask whether they have tried fitting one — because on
the evidence here the signal exists and is simply not being used.

### Tested on a second task: the prediction was partly wrong

The mechanism above makes a falsifiable claim. If confidence encodes *presence*
and pooled ranking collapses because presence dominates, then a balanced task
should behave differently. CUAD is 90% unanswerable; **SQuAD v2 is 50%**.

Three architectures were run on it — RoBERTa, BERT and ELECTRA — through the
identical measurement code, 7,905 answers and 2,019 errors.

| | CUAD | SQuAD v2 |
|---|---:|---:|
| Unanswerable share | 90% | 50% |
| Error rate, answer present | **21.8%** | 17.8% |
| Error rate, answer absent | **2.8%** | **33.6%** |
| Pooled raw lift | 0.00 | 0.25 |
| Lift within present-answer subset | 3.51 / 1.89 / 2.43 | 3.14 / 2.60 / 2.81 |
| Trained lift, per architecture | 4.10, 5.56 | **1.09, 1.20, 1.24** |

**What held.** Raw confidence is worse than random for triage on both tasks —
every pooled lift is below 1.0, meaning sorting by confidence is *worse* than
sorting at random. And ranking reappears inside the present-answer subset on
both, at roughly 3x, exactly as predicted.

**What did not.** The error asymmetry **reverses**. On CUAD the model is eight
times more likely to be wrong when an answer is present; on SQuAD it is twice
as likely to be wrong when the answer is **absent** — hallucinating answers to
unanswerable questions rather than mis-extracting present ones.

So the specific story — *errors concentrate where the model has found
something* — is a property of CUAD, not a general one. The defensible
generalisation is weaker and stated as such: **confidence tracks something
systematic that is orthogonal to correctness**, and which something it is
depends on the task.

**And the fix is much weaker off CUAD.** Trained lift falls from 4.10–5.56 to
1.09–1.24. A calibrator recovers most of the signal on contract extraction and
barely any on SQuAD. The pooled SQuAD figure of 2.68 is higher than any
individual model's, which is an artefact worth naming: pooling three models of
differing accuracy lets the calibrator discriminate between models rather than
between correct and incorrect answers. The per-architecture numbers are the
honest ones.

**Why this is reported rather than buried.** The prediction was made before the
run and it came back half wrong. A confirmed prediction would have been a
cleaner story and a weaker result — this one bounds the claim, which is what
the second task was for.

#### Scope, stated narrowly

A third checkpoint, `marshmellow77/roberta-base-cuad`, returned byte-identical
results to `akdeniz27` — same accuracy, same error count, same distinct-value
counts. It is the same weights mirrored, so this is **two distinct models, not
three**, and the run says so rather than counting it as independent evidence.

Other limits: 38 errors across 820 questions is a small sample — roughly seven
or eight errors per cross-validation fold, so the trained-lift estimate carries
a wide interval even though its direction is consistent across folds. The lift
estimate generally carries a wide interval — though zero of fourteen and one of
twenty-four in the bottom quartile is not an ambiguous direction. Contexts were
truncated to 4,000 characters, which raises the majority-class baseline to
about 90% and makes the task easier than a full contract. Both models share an
architecture and a fine-tuning recipe, so this is evidence about extractive QA
fine-tuning rather than about language models generally. A trained calibrator
or an ensemble was not tried and might recover ranking.

What it does establish is that **useful calibration cannot be assumed.** Every
real model tested has none, under every signal tried — and an automation
business case built on selective review would have been wrong about its central
input.

### And a threshold read off that curve is not safe to deploy

The capture curve above is a lift chart. It has been standard in credit scoring
since the 1990s, and it has a defect that matters: **the review threshold it
selects carries no guarantee on new data.**

Measured over 300 split-sample deployments at a 5% error target:

| Agent | Empirical threshold violates target | Risk-controlled violates | Review rate |
|---|---:|---:|---|
| Calibrated | **47.3%** | 6.7% | 5% → 7% |
| Weakly calibrated | **49.0%** | 5.7% | 37% → **60%** |
| Strongly calibrated | **27.0%** | 7.0% | 1% → 2% |

A threshold tuned to hit 5% on a validation sample **exceeds it in roughly half
of deployments**, because it is fitted to the mean of a quantity that varies.

`conformal.py` replaces it with a distribution-free construction: a
Clopper-Pearson upper bound on the selective risk at each of a pre-specified
grid of thresholds, taking the most permissive one whose bound clears the
target. The guarantee — selective risk ≤ α with probability ≥ 1−δ — holds
regardless of how good the underlying agent is.

**It is not free.** For the weakly calibrated agent, honest control requires
60% review against the 37% the empirical curve suggests. That gap is the cost
of a guarantee, and quoting the empirical number without it understates what
running the process safely takes.

#### On method, and what is not novel here

None of the statistical machinery is new. Selective prediction was formalised
by Geifman and El-Yaniv (2017); distribution-free risk control by Bates et al.
(2021), extended to conformal risk control by Angelopoulos et al. (ICLR 2024)
and Learn-then-Test (*Annals of Applied Statistics*, 2025). Applications to
language models include conformal abstention for hallucination (Abbasi-Yadkori
et al. 2024), prediction sets over multiple-choice answers (Kumar et al. 2023),
open-ended generation (Quach et al. 2023) and selective conformal judging
(SCOPE, 2026). Campos et al. (TACL 2024) survey the field.

**What this repository contributes is the join**: applying a standard
risk-control construction to the *human-review budget* rather than to
prediction-set size, so the output is a staffing decision with a guarantee
attached rather than an accuracy figure.

One defect found while building it, recorded because it is subtle: the
candidate threshold grid was originally derived from observed confidence
values. That makes the selection order data-dependent and breaks
fixed-sequence testing — violations ran at 10% against a 5% target, small and
consistent. Fixing the grid to a pre-specified range restored nominal coverage.

---

## Two things that make accuracy claims hard to read

**A blanket "not present" scores 70.3%.** Of 4,182 expert-labelled questions in
the test split, 70.3% have no answer in the contract. A system that extracts
nothing is right seven times out of ten. Any accuracy figure has to be read
against that floor, and most are quoted without it.

**Clause prevalence runs from 0% to 100%.** "Price Restrictions" appears in
none of the 102 test contracts; "Parties" appears in all of them. An aggregate
accuracy number across 41 categories spanning that range is close to
meaningless.

---

## Architecture: a 30× difference for identical output

| Approach | Inference cost |
|---|---:|
| Resend the contract for each of 41 questions | $41.69 |
| Send once, extract all 41 | **$3.46** |

Same work, same answers. The first is the obvious implementation.

It is worth knowing and it is not the headline — **even the wasteful version
leaves inference under 10% of total cost.** Engineering the token bill optimises
the wrong term.

---

## Using it

```bash
agentcost breakdown --review 0.15      # where the money goes
agentcost policy --accuracy 0.95 --calibration 0.0   # what review rate is required
agentcost audit                        # every parameter and its source
```

## What is measured, estimated, and assumed

Every parameter carries an evidence class, because the difference between a
published list price and my estimate of paralegal review time is the difference
between a number you can check and one you have to trust.

The load-bearing estimates, stated plainly:

- **45 minutes** for a human to extract 41 clause types from scratch
- **12 minutes** to verify answers the agent has already located and highlighted
- **$45/hour** fully loaded review cost

The ratio between the first two — about 4:1 — is the most consequential number
in the model after the review rate itself. All three are ESTIMATE and none has
been validated against a legal operations team. A reader who disagrees can
substitute their own and re-derive; the conclusion that inference is negligible
survives any plausible values, because it is three orders of magnitude away.

## Reproducing

```bash
pip install -e ".[dev]"
pytest -q
python scripts/run_study.py     # fetches CUAD and recomputes everything
```

No dependencies. Corpus statistics are vendored in `data/`; the 40MB source
archive is fetched by the study script.

## The case against this analysis

[`docs/against.md`](docs/against.md) — the strongest argument that this should
not drive a decision.

## Sources

CUAD v1, The Atticus Project, CC BY 4.0 —
[github.com/The-Atticus-Project/cuad](https://github.com/The-Atticus-Project/cuad)

## License

MIT.
