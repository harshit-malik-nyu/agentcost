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

### Measured on a real model: the confidence is worse than useless

Everything above about calibration was a mechanism. Here is a measurement.

**RoBERTa fine-tuned on CUAD** (`akdeniz27/roberta-base-cuad`), run over 492
questions from 12 real contracts. Confidence is the softmax margin between the
best answer span and the null span — the natural signal for this architecture,
and the one a practitioner would use.

| | |
|---|---:|
| Accuracy | 96.7% |
| Majority-class baseline | 90.2% |
| Errors | 16 of 492 |
| **Lift over random ranking** | **0.00×** |

The lift is zero, and the reason is worse than a null result:

| Review the least-confident… | Errors caught |
|---|---:|
| 5% | **0 of 16** |
| 10% | **0 of 16** |
| 25% | **0 of 16** |
| 50% | 1 of 16 |

**The model's errors are concentrated at the top of its confidence ranking, not
the bottom.** Sorting by confidence to decide what to review is not merely
uninformative here — it is anti-correlated with error.

The confidence distribution shows why. It is degenerate: 444 of 492 answers sit
at essentially 0.0 and 43 at essentially 1.0, with five values in between. The
model is almost never uncertain. And its mistakes look like this:

| Confidence | Clause present? | Predicted |
|---:|---|---|
| 1.0000 | no | `6th day of April, 1999` |
| 1.0000 | yes | `OCTOBER 15, 2009` |
| 1.0000 | yes | `NFLA-NC` |
| 0.9999 | no | `, 2013,` |

Fabricating a date with perfect confidence, and emitting `, 2013,` as a clause
with confidence 0.9999.

**What this means for the economics.** This agent is the "blind" row of the
table above. It cannot triage its own work, so the review rate that hits any
error tolerance is close to 100%, and the saving collapses from the 96% a
calibrated agent would deliver to the 73% available from verification speed
alone.

**What it does not establish.** One model, 492 questions, 16 errors — a small
sample. Contexts were truncated to 4,000 characters, which raises the
majority-class baseline to 90.2% and makes the task easier than the full
document. A different confidence estimator — span probability rather than the
null margin, or an ensemble — might rank errors better, and this measures the
natural signal rather than the best possible one.

What it does establish is that **useful calibration cannot be assumed.** The
first real model tested has none, and an automation business case built on
selective review would have been wrong about its central input.

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
