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
