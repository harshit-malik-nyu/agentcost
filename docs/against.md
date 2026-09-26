# The case against this analysis

The strongest argument I can make that this should not drive a decision.

---

## 1. The review-time estimates are mine, and they carry the whole result

45 minutes to extract, 12 minutes to verify, $45 an hour. The 4:1 ratio between
extraction and verification is the second most consequential number in the
model, and **none of the three has been validated against a legal operations
team.**

If verification is actually half as fast as I assume, savings at a 15% review
rate fall materially. If contract review is offshored at a third of the rate,
the human baseline drops and the case narrows.

**This is the most serious objection.** The finding that inference is
negligible survives any plausible values — it is three orders of magnitude away
— but the specific savings figures do not.

## 2. Calibration is simulated, not measured

The headline comparison — a calibrated 85% agent beating a blind 95% one —
comes from constructed answers with known calibration, not from a real agent's
confidence scores.

The simulation characterises the instrument: it shows the capture curve would
detect the effect if it were there. It does not show that any real agent has
useful calibration, and there is reason for doubt — self-reported confidence
from language models is known to be poorly calibrated in many settings.

**The honest scope: this establishes that calibration determines cost, not that
any particular agent has it.** Measuring a real agent's curve needs API access
and an evaluation run.

## 3. Verification may not be independent of the agent

The model assumes a reviewer checking a highlighted span catches the agent's
errors. Anchoring says otherwise: a reviewer shown a confident, plausible
answer is measurably less likely to notice it is wrong than one working from
scratch.

If verification is 4x faster *and* meaningfully less reliable, the effective
error rate after review is higher than modelled, and the comparison against
human-only extraction flatters the agent.

Nothing here measures that, and it is a known effect rather than a speculative
one.

## 4. CUAD is cleaner than the work

The corpus is 510 published commercial contracts — mostly filed with the SEC,
professionally drafted, in English, machine-readable. Real back-office document
work includes scans, handwriting, non-standard forms, and documents in four
languages.

Extraction accuracy on CUAD is an upper bound on a real queue, and the review
rate implied by it is a lower bound.

## 5. "Not present" dominates, which makes the task easier than it looks

70.3% of questions have no answer. A substantial share of the agent's work is
correctly declining to extract. That is genuinely useful, and it is an easier
problem than locating a clause — so aggregate performance overstates capability
on the part that matters.

The project flags this for accuracy claims and then uses the same corpus for
the cost model without adjusting for it.

## 6. The process may not be the bottleneck

The model prices contract extraction. In a real deal, extraction is rarely what
holds up a timeline — negotiation, approvals and counterparty response are. A
96% saving on a step that is not on the critical path is worth less than the
percentage suggests.

## 7. One process, generalised

Everything here is one document-extraction task. The claim that inference is
negligible should hold anywhere human review is the alternative, but the
specific ratios will not transfer to processes with different review
economics — high-volume, low-value classification where review is seconds
rather than minutes, for instance.

---

## What survives

- **Inference is a rounding error.** Computed from real token counts across
  every published price tier, three orders of magnitude from the review cost.
  No estimate of mine moves it.
- **The architecture finding** — 30x for identical output — is arithmetic on
  measured document lengths.
- **The 70.3% majority-class baseline** is a fact about the corpus, and it
  makes most accuracy claims unreadable without it.
- **The structural argument** that review rate determines cost, and calibration
  determines review rate, follows from the cost model regardless of what any
  particular agent's calibration turns out to be.

## What does not

Every specific savings percentage. They rest on three unvalidated time
estimates and a simulated calibration curve, and they should not be quoted.

## The objection I cannot answer

Argument 2. Until a real agent's capture curve is measured, the headline
comparison is a demonstration of a mechanism rather than a finding about any
system anyone can buy.

That needs an evaluation run against the corpus with a real model, which needs
API access this project does not have.
