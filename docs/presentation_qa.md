# Presentation Q&A

Use the answers as prompts, not as claims beyond the measured outputs in
`results/metrics.csv`, `results/experiment_summary.json`, and
`results/audit/audit_summary.json`.

## Why CUAD?

It is a public, lawyer-annotated legal-contract dataset with 510 contracts,
41 clause categories, and over 13,000 answer spans. It supports a meaningful
long-document classification task without presenting the model as legal advice.

## Is this just named-entity recognition?

No. The labels capture a clause's legal function and often cover full
sentences or paragraphs. Their spans can be much longer than names or dates.

## How were answer offsets checked?

Each answer was matched against its contract context at the supplied
character offsets. All 13,823 source answers in this release matched exactly.
Normalization keeps a mapping to source positions; repairs and drops are
logged rather than hidden.

## How do overlapping categories work?

The multi-label representation retains each category for a token. The
single-label BIO experiments resolve collisions with the documented
rare-category priority rule. This is a modeling choice, not a claim that the
annotations are exclusive.

## How did you prevent train/test leakage?

Contracts—not token windows—are split into train, validation, and test. The
duplicate audit reports the exact-match pair and checks approximate
near-duplicates. It found no detected cross-split duplicate at the configured
SimHash threshold; this approximate screen cannot prove that no boilerplate
overlaps.

## Why not report accuracy?

About 88.89% of priority-resolved tokens are outside (`O`). Accuracy could be
high for a model that misses the clause classes, so the main token metric is
macro F1 over the 41 positive categories, alongside exact and partial span F1.

## What is the strongest result?

Use the most recently generated `results/findings_summary.md`. If fewer than
three independent seeds are recorded for a model, describe its result as
exploratory and do not claim a robust ranking.

## Why can a partial-span score be positive when strict span F1 is zero?

Partial scoring accepts a one-to-one same-category token overlap. Strict
scoring requires the exact start and end boundaries. A model can identify part
of a clause while still failing exact extraction.

## What are the main limitations?

There are only 510 contracts, three categories have fewer than five gold test
spans, and category annotations overlap. SimHash is approximate. Results can
also vary with the random seed and model checkpoint.

## Is this tool suitable for legal decisions?

No. It is an NLP course experiment on CUAD and has not been validated for
production, legal review, or decision-making.

## What is next?

Finish and compare the documented model/ablation runs, repeat the main models
with at least three seeds, review their error samples, and publish the
reproducible artifacts with real author contributions.
