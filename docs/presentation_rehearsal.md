# Presentation rehearsal notes

The generated deck is an evidence-based project update, not a claim that the
planned model comparison is complete. Keep the pending-experiment caveat visible
if you rearrange slides or export selected slides.

## Five-minute talk track

1. **Opening (20 seconds):** Frame the task as finding and classifying legal
   clause spans in full contracts. CUAD tags legal functions; the system does
   not give legal advice.
2. **Dataset (35 seconds):** Explain the 510-contract, 41-category data and
   why both short fields and long provisions make this different from ordinary
   entity extraction.
3. **Integrity (45 seconds):** Describe source-offset validation, exact
   agreement for all 13,823 answers, reversible offset mapping, and
   contract-level splits.
4. **Challenges (40 seconds):** Highlight the 88.89% outside-token share,
   cross-category overlaps, and three categories with fewer than five test
   answers.
5. **Method (40 seconds):** Walk from original CUAD answers through normalized
   tokens, multi-label/BIO targets, grouped splits, windows, and stitched
   predictions.
6. **Evidence (50 seconds):** State the baseline results accurately. The
   lexical lookup has some partial overlap but no exact spans. The CRF number
   comes from a 10-contract smoke run and is not a full-data result.
7. **Evaluation (35 seconds):** Contrast exact and partial span scoring, then
   explain contract bootstrap intervals and paired comparisons.
8. **Close (35 seconds):** Summarize completed reproducibility work and name
   neural runs, multiple seeds, and error review as the next steps.

## Rehearsal checklist

- Replace generic presenter details with the team's names and contributions.
- Do not describe the 10-contract CRF run as a trained full-data baseline.
- Do not say that the model comparison is complete until neural runs and
  repeated seeds are recorded.
- Explain the single-label overlap priority if asked; the multi-label targets
  retain all category assignments.
- Allow time for questions about legal-domain generalization and rare labels.
