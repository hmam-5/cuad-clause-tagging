# CUAD Clause Tagging: Complete Project and Results Report

**Prepared:** 31 October 2026

**Project repository:** <https://github.com/hmam-5/cuad-clause-tagging>

**Primary dataset:** Contract Understanding Atticus Dataset (CUAD) v1

**Purpose:** presentation briefing and technical reference

> **Important status statement:** The data pipeline, audit trail, baseline
> runs, reproducibility scripts, tests, and draft presentation are implemented.
> The planned model study is **not complete**: the CRF and BERT runs recorded
> here are smoke tests, only one seed was run for each full-data trivial
> baseline, and the planned full-data BiLSTM-CRF, transformer comparisons, and
> multi-seed experiments have not been completed. Do not present smoke-test
> scores as final model comparisons or describe the experimental study as
> finished.

## 1. Executive summary

This project builds a reproducible natural-language-processing (NLP) pipeline
for identifying and classifying legal clause spans in commercial contracts.
It uses CUAD v1, a dataset of 510 contracts annotated for 41 legal clause
categories. The prediction task is token classification: tokens are assigned
BIO labels (`B-category`, `I-category`) or `O` (“outside”). There are 83 BIO
labels in total.

The implementation covers source-data acquisition and audits, offset-preserving
text cleaning, span validation, multi-label and BIO targets, contract-level
train/validation/test splits, long-document windows, dataset analysis, scoring,
several baseline/model training entry points, error and duplicate audits,
report generation, and unit tests.

The most useful verified result so far is a deliberately simple
most-frequent-word baseline trained on all 408 training contracts and tested on
the 51-contract test split. Its token macro F1 is **0.008415** (contract
bootstrap 95% interval **0.006734–0.009645**); strict span micro F1 is **0**,
while partial span micro and macro F1 are **0.169492** and **0.110983**.
These results show that exact span extraction is not solved by this lexical
baseline. They do not establish whether a trained sequence model can solve
the task.

## 2. Research task and scope

### Research question

Can a model locate passages in full contracts and assign each token to one of
41 lawyer-defined legal clause categories, with reliable boundaries and
generalization to contracts not seen during training?

### Task definition

- **Input:** a contract’s text.
- **Output:** token-level BIO tags for 41 categories.
- **BIO convention:** `B-X` begins a span of category `X`; `I-X` continues
  the span; `O` is not assigned to a clause.
- **Label count:** 83, counting `O`, `B-X`, and `I-X` for each category.
- **Overlapping labels:** source answers can overlap or assign multiple
  categories to the same words. The pipeline retains sparse multi-label
  targets. The single-label BIO baselines must resolve overlap; they assign
  priority to the globally rarer category, with category order as a
  tie-breaker. This is a modeling convention, not a change to the source
  annotations.
- **Intended use:** research evaluation and locating candidate passages for
  human review. This is not a legal decision system and does not provide legal
  advice.

## 3. Dataset, provenance, and licensing

The project uses the official CUAD v1 release from the
[Atticus Project CUAD repository](https://github.com/The-Atticus-Project/cuad).
The retained `data/raw/CUADv1.json` is the release’s original JSON: contract
contexts and question/answer annotations, including character offsets, are
embedded in the file. The separate Zenodo archive is 105,883,672 bytes and was
not included because it exceeds the project’s 50 MiB per-file limit.

The source dataset is licensed **CC BY 4.0**; the project code is licensed
**MIT**. These are separate licenses. Preserve attribution to The Atticus
Project and the CUAD paper authors when redistributing the dataset or its
derived content. The source and acquisition notes are in `data/raw/README.md`.

The raw JSON is 40,128,638 bytes. The processed compressed corpus is
approximately 24,354,900 bytes. Both are below 50 MiB individually.

## 4. Project workflow, step by step

### Step 1 — Acquire and inspect the source

`scripts/download_data.py` obtains the official data archive and extracts the
source JSON. The source audit inspects question records and answer entries
before modeling. Its measured counts are:

| Source audit measure | Count |
|---|---:|
| Contracts | 510 |
| Questions | 20,910 |
| Questions with no answer spans | 14,208 |
| Valid non-empty answer spans | 13,823 |
| Empty answer entries | 0 |
| Missing answer lists | 0 |
| Non-empty answers missing offsets | 0 |

Questions without spans are expected negative/empty annotations, not invalid
source records.

### Step 2 — Normalize text while preserving source offsets

`src/cuad_tagging/pipeline.py` normalizes whitespace and removes invisible
Unicode format characters while maintaining a mapping between normalized
characters and positions in the original contract text. This lets processed
tokens and predicted spans be related back to source text.

For each annotation, the pipeline validates the exact answer substring and
its character offsets. A nearby offset is corrected only if the answer text
has one unique match within 32 characters. Corrections and dropped spans are
written to `data/processed/span_corrections.csv`.

Measured outcomes:

- 13,823 non-empty source spans.
- 13,823 kept at their exact source offsets.
- 0 corrected and 0 dropped.
- 184 invisible Unicode format characters removed.
- 100% token-to-span round-trip agreement.

### Step 3 — Tokenize and create targets

The pipeline tokenizes the full contract and records cleaned and original
offsets. It stores both:

1. sparse multi-label category assignments, preserving annotation overlap; and
2. a single BIO label per word for the sequence-labeling baselines.

Where categories overlap in a BIO target, the less frequent category takes
priority; category order breaks ties. This makes the single-label conversion
explicit and reproducible while retaining the original multi-label data in the
processed record.

### Step 4 — Split by contract, then create long-document windows

Contracts—not individual tokens or windows—are split using a seeded
multilabel stratification process. No windows from a contract are distributed
between splits.

| Split | Contracts |
|---|---:|
| Train | 408 |
| Validation | 51 |
| Test | 51 |

For word-based processing the configured windows are 512 words with stride
128, yielding 13,274 windows. The transformer path additionally enforces the
tokenizer’s subword budget, adjusts boundaries around spans where feasible,
and merges overlapping window predictions. A single word exceeding the
subword budget is an explicit error rather than silently truncating the
sequence.

### Step 5 — Produce dataset analyses

`scripts/analyze_data.py` writes summary tables, category/span distributions,
BIO-label distributions, overlap analyses, and plots under
`results/analysis/`.

| Preprocessing measure | Result |
|---|---:|
| Word tokens | 5,114,638 |
| Mean answer length | 262.02 characters / 50.02 word tokens |
| 95th-percentile answer length | 758 characters / 147 word tokens |
| `O` share after BIO priority resolution | 88.8905% |
| Tokens carrying multiple categories | 90,720 (1.7737%) |
| Cross-category overlapping annotation pairs | 2,122 |
| Word windows | 13,274 |
| Span round-trip agreement | 100% |

Three categories have fewer than five gold spans in the test split:
`Most Favored Nation` (4), `Price Restrictions` (4), and
`Affiliate License-Licensor` (4). Scores for these categories are
particularly uncertain.

### Step 6 — Train baseline and model implementations

The repository includes these training approaches:

1. **All-`O` diagnostic:** predicts that every token is outside a clause.
2. **Most-frequent-word lookup:** learns a BIO label for each lowercased word
   from the training set.
3. **Feature CRF:** linear-chain conditional random field with word, casing,
   affix, shape, and neighboring-word features.
4. **BiLSTM-CRF:** bidirectional LSTM with a CRF output layer and random
   embeddings by default; retains labeled windows and subsamples O-only
   windows to control training volume.
5. **Hugging Face token classifiers:** a shared training pipeline supporting
   pretrained token-classification checkpoints. Planned comparisons include
   BERT, LegalBERT, RoBERTa, and DeBERTa.

The transformer path aligns word labels to subtokens, computes loss on first
subtokens, uses class weights, keeps windows inside model token limits, and
stitches overlapping predictions. Checkpoints and smoke-run files under
`artifacts/` are local/ignored and are not part of the public repository.

### Step 7 — Evaluate predictions

The metrics implementation reports:

- **Token macro F1:** macro-average across the 41 positive clause categories;
  `O` is excluded from the macro average.
- **Strict span micro/macro F1:** a prediction is correct only when both
  category and token boundaries match exactly.
- **Partial-overlap span micro/macro F1:** same-category spans count as
  matches when they overlap, using one-to-one matching.
- **Contract-bootstrap confidence intervals:** contracts, not individual
  tokens, are resampled.
- **Paired contract bootstrap:** estimates the difference between two models
  on the same held-out contracts.

The split is fixed by the configured seed (42). Training scripts use the
validation set for checkpoint selection; the test set is for reporting.

### Step 8 — Audit errors and possible duplicate leakage

`scripts/audit_predictions.py` classifies unmatched and overlapping spans into
missed-span, hallucinated-span, boundary-error, and wrong-category errors. It
also checks exact normalized-text duplicates and uses 64-bit SimHash over
five-word shingles to screen for near duplicates. The approximate audit uses
up to 512 uniformly spaced shingles per contract and can miss partial
boilerplate overlap; it is not proof that no leakage exists.

### Step 9 — Summarize and create presentation materials

`scripts/summarize_experiments.py` creates the machine-readable experiment
summary, per-category table/plot, and `results/findings_summary.md`. Smoke runs
are excluded from full-data model aggregates. The 10-slide
`results/CUAD_project_presentation.pptx`, speaker notes, and Q&A guidance are
included in the repository. The deck is a **draft presentation**, not evidence
that a spoken rehearsal was completed.

### Step 10 — Test and reproduce

The test suite covers offset mapping, span correction/drop behavior, labels,
splits, windows, metrics, source audits, duplicate checks, and CSV export.
The final local verification recorded 14 passing unit tests; Ruff lint,
Python compilation, and PowerShell parsing also passed.

On Windows, run from the project root:

```powershell
python -m pip install -r requirements.txt
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_all.ps1
```

The workflow downloads/uses the source data, preprocesses it, creates data
analysis outputs, runs the O-only and word-lookup baselines, runs a **10-train
contract CRF smoke test**, compares the simple baselines, audits the
word-lookup predictions, summarizes the recorded experiments, rebuilds the
presentation, and runs tests. It does **not** run full neural experiments or
the full-data CRF. A full CRF fit can be requested with:

```powershell
python scripts/train_baselines.py --model crf
```

Transformer and BiLSTM-CRF entry points and multi-seed examples are described
in `docs/experiments.md`. A planned comparison requires at least three seeds
per main model; the runs reported below do not meet that requirement.

## 5. Results

### 5.1 Results table

All reported baseline scores below use the same 51-contract test split.
“Training contracts” and “evaluation scope” distinguish the full-data
baselines from the CRF smoke test. The smoke-test result is shown for
transparency, but is not a fair full-data model comparison.

| Run | Model | Train contracts | Test contracts | Scope | Token macro F1 (contract bootstrap 95% CI) | Strict span micro F1 | Strict span macro F1 | Partial span micro F1 | Partial span macro F1 |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| `o-only-seed42` | All-`O` diagnostic | 408* | 51 | Full split diagnostic | 0.000000 (0–0) | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| `most-frequent-word-seed42` | Word lookup | 408 | 51 | Full train and test | 0.008415 (0.006734–0.009645) | 0.000000 | 0.000000 | 0.169492 | 0.110983 |
| `crf-seed42` | Linear-chain CRF | 10 | 51 | **Smoke training**, full test | 0.000000 (0–0) | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

`*` The all-`O` predictor does not learn from the training contracts; the run
metadata records the full training split size as context.

The all-`O` and word-lookup results are each from **one seed**. The aggregate
report therefore has one run per baseline, not a multi-seed mean with
independently estimated seed variance. The CRF was trained on only 10 of 408
training contracts, although it was evaluated on all 51 test contracts.
Its score must not be interpreted as the performance of a full-data CRF.

### 5.2 Training and inference time

Times are hardware- and system-dependent and are reported for these individual
runs only.

| Run | Training time | Inference time per test contract |
|---|---:|---:|
| O-only | effectively zero | 0.016 ms |
| Most-frequent-word | 1.412 s | 1.060 ms |
| CRF smoke (10 training contracts) | 165.422 s | 345.715 ms |

### 5.3 Paired baseline comparison

The word-lookup baseline was compared against the all-`O` diagnostic on the
same 51 test contracts using 1,000 paired contract-bootstrap resamples
(seed 42):

| Measure (word lookup minus O-only) | Estimate |
|---|---:|
| Mean token macro-F1 difference | 0.008205 |
| 95% paired-bootstrap interval | 0.006748–0.009741 |
| Two-sided bootstrap p-value (finite-sample corrected) | 0.001998 |

This is a descriptive comparison of two single-seed baselines on one split.
It is **not** evidence that the lexical model is better than a trained CRF or
transformer, and it does not replace repeated-seed evaluation. The paired
bootstrap uses held-out contracts as the resampling unit.

### 5.4 Word-lookup per-category findings

The word-lookup run’s category-level token F1 is in
`results/per_category_f1.csv`. Highest observed categories include:

| Category | Token F1 | Gold test spans |
|---|---:|---:|
| Governing Law | 0.061 | 47 |
| Insurance | 0.043 | 44 |
| Anti-Assignment | 0.040 | 68 |
| Cap On Liability | 0.034 | 73 |
| Irrevocable Or Perpetual License | 0.026 | 21 |

Several categories have zero token F1 for this baseline. These category scores
are from one run and small test set; they should be treated as descriptive.

### 5.5 Error analysis and duplicate screening

For the word-lookup test predictions, the audit counted **1,981** span errors:

| Error type | Count | Share of classified errors |
|---|---:|---:|
| Missed span | 910 | 45.9% |
| Hallucinated span | 810 | 40.9% |
| Boundary error | 184 | 9.3% |
| Wrong category | 77 | 3.9% |

A reproducible random sample of 30 errors contained 14 missed spans, 12
hallucinated spans, 3 boundary errors, and 1 wrong-category error.

The duplicate screen found **one exact duplicate contract pair** among 510
contracts. Both records are in the same split. At a SimHash Hamming-distance
threshold of 4, it found no cross-split near-duplicate pairs. Since the
near-duplicate method is approximate and sparse-sampled, it cannot rule out
shared boilerplate or partial duplicates across splits.

### 5.6 BERT pipeline smoke test

A separate local BERT smoke run verified that a one-contract GPU train,
validation, prediction, and result-export path could complete after fixing a
CSV schema bug. It used `bert-base-cased`, seed 42, one epoch, batch size 1,
maximum sequence length 256, word windows of 64 with stride 16, one capped
training contract, and one capped evaluation contract per split.

| Diagnostic | Result |
|---|---:|
| Training contracts | 1 |
| Test contracts | 1 |
| Training loss | 0.7492 |
| Validation loss | 0.3427 |
| Validation macro F1 | 0.0000 |
| Test token macro F1 | 0.0000 |
| Test strict span micro F1 | 0.0000 |
| Training time | 86.818 s |

This was a **pipeline smoke test only**, not a meaningful BERT benchmark. Its
outputs are under ignored local `artifacts/smoke_minimal/` and are not included
in the committed metrics table.

## 6. Interpretation and limitations

1. **The lexical baseline is weak.** Its token macro F1 is 0.008415 and its
   strict-span micro F1 is zero. Partial overlap is higher because an
   overlapping prediction can receive partial credit without exact
   boundaries.
2. **The class distribution is difficult.** `O` accounts for 88.89% of
   priority-resolved tokens, category frequencies vary, and three categories
   have fewer than five test spans.
3. **Span boundaries matter.** Token-level scores and span-level scores answer
   different questions. A token classifier can capture part of a clause
   without recovering an exact legal provision.
4. **Overlaps are a real data property.** Multi-label annotations are
   preserved, but single-label BIO results depend on the documented rare-class
   priority convention.
5. **Model comparison is incomplete.** There is no full-data CRF result, no
   full BiLSTM-CRF result, no full transformer result, no planned checkpoint
   ablation, and no three-seed comparison. The smoke results cannot fill those
   gaps.
6. **Statistical evidence is limited.** The full-data baselines each have one
   seed; the paired comparison concerns only those baselines on the same test
   split. Do not claim robust generalization or model superiority.
7. **Leakage screening is not proof.** The exact duplicate was within one
   split, and the approximate SimHash check found no cross-split pair under
   its threshold. Partial boilerplate overlap may remain undetected.
8. **Human presentation rehearsal is not recorded.** A draft deck, talk track,
   and Q&A notes exist, but no rehearsals should be claimed unless actually
   completed by the presenters.
9. **Team contribution records are not complete.** The repository contains a
   contribution template; named contributions should be added only by the real
   team members based on work they actually completed.

## 7. Repository map and generated artifacts

| Path | Contents |
|---|---|
| `data/raw/` | Official CUAD source JSON and provenance/licensing notes |
| `data/processed/` | Compressed tokenized corpus, preprocessing statistics, correction log |
| `src/cuad_tagging/pipeline.py` | Normalization, offsets, span validation, labels, splits, windows |
| `src/cuad_tagging/metrics.py` | Token/span metrics and contract bootstrap methods |
| `src/cuad_tagging/baselines.py` | O-only, word lookup, CRF, common result export |
| `src/cuad_tagging/bilstm_crf.py` | BiLSTM-CRF training implementation |
| `src/cuad_tagging/transformer.py` | Hugging Face token-classification training implementation |
| `scripts/analyze_data.py` | Source and processed-data analyses |
| `scripts/compare_models.py` | Paired prediction comparison |
| `scripts/audit_predictions.py` | Error categorization and duplicate screening |
| `scripts/summarize_experiments.py` | Aggregates, per-category reports, findings summary |
| `results/metrics.csv`, `results/metrics.tex` | Run-level metric tables with scope metadata |
| `results/experiment_summary.json` | Machine-readable summary and caveats |
| `results/findings_summary.md` | Short findings report |
| `results/analysis/` | Dataset analysis tables, plots, and LaTeX tables |
| `results/audit/` | Sampled errors and duplicate-screening report |
| `results/CUAD_project_presentation.pptx` | Draft 10-slide deck |
| `docs/experiments.md` | Training and evaluation protocol |
| `docs/presentation_rehearsal.md` | Suggested talk track and rehearsal checklist |
| `docs/presentation_qa.md` | Likely presentation questions and answers |

## 8. Suggested presentation wording

### Short project description

“We built an auditable pipeline for tagging 41 types of legal clause spans in
CUAD contracts. It validates source offsets, preserves overlapping annotations,
splits by contract, supports long-document windows, and reports token-level and
span-level metrics.”

### Results statement

“On our 51-contract test split, the most-frequent-word baseline reached 0.0084
token macro F1, with a contract-bootstrap interval of 0.0067 to 0.0096. Its
strict span micro F1 was zero, while partial span micro F1 was 0.1695. The
all-`O` baseline scored zero. Our CRF was trained on only 10 contracts and
BERT was tested only in a one-contract smoke run, so we do not claim a complete
neural-model comparison.”

### If asked whether the study is finished

“The data pipeline, audits, baseline results, reproducibility workflow, tests,
and draft presentation are complete. The model study is still in progress:
full-data neural runs, a full-data CRF, ablations, and repeated seeds remain.”

## 9. References

- The Atticus Project. [CUAD: Contract Understanding Atticus Dataset and
  repository](https://github.com/The-Atticus-Project/cuad).
- Hendrycks, D. et al. “CUAD: An Expert-Annotated NLP Dataset for Legal
  Contract Review.” NeurIPS 2021 Datasets and Benchmarks.
- [Public project repository](https://github.com/hmam-5/cuad-clause-tagging).
