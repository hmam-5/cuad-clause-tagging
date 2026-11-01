# CUAD Clause Tagging

Reproducible NLP project for finding and classifying legal clause spans in
CUAD v1. The goal is token classification over 41 lawyer-annotated contract
clause categories represented by 83 BIO labels. The labels describe the legal
function of a passage; this is not legal advice.

This task is difficult because clauses may be long, overlap, and occur in
contracts from different domains; most tokens are outside any annotated
clause; and some categories have very few examples. A correct boundary and
category are both needed for a strict span match.

## Dataset and measured preprocessing

The project downloads the official Atticus Project release from its
[CUAD GitHub repository](https://github.com/The-Atticus-Project/cuad). The
retained raw file is the original `CUADv1.json` from the official `data.zip`;
the contexts contain the contract text and the QA-style character offsets.
This source release does not provide a separate `full_contracts_txt` directory.
CUAD is licensed CC BY 4.0; retain attribution to The Atticus Project and the
CUAD paper authors. The project code is MIT licensed; these licenses are
distinct.

The original raw JSON is about 40.1 MB, below 50 MiB. On the downloaded
release, preprocessing found:

| Measure | Result |
|---|---:|
| Contracts / categories | 510 / 41 |
| Questions with no answer spans | 14,208 of 20,910 |
| Non-empty answer spans | 13,823 |
| Empty answer entries / non-empty answers missing offsets | 0 / 0 |
| Exact offsets kept / corrected / dropped | 13,823 / 0 / 0 |
| Mean span length | 262 characters / 50.02 word tokens |
| 95th-percentile span length | 758 characters / 147 word tokens |
| Word tokens | 5,114,638 |
| Outside (`O`) share after priority resolution | 88.89% |
| Tokens carrying multiple categories | 90,720 (1.77%) |
| Cross-category overlapping annotation pairs | 2,122 |
| Contract-level split | 408 train / 51 validation / 51 test |
| 512-word windows, stride 128 | 13,274 |
| Token-span round-trip agreement | 100% |
| Invisible Unicode format characters removed | 184 |

Three categories have fewer than five test spans (`Most Favored Nation`,
`Price Restrictions`, `Affiliate License-Licensor`); their test F1 will be
uncertain. These are dataset/preprocessing counts, not model results. The
measured all-`O` baseline has 0.0000 token macro F1 and 0.0000 strict-span
micro F1. The most-frequent-word baseline has 0.0084 token macro F1 (contract
bootstrap 95% CI 0.0067–0.0096), 0.0000 strict-span micro F1, and 0.1110
partial-span macro F1. The severe class imbalance and scarce categories make
macro-F1 and per-class span evaluation more informative than accuracy alone.
Plain token accuracy can exceed 90% because 88.89% of tokens are `O`; it is
therefore not a suitable headline measure of clause extraction quality.

## Full-data results

The table reports full-data runs on the fixed 51-contract test split. Token
macro F1 is the average across the 41 clause categories (the `O` label is
excluded). Span micro F1 uses exact boundaries for strict scoring and any
overlap for partial scoring. LegalBERT seed 43 is the best single run by
validation token macro F1. A separate ensemble uses weights selected on validation from
80 combinations; the test set was not used for that selection. Its result is
exploratory because the validation grid search can overfit.

| Model / checkpoint | Seed | Validation token macro F1 | Test token macro F1 | Test strict span micro F1 | Test partial span micro F1 |
|---|---:|---:|---:|---:|---:|
| CRF | 42 | 0.3058 | 0.3103 | 0.1035 | 0.3788 |
| BERT `bert-base-cased` | 42 | 0.5105 | 0.5076 | 0.1517 | 0.3306 |
| BERT `bert-base-uncased` | 42 | 0.5120 | 0.4827 | 0.1577 | 0.3365 |
| LegalBERT `nlpaueb/legal-bert-base-uncased` | 42 | 0.5387 | 0.5125 | 0.2044 | 0.4079 |
| LegalBERT `nlpaueb/legal-bert-base-uncased` | 43 | **0.5631** | 0.5118 | 0.2072 | 0.3893 |
| RoBERTa `roberta-base` | 42 | 0.5223 | 0.5202 | 0.2018 | 0.3985 |
| DeBERTa `microsoft/deberta-v3-base` | 42 | 0.5328 | **0.5402** | 0.2166 | 0.4082 |
| Weighted-vote ensemble (uncased BERT 1, LegalBERT 2, RoBERTa 2) | 42 per model | 0.5528 | 0.5342 | **0.2242** | **0.4423** |

Model rows are from [`results/metrics.csv`](./results/metrics.csv); the
validation-selected ensemble row is from
[`results/ensemble_validation_selected.json`](./results/ensemble_validation_selected.json).
Rows report individual seeds: LegalBERT has two runs; each other completed
model has one. DeBERTa has the highest individual test token macro F1, while
LegalBERT seed 43 remains the best individual run by validation F1.
Bootstrap intervals quantify test-contract resampling, not run-to-run training
variation. The seed-42/43 paired LegalBERT comparison is in
[`results/paired_comparison_legalbert_seed42_vs_seed43.json`](./results/paired_comparison_legalbert_seed42_vs_seed43.json);
its difference was not significant (p=0.8106). The ensemble result and its 95% test
contract-bootstrap interval (0.4791–0.5571) are recorded separately in
[`results/ensemble_validation_selected.json`](./results/ensemble_validation_selected.json).
Its paired improvement over RoBERTa was not significant (p=0.1819).
Two validation-selected LegalBERT calibration experiments are recorded in
[`results/exploratory_calibration.json`](./results/exploratory_calibration.json)
and are excluded from the primary table: the scalar offset slightly raised
test token macro F1 but reduced both span scores, while the 41-parameter
per-category calibration did not improve test token macro F1. Test results
were descriptive and were not used for calibration selection.
The four seed-42 transformer checkpoints were re-evaluated with the current
gold-independent test-window code. Uncased BERT matched all three reported
scores exactly; the other runs had only small metric differences. The original
table values were retained. See the [verification record](./docs/experiments.md#post-hoc-fixed-window-verification)
for exact comparisons and timestamp limitations.

### CRF and transformer span behavior

The transformers and their ensemble have higher strict span micro F1 than the
CRF, while the CRF's partial span score is closer to the individual
LegalBERT/RoBERTa models than its strict score suggests. Test audits show
different error profiles: the CRF mostly misses gold spans; the transformers
mostly hallucinate spans.

| Model | Strict span micro F1 | Partial span micro F1 | Missed spans | Hallucinated spans |
|---|---:|---:|---:|---:|
| CRF | 0.1035 | 0.3788 | 837 | 79 |
| LegalBERT | 0.2044 | 0.4079 | 226 | 2,140 |
| RoBERTa | 0.2018 | 0.3985 | 233 | 2,222 |
| DeBERTa | 0.2166 | 0.4082 | 202 | 2,162 |

Counts are from the test audits in
[`results/audit/crf_full_data/`](./results/audit/crf_full_data/),
[`results/audit/legalbert_full_data/`](./results/audit/legalbert_full_data/),
[`results/audit/roberta_full_data/`](./results/audit/roberta_full_data/), and
[`results/audit/deberta_seed42/`](./results/audit/deberta_seed42/).

## Quick start

Python 3.10+ is required. From the project root:

```powershell
python -m pip install -r requirements.txt
python scripts/download_data.py
python scripts/prepare_data.py
python scripts/analyze_data.py
python scripts/summarize_experiments.py
python scripts/train_baselines.py --model o-only
python -m unittest discover -s tests
```

The first four commands download the official raw source, regenerate prepared
data and descriptive analysis, and rebuild result tables from the recorded
run JSON files and loss histories in `results/`. Reproducing model scores
requires rerunning the corresponding training commands below; raw annotations
alone do not determine trained model results.

`python scripts/train_baselines.py --model most-frequent-word` runs the
lexical baseline. The feature CRF is available as
`python scripts/train_baselines.py --model crf`; it is CPU- and
memory-intensive. Use `--max-train-contracts 10` for a smoke run, not a
reported final comparison. All splits are contract-level, and no window from
one contract can leak into another split.

The transformer trainer accepts Hugging Face token-classification
checkpoints. Install a PyTorch build suitable for your hardware. Checkpoints
and run logs are written under ignored `artifacts/`.

```powershell
python scripts/train_transformer.py --checkpoint bert-base-cased --seed 42 --epochs 1 --max-train-contracts 4 --max-eval-contracts 2 --output-dir artifacts/smoke
python scripts/train_bilstm_crf.py --seed 42
python scripts/train_transformer.py --checkpoint bert-base-cased --seed 42 --epochs 8 --patience 2 --batch-size 2 --gradient-accumulation-steps 8 --learning-rate 3e-5 --max-length 512 --window-words 128 --stride-words 32 --o-only-fraction 0.25 --mixed-precision auto
python scripts/train_transformer.py --checkpoint bert-base-uncased --seed 42 --epochs 8 --patience 2 --batch-size 2 --gradient-accumulation-steps 8 --learning-rate 3e-5 --max-length 512 --window-words 128 --stride-words 32 --o-only-fraction 0.25 --mixed-precision auto
python scripts/train_transformer.py --checkpoint nlpaueb/legal-bert-base-uncased --seed 42 --epochs 8 --patience 2 --batch-size 2 --gradient-accumulation-steps 8 --learning-rate 3e-5 --max-length 512 --window-words 128 --stride-words 32 --o-only-fraction 0.25 --mixed-precision auto
python scripts/train_transformer.py --checkpoint roberta-base --seed 42 --epochs 8 --patience 2 --batch-size 2 --gradient-accumulation-steps 8 --learning-rate 3e-5 --max-length 512 --window-words 128 --stride-words 32 --o-only-fraction 0.25 --mixed-precision auto
```

Checkpoint selection uses validation token macro F1. The test split is
evaluated after selection and is not used to tune hyperparameters. For a
multi-seed study, change `--seed` while keeping the contract split fixed.

Run each main model with at least three seeds for the final report. Compare
saved prediction files with `scripts/compare_models.py`; run
`scripts/audit_predictions.py` on a test prediction file for 30 sampled span
errors and exact/near-duplicate contract checks. The near-duplicate test uses
a 64-bit SimHash over up to 512 uniformly spaced word shingles per contract,
so it is approximate and can miss boilerplate shared only by parts of otherwise
different contracts.
The reduced train/evaluation caps in the first command are only for a quick
pipeline smoke test; those outputs are explicitly marked `smoke` and must not
be included in full-data comparisons.

On Windows, run `powershell -NoProfile -ExecutionPolicy Bypass -File .\run_all.ps1`
to run data acquisition, preprocessing, dataset analysis, the O-only and
most-frequent-word baselines, a clearly labeled 10-contract CRF smoke run,
paired baseline comparison, sampled-error and duplicate audits,
experiment summarization, the presentation builder, and tests. `make reproduce`
does the same on systems with `make`; `make crf` requests a full CRF fit and
`make smoke-crf` runs the limited diagnostic. Install dependencies with
`python -m pip install -r requirements.txt`.
The generated draft slides are at `results/CUAD_project_presentation.pptx`;
rehearsal and likely-question guidance are in
`docs/presentation_rehearsal.md` and `docs/presentation_qa.md`. The deck labels
experiments according to the results actually recorded.

The most-frequent-word model produced 1,981 test span errors; the audit sampled
30 (14 missed, 12 hallucinated, 3 boundary, and 1 wrong-category errors). The
duplicate audit found one exact pair among 510 contracts; both are in the same
split, with no detected cross-split near duplicates at Hamming distance 4.
SimHash is approximate and does not rule out partial boilerplate overlap.

## Reproducibility and outputs

- Fixed default seed: `42` (`src/cuad_tagging/config.py`).
- Source offsets are validated against the original text. A nearby offset is
  corrected only when the exact answer text has one unique match within 32
  characters; every correction or drop is recorded.
- Whitespace normalization retains a normalized-to-source offset mapping.
  Tokens and spans include both cleaned and original offsets.
- Multi-label targets preserve overlaps. The single-label BIO baseline resolves
  overlaps in favor of the globally rarer category, with category order as a
  tie-breaker. This is an explicit modeling choice.
- Contracts are split before windowing using a seeded multilabel stratifier.
- Preprocessed output is compact LZMA-compressed JSONL at
  `data/processed/contracts.jsonl.xz`; its parallel arrays and category IDs are
  documented in [docs/preprocessing.md](./docs/preprocessing.md). It can be
  regenerated from the raw JSON and is safe to keep under the 50 MiB file
  threshold.
- Prepared-data analysis uses 512-word windows with stride 128
  ([`data/processed/stats.json`](./data/processed/stats.json)). Transformer
  training re-tokenizes and creates subword-safe windows of up to 128 words
  with 32-word overlap; it retains all labeled training windows and 25% of
  O-only training windows. These are distinct windowing stages and settings.
- Dataset tables and plots are written under `results/analysis/`. Metric
  tables, summaries, audits, paired comparisons, and loss CSVs are tracked
  under `results/`; checkpoints, model predictions, and run logs are ignored
  under `artifacts/`.
- `results/findings_summary.md` and `results/experiment_summary.json` are
  regenerated by `scripts/summarize_experiments.py`; they exclude smoke runs
  from full-data aggregates and state seed counts and caveats.
- `results/analysis/source_annotation_audit.json` reports 14,208 empty-answer
  questions and zero empty answer entries or non-empty answers missing offsets.

## Repository map

```text
data/                           Source, processed data, and derived dataset assets
data/raw/                       Downloaded source data and provenance
data/processed/                 Compressed contracts, labels, splits, and preparation statistics
scripts/                        Data, training, evaluation, and reporting entry points
src/                            Python package sources
src/cuad_tagging/               Configuration, preprocessing, metrics, baselines, and model trainers
scripts/download_data.py        Download CUAD source data and record provenance
scripts/prepare_data.py         Validate spans and create processed contracts
scripts/analyze_data.py         Summarize dataset distributions and export plots
scripts/train_baselines.py      Train O-only, word-lookup, or CRF baselines
scripts/train_bilstm_crf.py     Train the BiLSTM-CRF (implemented, not benchmarked)
scripts/train_transformer.py    Fine-tune and evaluate transformer token classifiers
scripts/compare_models.py       Compare prediction files with paired contract bootstrap
scripts/audit_predictions.py    Audit span errors and screen for duplicate contracts
scripts/summarize_experiments.py Export result tables and experiment summaries
scripts/plot_loss_curves.py     Plot recorded training and validation losses
scripts/build_presentation.py   Build a PowerPoint from recorded project results
run_all.ps1                     Run the Windows reproduction pipeline
docs/                           Data-preparation and experiment documentation
tests/                          Unit and integration tests
contributions/                  Per-member contribution records and templates
results/analysis/               Dataset analysis tables and plots
results/audit/                  Test error and duplicate-screening audits
results/loss/                   Tracked transformer loss histories
results/                         Tracked experiment metrics, summaries, comparisons, and plots
artifacts/                      Ignored checkpoints, predictions, and run logs
```

## Attribution

Hendrycks, D., Burns, C., Chen, A., and Ball, S. (2021). *CUAD: An
Expert-Annotated NLP Dataset for Legal Contract Review*. NeurIPS.
[Paper](https://arxiv.org/abs/2103.06268) ·
[Dataset and source repository](https://github.com/The-Atticus-Project/cuad).

## Project status and limitations

The raw data, auditable cleaning, BIO/multi-label conversion, contract splits,
windowing, dataset analysis, scoring utilities, baselines, prediction audits,
and transformer/BiLSTM-CRF training code are implemented. All completed
full-data models use one fixed contract split; LegalBERT has two completed
seeds and each other completed model has one. The CRF reached its 100-iteration
cap without convergence. Three categories have fewer than five gold test
spans. Duplicate screening uses approximate SimHash and cannot rule out
partial boilerplate overlap. DeBERTa completed its eight-epoch seed-42 run;
BiLSTM-CRF has not yet been run as a full experiment. Transformer validation
loss rises after about epoch 2 while validation F1 continues to improve;
checkpoints are selected by validation token macro F1 (validation loss only
breaks ties). DeBERTa's test audit is still hallucination-heavy. The original
four seed-42 transformer result records report 128-word windows and stride 32,
but do not record a code
revision or evaluation-window-boundary flag. The fixed leakage-prevention code
is present in commit `17b18e1`; historical run metadata alone cannot prove
which source revision produced those outputs. The saved run directories were
created after that fix commit, which corroborates—but does not replace—the
missing per-run source revision. LegalBERT seed 43 completed all 8 requested
epochs within the 80-minute cap and is included as its own result row. Loss
curves show validation loss
rising after about epoch 2, while validation F1 often continues to improve
through later epochs; checkpoint selection uses validation token macro F1.

Contribution records must use real names, IDs, accounts, and commit history.
At baseline commit `0eae55f`, Git recorded one author (`hmam-5`) with eight
commits. The contribution log documents that baseline and how to inspect the
current history; see [contributions/README.md](./contributions/README.md).
