# CUAD Clause Tagging

Reproducible NLP course project for tagging legal clause spans in the
Contract Understanding Atticus Dataset (CUAD v1). The task is token
classification over 41 lawyer-annotated clause categories, represented as 83
BIO labels. It is not named-entity recognition: the labels describe the legal
function of a passage.

> **Scope and caveat:** the source contains legal contracts and annotations;
> this is an NLP research project, not legal advice. No model scores are
> claimed until the corresponding experiment has been run.

## Dataset and measured preprocessing

The project downloads the official Atticus Project release from its
[CUAD GitHub repository](https://github.com/The-Atticus-Project/cuad). The
retained raw file is the original `CUADv1.json` from the official `data.zip`;
the contexts contain the contract text and the QA-style character offsets.
This source release does not provide a separate `full_contracts_txt` directory.
The separate Zenodo archive is 105,883,672 bytes and is not included because it
exceeds the project's 50 MiB per-file limit. CUAD is licensed CC BY 4.0; retain
attribution to The Atticus Project and the CUAD paper authors. The project code
is MIT licensed; these licenses are distinct.

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

For a presentation-ready walkthrough of the task, complete pipeline, measured
results, caveats, and likely questions, see
[docs/PROJECT_REPORT.md](./docs/PROJECT_REPORT.md). It distinguishes
implemented components from experiments that are still pending.

Three categories have fewer than five test spans (`Most Favored Nation`,
`Price Restrictions`, `Affiliate License-Licensor`); their test F1 will be
uncertain. These are dataset/preprocessing counts, not model results. Run the
majority baseline before making a go/no-go claim about task difficulty. The
measured all-`O` baseline has 0.0000 token macro F1 and 0.0000 strict-span micro
F1. The most-frequent-word baseline has 0.0084 token macro F1 (contract
bootstrap 95% CI 0.0067–0.0096), 0.0000 strict-span micro F1, and 0.1110
partial-span macro F1. These trivial baselines do not saturate the task. This
alone is not a final go/no-go; compare a trained feature CRF and transformer
before claiming feasibility.

## Quick start

Python 3.10+ is required. From the project root:

```powershell
python -m pip install -r requirements.txt
python scripts/download_data.py
python scripts/prepare_data.py
python scripts/analyze_data.py
python scripts/train_baselines.py --model o-only
python -m unittest discover -s tests
```

`python scripts/train_baselines.py --model most-frequent-word` runs the
lexical baseline. The feature CRF is available as
`python scripts/train_baselines.py --model crf`; it is CPU- and
memory-intensive. Use `--max-train-contracts 10` for a smoke run, not a
reported final comparison. All splits are contract-level, and no window from
one contract can leak into another split.

The transformer experiment accepts any Hugging Face token-classification
checkpoint, so the planned BERT, LegalBERT, RoBERTa, and DeBERTa comparisons
use one training/evaluation pipeline. Install a PyTorch build suitable for
your hardware before those runs. Save checkpoints under `artifacts/` (ignored
by Git); report architecture and exact checkpoint separately.

```powershell
python scripts/train_transformer.py --checkpoint bert-base-cased --seed 42 --epochs 1 --max-train-contracts 4 --max-eval-contracts 2 --output-dir artifacts/smoke
python scripts/train_bilstm_crf.py --seed 42
python scripts/train_transformer.py --checkpoint bert-base-cased --seed 42
python scripts/train_transformer.py --checkpoint nlpaueb/legalbert-base-uncased --seed 42
python scripts/train_transformer.py --checkpoint roberta-base --seed 42
python scripts/train_transformer.py --checkpoint microsoft/deberta-v3-base --seed 42
```

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
- Dataset tables and plots are written under `results/analysis/`. Model
  metrics, predictions, and loss curves are written under `results/`; these
  outputs are kept by `.gitignore`.
- `results/findings_summary.md` and `results/experiment_summary.json` are
  regenerated by `scripts/summarize_experiments.py`; they exclude smoke runs
  from full-data aggregates and state seed counts and caveats.
- `results/analysis/source_annotation_audit.json` reports 14,208 empty-answer
  questions and zero empty answer entries or non-empty answers missing offsets.

## Repository map

```text
data/raw/                  Official source JSON and provenance
data/processed/            Reproducible compressed tokenized dataset and stats
src/cuad_tagging/          Config, offsets/windows/splits, metrics, baselines, neural trainers
scripts/download_data.py   Fetch official source JSON and record provenance
scripts/prepare_data.py    Create the compact offset-preserving processed dataset
scripts/analyze_data.py    Export class/span/split tables and dataset figures
scripts/train_baselines.py Run O-only, word lookup, or CRF baseline
scripts/train_bilstm_crf.py Train the random-embedding BiLSTM-CRF
scripts/train_transformer.py Fine-tune a Hugging Face token classifier
scripts/compare_models.py  Paired contract-bootstrap comparison
scripts/audit_predictions.py Sample span errors and check duplicate contracts
scripts/summarize_experiments.py Create Stage 6 findings and model figures
scripts/build_presentation.py Generate the evidence-based PowerPoint draft
docs/                      Preprocessing, experiment protocol, rehearsal and Q&A
tests/                     Offset, tagging, split, metric, audit, integration tests
contributions/             Per-member contribution template
results/                   Generated CSV/LaTeX tables, plots, and metrics
```

## Attribution

Hendrycks, D., Burns, C., Chen, A., and Ball, S. (2021). *CUAD: An
Expert-Annotated NLP Dataset for Legal Contract Review*. NeurIPS.
[Paper](https://arxiv.org/abs/2103.06268) ·
[Dataset](https://zenodo.org/records/4595826) ·
[Source repository](https://github.com/The-Atticus-Project/cuad).

## Project status and limitations

The raw data, auditable cleaning, BIO/multi-label conversion, contract splits,
windowing, dataset analysis, scoring utilities, majority/CRF baselines,
prediction audits, and the transformer/BiLSTM-CRF trainers are implemented.
The CRF row in the checked-in table is a 10-contract smoke result, not a
full-training score. Run the neural models and repeat the main comparisons
with at least three seeds before claiming a final ranking. The generated
`results/findings_summary.md` reports actual runs and flags single-seed
uncertainty. Complete one separately named file per real team member based on
`contributions/CONTRIBUTIONS_TEMPLATE.md`; never invent names, commits, or
contribution history.
