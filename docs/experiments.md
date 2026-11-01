# Training and evaluation protocol

## Baselines

`scripts/train_baselines.py` implements:

1. All-`O` predictions.
2. Most-frequent BIO tag for each lowercased word, learned on train contracts.
3. A linear-chain CRF with word, casing, affix, shape, and neighbor features
   (`sklearn-crfsuite`).

The CRF fits on the configured train split; `--max-train-contracts` is a smoke
test only and must not be presented as a full-data result. The O-only
classifier is a diagnostic: since its precision/recall are zero for every
positive clause class, its main scores should be zero even though the O token
share is high.

The Windows/Make reproduction target uses the explicitly limited
`--max-train-contracts 10` CRF smoke run. Use `make crf` or
`python scripts/train_baselines.py --model crf` for a full fit; this feature
CRF is CPU-intensive and does not provide epoch-level validation-loss early
stopping in the current sklearn-crfsuite implementation.

## Neural models

`scripts/train_bilstm_crf.py` trains a bidirectional LSTM with a linear-chain
CRF and random 100-dimensional embeddings by default. It samples O-only
training windows (25% retained by default) to reduce wasted compute; labeled
windows are retained. The run metadata records this choice.

`scripts/train_transformer.py` supports Hugging Face fast tokenizers and
token-classification checkpoints. The intended comparison is:

| Architecture | Exact checkpoint |
|---|---|
| BERT | `bert-base-cased` |
| BERT | `bert-base-uncased` |
| BERT | `nlpaueb/legal-bert-base-uncased` |
| RoBERTa | `roberta-base` |
| DeBERTa | `microsoft/deberta-v3-base` |

Only first subtokens contribute to the loss and metrics. Word windows overlap;
subword logits are averaged when stitching a token's repeated predictions.
Windows are bounded by both the configured word count and the tokenizer's
subword-piece budget, so a word that expands into many pieces cannot silently
truncate neighboring tokens. A span is kept intact where its full tokenized
length fits during training; exceptionally long spans can still cross windows.
Validation and test window boundaries depend only on tokens and subword lengths,
not on gold BIO tags. RoBERTa fast tokenizers enable `add_prefix_space` for
pretokenized word input.
Class weights use inverse-square-root train frequencies, capped at 10 before
normalization. A fast tokenizer window that would truncate words fails loudly;
adaptive subword-safe boundaries are applied before encoding rather than
silently dropping tokens. DeBERTa-v3 requires `protobuf` for SentencePiece
tokenizer conversion, and Transformers warns that the converted fast tokenizer
does not implement byte fallback; some pieces may become unknown tokens.

Transformer training defaults to a physical batch size of 2 with 8 gradient
accumulation steps (effective batch size 16), AdamW, a linear learning-rate
schedule with 10% warmup, gradient clipping at 1.0, and automatic CUDA bf16 or
fp16 when supported. All labeled windows are retained and 25% of O-only windows
are sampled by default. Windows are tokenized once and cached before epochs.
Checkpoint selection uses validation token macro F1, breaking ties by lower
validation loss; the test split is evaluated once after loading the selected
checkpoint. The loss CSV records
training/validation loss, validation macro F1, learning rate, elapsed time, and
the selected epoch. Transformer outputs canonicalize malformed BIO sequences:
an `I-category` without an active same-category span becomes `B-category`,
matching the span decoder's existing behavior. Smoke runs report training-set
macro F1 and raw versus canonical B-tag counts as an overfit diagnostic.

## Current benchmark status

The completed full-data table contains seed-42 results for the CRF,
BiLSTM-CRF,
`bert-base-cased`, `bert-base-uncased`,
`nlpaueb/legal-bert-base-uncased`, `roberta-base`, and DeBERTa, plus a second
LegalBERT run with seed 43. Both LegalBERT results are reported as separate
seed-specific rows, not combined into a mean or standard deviation. The
validation-selected weighted-vote ensemble is a separate exploratory record.
All completed runs use the same single contract-level split. The CRF reached
its 100-iteration cap without convergence. The full BiLSTM-CRF seed-42 run
completed eight epochs; its selected checkpoint and test results are recorded
in `results/bilstm-crf-seed42.json`, its loss history in
`results/loss/bilstm-crf-seed42.csv`, and its audit in
`results/audit/bilstm_seed42/`. The tracked metrics and latest completion state are in `results/metrics.csv`,
`results/findings_summary.md`, and the repository README.

The BiLSTM-CRF used 100-dimensional random embeddings, hidden size 128,
batch size 16, AdamW at 1e-3, 25% O-only-window retention, and eight requested
epochs on the 408/51/51 contract split. Its best validation token macro F1 was
0.335704 at epoch 8. Training took 3,258.21 seconds. On the 51-contract test
split it scored token macro F1 0.335111 (95% contract-bootstrap CI
0.296805–0.351668), strict span micro/macro F1 0.150849/0.087415, and partial
span micro/macro F1 0.423576/0.344502. The test audit found 1,314 span errors:
634 missed (48.25%), 294 hallucinated (22.37%), 239 boundary (18.19%), and
147 wrong-category (11.19%). These results are descriptive and were not used
for checkpoint selection. The BiLSTM-CRF has a lower token macro F1 than the
transformers but higher partial span micro F1 than each individual transformer
run; it does not resolve the CRF's missed-span tendency.

The full DeBERTa-v3 seed-42 run completed all 8 configured epochs on the
408/51/51 contract split. The checkpoint was selected at epoch 8 by validation
token macro F1 (0.532786); training took 8,399.76 seconds. On the 51-contract
test split it scored token macro F1 0.540219 (95% contract-bootstrap CI
0.479802–0.564610), strict span micro/macro F1 0.216643/0.177191, and partial
span micro/macro F1 0.408182/0.385561. Test metrics are descriptive and were
not used for checkpoint selection. Training loss fell to 0.0361 while
validation loss rose to 0.2758; validation F1 continued to improve, so the
checkpoint rule remained validation-F1 based.

The DeBERTa test audit found 2,867 span errors: 2,162 hallucinated (75.41%),
386 boundary (13.46%), 202 missed (7.05%), and 117 wrong-category (4.08%).
Thus, completing DeBERTa did not eliminate the transformer hallucination-heavy
error profile. The validation-selected scalar non-O offset and per-category
logit-bias experiments in `results/exploratory_calibration.json` did not
improve both strict and partial span F1; they were not selected using test
metrics.

An alternate seed-43 contract split has been prepared and integrity-checked:
all 510 contract IDs are present, 153 assignments differ from the original,
and sizes are 407/51/52 (train/validation/test). No model has yet been trained
or evaluated on that split, so it does not change the single-split limitation
of the reported model comparisons.

The original four seed-42 transformer result records include 128-word windows,
32-word stride, and the 25% O-only retention setting. Their metadata does not
include a source revision or an explicit gold-independent evaluation-window
flag. Their historical metadata therefore cannot by itself attest to the
evaluation code used in those original runs; the post-hoc verification below
re-evaluates their saved checkpoints with the fixed code. The new DeBERTa run
used the current training/evaluation implementation and the same 128/32
transformer window settings.

### Post-hoc fixed-window verification

On 2026-11-01, each available seed-42 best checkpoint was loaded for
inference-only scoring on the same 51 test contracts using the current
gold-independent evaluation windows in `src/cuad_tagging/transformer.py`.
Inference used CUDA bf16 and did not invoke training or change the recorded
results in `results/metrics.csv`. Values below show recorded -> re-evaluated
with the signed difference (re-evaluated minus recorded).

| Model | Token macro F1 | Strict span micro F1 | Partial span micro F1 |
|---|---|---|---|
| BERT `bert-base-cased` | 0.507604 -> 0.507618 (+0.000014) | 0.151697 -> 0.151697 (0) | 0.330553 -> 0.330553 (0) |
| BERT `bert-base-uncased` | 0.482711 -> 0.482711 (0) | 0.157749 -> 0.157749 (0) | 0.336504 -> 0.336504 (0) |
| LegalBERT `nlpaueb/legal-bert-base-uncased` | 0.512516 -> 0.512521 (+0.000004) | 0.204417 -> 0.204417 (0) | 0.407895 -> 0.407895 (0) |
| RoBERTa `roberta-base` | 0.520201 -> 0.520236 (+0.000035) | 0.201801 -> 0.202174 (+0.000373) | 0.398522 -> 0.399260 (+0.000737) |

The uncased BERT scores are identical. The cased BERT and LegalBERT span
scores are identical, but their token macro F1 differs slightly; RoBERTa
differs on all three scores. These are reported as differences, not as
identical results, and the recorded scores have not been replaced.

The window-fix commit `17b18e1` was created at 2026-11-01 03:25:52 +03
(00:25:52 UTC). The selected checkpoint and run-record modification times
below are UTC; all are later than the fix commit. This supports, but cannot
prove, which source revision generated each run. A dedicated log for the
selected cased retry and logs for the other three selected runs were not
retained. The older cased logs show a pre-fix run (00:23:29/00:24:03 UTC)
and a separate partial leak-fix run (00:26:28/01:03:07 UTC), not the selected
retry listed here.

| Model | Selected run-record mtime (UTC) | Best checkpoint `model.safetensors` mtime (UTC) |
|---|---|---|
| BERT `bert-base-cased` | 2026-11-01 02:24:17 | 2026-11-01 02:23:31 |
| BERT `bert-base-uncased` | 2026-11-01 03:36:03 | 2026-11-01 03:35:17 |
| LegalBERT `nlpaueb/legal-bert-base-uncased` | 2026-11-01 04:50:37 | 2026-11-01 04:41:47 |
| RoBERTa `roberta-base` | 2026-11-01 06:00:31 | 2026-11-01 05:59:45 |

## Metrics and uncertainty

`src/cuad_tagging/metrics.py` implements:

- Token-level macro F1 over all 41 positive clause categories (O excluded).
- Strict span micro/macro F1 with exact token-boundary match.
- Partial-overlap span micro/macro F1 with one-to-one matching of overlapping
  same-category spans.
- Percentile 95% confidence intervals by resampling contracts, not tokens.
- Paired contract bootstrap for model differences.

The bootstrap uses the fixed seed in the command and reports the number of
iterations. Run three seeds (five if compute permits) for each main model;
report each seed's scores separately alongside its test-contract bootstrap
interval. Do not collapse a small number of runs into a mean and standard
deviation. Do not select hyperparameters on the test set. The training scripts
select checkpoints using validation token macro F1.

`scripts/compare_models.py` compares paired predictions on identical test
contracts and reports a two-sided paired-bootstrap p-value and confidence
interval for token macro-F1 differences.
`scripts/audit_predictions.py` labels sampled span errors as boundary, wrong
category, missed, or hallucinated; it also reports exact and approximate
near-duplicate contracts across splits. Its 64-bit SimHash uses at most 512
uniformly spaced word shingles per contract. It is a screening check, not a
proof that no boilerplate leakage exists.
`scripts/summarize_experiments.py` aggregates only full-data runs, writes
individual seed rows in its tables and findings summary, and creates
per-category and loss-curve figures. It does not collapse the available
LegalBERT seeds into a mean or standard deviation. A model with fewer than
three completed seeds is explicitly exploratory. Completed full transformer
test and validation records are imported from their run directories
under ignored `artifacts/`; their loss histories are copied into tracked
`results/loss/`. Incomplete transformer runs are not imported.
The exploratory validation-selected calibration records are in
`results/exploratory_calibration.json`; test metrics there are descriptive
only and were not used for selection.

## Reproducible runs

```powershell
python scripts/train_baselines.py --model all --seed 42
python scripts/train_bilstm_crf.py --seed 42
python scripts/train_transformer.py --checkpoint bert-base-cased --seed 42
python scripts/train_transformer.py --checkpoint bert-base-uncased --seed 42
python scripts/train_transformer.py --checkpoint nlpaueb/legal-bert-base-uncased --seed 42
python scripts/train_transformer.py --checkpoint roberta-base --seed 42
python scripts/train_transformer.py --checkpoint microsoft/deberta-v3-base --seed 42
```

Repeat with seeds 43 and 44, then compare prediction files from the same
contract-level test split. Loss history is CSV under `results/loss/`; transformer
and BiLSTM-CRF scripts also produce loss figures. Checkpoints go to the
Git-ignored `artifacts/checkpoints/`. To aggregate available loss histories into
`results/figures/`, run `python scripts/plot_loss_curves.py`.
