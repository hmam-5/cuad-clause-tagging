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
`bert-base-cased`, `bert-base-uncased`,
`nlpaueb/legal-bert-base-uncased`, and `roberta-base`. Each was evaluated on
the same single contract-level split. LegalBERT seed 43 is currently being
run; it is not a result until the full run and validation-selected test
evaluation finish. The CRF reached its 100-iteration cap without convergence.
The DeBERTa full run is incomplete and has no test score; BiLSTM-CRF is
implemented but has not been run as a full experiment. The tracked metrics
and latest completion state are in `results/metrics.csv`,
`results/findings_summary.md`, and the repository README.

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
`scripts/summarize_experiments.py` aggregates only full-data runs, reports
mean/standard deviation across available seeds, writes a one-page findings
summary, and creates per-category and loss-curve figures. A model with fewer
than three completed seeds is explicitly exploratory. Completed full
transformer test and validation records are imported from their run directories
under ignored `artifacts/`; their loss histories are copied into tracked
`results/loss/`. Incomplete transformer runs are not imported.

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
