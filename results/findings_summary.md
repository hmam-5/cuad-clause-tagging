# CUAD project: findings summary

## Three key numbers
- **510 contracts** and **13,823 validated spans**.
- **100%** token-to-span round-trip agreement.
- Highest validation-selected group: **BERT | nlpaueb/legal-bert-base-uncased** (validation token macro F1 **0.5387**).

## Full-data test results (descriptive; not used for selection)

| Model | Seeds | Token macro F1 mean (SD) | Strict span micro F1 | Partial span micro F1 | Mean train minutes |
|---|---:|---:|---:|---:|---:|
| RoBERTa \| roberta-base | 1 | 0.5202 (0.0000) | 0.2018 | 0.3985 | 65.56 |
| BERT \| nlpaueb/legal-bert-base-uncased | 1 | 0.5125 (0.0000) | 0.2044 | 0.4079 | 64.53 |
| BERT \| bert-base-cased | 1 | 0.5076 (0.0000) | 0.1517 | 0.3306 | 64.46 |
| BERT \| bert-base-uncased | 1 | 0.4827 (0.0000) | 0.1577 | 0.3365 | 64.54 |
| linear-chain CRF \| crf_seed42_c1-0.1_c2-0.1_iter100.crfsuite | 1 | 0.3103 (0.0000) | 0.1035 | 0.3788 | 209.83 |
| word lookup \| none | 1 | 0.0084 (0.0000) | 0.0000 | 0.1695 | 0.02 |
| constant classifier \| none | 1 | 0.0000 (0.0000) | 0.0000 | 0.0000 | 0.00 |

## Validation results

- **BERT | bert-base-cased** (bert-base-cased-full-seed42-validation): token macro F1 0.5105; strict span micro F1 0.1470; partial span micro F1 0.3193.
- **BERT | bert-base-uncased** (bert-base-uncased-full-seed42-validation): token macro F1 0.5120; strict span micro F1 0.1630; partial span micro F1 0.3351.
- **linear-chain CRF | crf_seed42_c1-0.1_c2-0.1_iter100.crfsuite** (crf-full-seed42-validation): token macro F1 0.3058; strict span micro F1 0.1124; partial span micro F1 0.3702.
- **BERT | nlpaueb/legal-bert-base-uncased** (nlpaueb_legal-bert-base-uncased-full-seed42-validation): token macro F1 0.5387; strict span micro F1 0.2093; partial span micro F1 0.4196.
- **RoBERTa | roberta-base** (roberta-base-full-seed42-validation): token macro F1 0.5223; strict span micro F1 0.2073; partial span micro F1 0.4036.

## Top and bottom five categories

### Highest F1

- **Governing Law** — F1 0.947; 47 gold test spans; Recall trails precision; the model misses some annotated tokens.
- **Insurance** — F1 0.805; 44 gold test spans; Test support and annotation boundary ambiguity affect this estimate.
- **Covenant Not To Sue** — F1 0.791; 15 gold test spans; Test support and annotation boundary ambiguity affect this estimate.
- **Rofr/Rofo/Rofn** — F1 0.777; 52 gold test spans; Recall trails precision; the model misses some annotated tokens.
- **No-Solicit Of Employees** — F1 0.751; 8 gold test spans; Recall trails precision; the model misses some annotated tokens.

### Lowest F1

- **Source Code Escrow** — F1 0.304; 8 gold test spans; Recall trails precision; the model misses some annotated tokens.
- **Third Party Beneficiary** — F1 0.263; 5 gold test spans; Recall trails precision; the model misses some annotated tokens.
- **Affiliate License-Licensee** — F1 0.191; 17 gold test spans; Recall trails precision; the model misses some annotated tokens.
- **Affiliate License-Licensor** — F1 0.079; 4 gold test spans; Only 4 gold test spans; per-category estimate is unstable.
- **Price Restrictions** — F1 0.000; 4 gold test spans; Only 4 gold test spans; per-category estimate is unstable.

## Three caveats
- 3 categories have fewer than five test spans.
- Outside tokens account for 88.9% of priority-resolved tokens.
- Runs with fewer than three seeds (and the 10-contract CRF smoke run) are exploratory; seed variance and robust model significance remain unestablished.
- Test results are descriptive only; model selection is not performed from test scores.
