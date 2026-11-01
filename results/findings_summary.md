# CUAD project: findings summary

## Three key numbers
- **510 contracts** and **13,823 validated spans**.
- **100%** token-to-span round-trip agreement.
- Highest validation-selected group: **BERT | nlpaueb/legal-bert-base-uncased** (validation token macro F1 **0.5631**).

## Full-data test results (descriptive; not used for selection)

| Model | Seed | Token macro F1 (contract-bootstrap 95% CI) | Strict span micro F1 | Partial span micro F1 | Train minutes |
|---|---:|---:|---:|---:|---:|
| DeBERTa \| microsoft/deberta-v3-base | 42 | 0.5402 (0.4798–0.5646) | 0.2166 | 0.4082 | 140.00 |
| RoBERTa \| roberta-base | 42 | 0.5202 (0.4651–0.5433) | 0.2018 | 0.3985 | 65.56 |
| BERT \| nlpaueb/legal-bert-base-uncased | 42 | 0.5125 (0.4546–0.5387) | 0.2044 | 0.4079 | 64.53 |
|  | 43 | 0.5118 (0.4597–0.5359) | 0.2072 | 0.3893 | 64.82 |
| BERT \| bert-base-cased | 42 | 0.5076 (0.4489–0.5314) | 0.1517 | 0.3306 | 64.46 |
|  | 43 | 0.4873 (0.4379–0.5051) | 0.1579 | 0.3189 | 64.81 |
| BERT \| bert-base-uncased | 42 | 0.4827 (0.4287–0.5073) | 0.1577 | 0.3365 | 64.54 |
|  | 43 | 0.4873 (0.4352–0.5128) | 0.1528 | 0.3194 | 64.72 |
| BiLSTM-CRF \| random embeddings (100d) | 42 | 0.3351 (0.2968–0.3517) | 0.1508 | 0.4236 | 54.30 |
| linear-chain CRF \| crf_seed42_c1-0.1_c2-0.1_iter100.crfsuite | 42 | 0.3103 (0.2615–0.3333) | 0.1035 | 0.3788 | 209.83 |
| word lookup \| none | 42 | 0.0084 (0.0067–0.0096) | 0.0000 | 0.1695 | 0.02 |
| constant classifier \| none | 42 | 0.0000 (0.0000–0.0000) | 0.0000 | 0.0000 | 0.00 |

## Validation results

- **BERT | bert-base-cased** (bert-base-cased-full-seed42-validation): token macro F1 0.5105; strict span micro F1 0.1470; partial span micro F1 0.3193.
- **BERT | bert-base-cased** (bert-base-cased-full-seed43-validation): token macro F1 0.5050; strict span micro F1 0.1547; partial span micro F1 0.3292.
- **BERT | bert-base-uncased** (bert-base-uncased-full-seed42-validation): token macro F1 0.5120; strict span micro F1 0.1630; partial span micro F1 0.3351.
- **BERT | bert-base-uncased** (bert-base-uncased-full-seed43-validation): token macro F1 0.5318; strict span micro F1 0.1547; partial span micro F1 0.3399.
- **linear-chain CRF | crf_seed42_c1-0.1_c2-0.1_iter100.crfsuite** (crf-full-seed42-validation): token macro F1 0.3058; strict span micro F1 0.1124; partial span micro F1 0.3702.
- **DeBERTa | microsoft/deberta-v3-base** (microsoft_deberta-v3-base-full-seed42-validation): token macro F1 0.5328; strict span micro F1 0.2056; partial span micro F1 0.3890.
- **BERT | nlpaueb/legal-bert-base-uncased** (nlpaueb_legal-bert-base-uncased-full-seed42-validation): token macro F1 0.5387; strict span micro F1 0.2093; partial span micro F1 0.4196.
- **BERT | nlpaueb/legal-bert-base-uncased** (nlpaueb_legal-bert-base-uncased-full-seed43-validation): token macro F1 0.5631; strict span micro F1 0.2201; partial span micro F1 0.4116.
- **RoBERTa | roberta-base** (roberta-base-full-seed42-validation): token macro F1 0.5223; strict span micro F1 0.2073; partial span micro F1 0.4036.

## Top and bottom five categories

### Highest F1

- **Governing Law** — F1 0.957; 47 gold test spans; Recall trails precision; the model misses some annotated tokens.
- **Covenant Not To Sue** — F1 0.881; 15 gold test spans; Recall trails precision; the model misses some annotated tokens.
- **Insurance** — F1 0.797; 44 gold test spans; Test support and annotation boundary ambiguity affect this estimate.
- **No-Solicit Of Employees** — F1 0.792; 8 gold test spans; Test support and annotation boundary ambiguity affect this estimate.
- **Anti-Assignment** — F1 0.734; 68 gold test spans; Test support and annotation boundary ambiguity affect this estimate.

### Lowest F1

- **Affiliate License-Licensee** — F1 0.280; 17 gold test spans; Recall trails precision; the model misses some annotated tokens.
- **Competitive Restriction Exception** — F1 0.269; 18 gold test spans; Recall trails precision; the model misses some annotated tokens.
- **Most Favored Nation** — F1 0.264; 4 gold test spans; Only 4 gold test spans; per-category estimate is unstable.
- **Affiliate License-Licensor** — F1 0.035; 4 gold test spans; Only 4 gold test spans; per-category estimate is unstable.
- **Price Restrictions** — F1 0.000; 4 gold test spans; Only 4 gold test spans; per-category estimate is unstable.

## Three caveats
- 3 categories have fewer than five test spans.
- Outside tokens account for 88.9% of priority-resolved tokens.
- Runs with fewer than three seeds (and the 10-contract CRF smoke run) are exploratory; seed variance and robust model significance remain unestablished.
- Test results are descriptive only; model selection is not performed from test scores.
