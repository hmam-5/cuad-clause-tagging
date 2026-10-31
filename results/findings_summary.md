# CUAD project: findings summary

## Three key numbers
- **510 contracts** and **13,823 validated spans**.
- **100%** token-to-span round-trip agreement.
- Highest validation-selected group: **linear-chain CRF | crf_seed42_c1-0.1_c2-0.1_iter100.crfsuite** (validation token macro F1 **0.3058**).

## Full-data test results (descriptive; not used for selection)

| Model | Seeds | Token macro F1 mean (SD) | Strict span micro F1 | Partial span micro F1 | Mean train minutes |
|---|---:|---:|---:|---:|---:|
| linear-chain CRF \| crf_seed42_c1-0.1_c2-0.1_iter100.crfsuite | 1 | 0.3103 (0.0000) | 0.1035 | 0.3788 | 209.83 |
| word lookup \| none | 1 | 0.0084 (0.0000) | 0.0000 | 0.1695 | 0.02 |
| constant classifier \| none | 1 | 0.0000 (0.0000) | 0.0000 | 0.0000 | 0.00 |

## Validation results

- **linear-chain CRF | crf_seed42_c1-0.1_c2-0.1_iter100.crfsuite** (crf-full-seed42-validation): token macro F1 0.3058; strict span micro F1 0.1124; partial span micro F1 0.3702.

## Top and bottom five categories

### Highest F1

- **Governing Law** — F1 0.825; 47 gold test spans; Recall trails precision; the model misses some annotated tokens.
- **Rofr/Rofo/Rofn** — F1 0.748; 52 gold test spans; Recall trails precision; the model misses some annotated tokens.
- **Uncapped Liability** — F1 0.724; 18 gold test spans; Test support and annotation boundary ambiguity affect this estimate.
- **Insurance** — F1 0.719; 44 gold test spans; Recall trails precision; the model misses some annotated tokens.
- **Notice Period To Terminate Renewal** — F1 0.693; 16 gold test spans; Test support and annotation boundary ambiguity affect this estimate.

### Lowest F1

- **Most Favored Nation** — F1 0.000; 4 gold test spans; Only 4 gold test spans; per-category estimate is unstable.
- **No-Solicit Of Customers** — F1 0.000; 7 gold test spans; No correctly labeled test tokens; class imbalance and overlap are likely factors.
- **Price Restrictions** — F1 0.000; 4 gold test spans; Only 4 gold test spans; per-category estimate is unstable.
- **Source Code Escrow** — F1 0.000; 8 gold test spans; No correctly labeled test tokens; class imbalance and overlap are likely factors.
- **Third Party Beneficiary** — F1 0.000; 5 gold test spans; No correctly labeled test tokens; class imbalance and overlap are likely factors.

## Three caveats
- 3 categories have fewer than five test spans.
- Outside tokens account for 88.9% of priority-resolved tokens.
- Runs with fewer than three seeds (and the 10-contract CRF smoke run) are exploratory; seed variance and robust model significance remain unestablished.
- Test results are descriptive only; model selection is not performed from test scores.
