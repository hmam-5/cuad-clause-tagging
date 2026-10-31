# CUAD project: findings summary

## Three key numbers
- **510 contracts** and **13,823 validated spans**.
- **100%** token-to-span round-trip agreement.
- Best full-data run so far: **word lookup | none**, token macro F1 **0.0084** (n=1 seed(s)).

## Full-data model results

| Model | Seeds | Token macro F1 mean (SD) | Strict span micro F1 | Partial span micro F1 | Mean train minutes |
|---|---:|---:|---:|---:|---:|
| word lookup \| none | 1 | 0.0084 (0.0000) | 0.0000 | 0.1695 | 0.02 |
| constant classifier \| none | 1 | 0.0000 (0.0000) | 0.0000 | 0.0000 | 0.00 |

## Top and bottom five categories

### Highest F1

- **Governing Law** — F1 0.061; 47 gold test spans; Recall trails precision; the model misses some annotated tokens.
- **Insurance** — F1 0.043; 44 gold test spans; Recall trails precision; the model misses some annotated tokens.
- **Anti-Assignment** — F1 0.040; 68 gold test spans; Recall trails precision; the model misses some annotated tokens.
- **Cap On Liability** — F1 0.034; 73 gold test spans; Recall trails precision; the model misses some annotated tokens.
- **Irrevocable Or Perpetual License** — F1 0.026; 21 gold test spans; Recall trails precision; the model misses some annotated tokens.

### Lowest F1

- **Termination For Convenience** — F1 0.000; 21 gold test spans; No correctly labeled test tokens; class imbalance and overlap are likely factors.
- **Third Party Beneficiary** — F1 0.000; 5 gold test spans; No correctly labeled test tokens; class imbalance and overlap are likely factors.
- **Uncapped Liability** — F1 0.000; 18 gold test spans; No correctly labeled test tokens; class imbalance and overlap are likely factors.
- **Volume Restriction** — F1 0.000; 20 gold test spans; No correctly labeled test tokens; class imbalance and overlap are likely factors.
- **Warranty Duration** — F1 0.000; 21 gold test spans; No correctly labeled test tokens; class imbalance and overlap are likely factors.

## Three caveats
- 3 categories have fewer than five test spans.
- Outside tokens account for 88.9% of priority-resolved tokens.
- Runs with fewer than three seeds (and the 10-contract CRF smoke run) are exploratory; seed variance and robust model significance remain unestablished.
