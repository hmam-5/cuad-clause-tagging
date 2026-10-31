# Preprocessing and label policy

## Source and input schema

`scripts/download_data.py` downloads the official Atticus Project archive and
extracts its original `CUADv1.json`. The archive contains 510 contract contexts
and 41 SQuAD-style questions per contract. Answers carry character offsets into
the original context; contract text is embedded in the context field rather
than stored as a separate `.txt` file in this GitHub release. The source file
and its offsets are never edited.

## Cleaning and offset preservation

`src/cuad_tagging/pipeline.py` collapses whitespace runs, normalizes CR/LF,
form-feed, and non-breaking spaces, and removes null and non-whitespace control
characters plus invisible Unicode format characters such as soft hyphens that
are discarded by common subword tokenizers. It retains a character-to-source-
offset map for every normalized character. Token records contain both
normalized and original half-open character offsets. Replacement characters
are retained and counted rather than guessed at or silently deleted.

Each answer is checked against the exact substring at its reported offset. An
offset is corrected only if the answer text has one unique exact match within
32 characters. Ambiguous or missing matches are dropped and written to
`data/processed/span_corrections.csv` with the contract, category, reported
offset, action, and reason. A span that cannot be projected onto any token is
also reported.

Tokenization uses the documented Unicode word/punctuation regular expression
`\\w+|[^\\w\\s]`. Token offsets allow downstream subword tokenizers to align
their first subtokens with the same gold spans.

## Overlapping annotations

For multi-label experiments each token stores a 41-element binary vector in
the fixed category order in `config.py`. The `categories` list preserves every
category assigned to a token. The default single-label BIO representation
resolves collisions in favor of the globally rarer category; ties follow the
published category order. `stats.json` reports character-level overlapping
annotation pairs, tokens carrying multiple categories, and single-label
outside-token share. This priority rule is an explicit modeling choice, not a
claim that the source annotations are mutually exclusive.

## Windows and splits

The default word-level windows contain at most 512 tokens with a stride of 128.
Boundaries move to avoid splitting a BIO span when the whole span fits in one
window. Spans longer than a window cannot be kept intact. Every record retains
its contract identifier and global token offsets so predictions can be stitched
back together. Splits are assigned before windows, by contract, using a seeded
greedy multilabel stratifier. The statistics report categories with fewer than
five test spans; those categories need cautious interpretation.

## Outputs and checks

- `contracts.jsonl.xz`: an LZMA-compressed JSONL file with one cleaned contract
  per line. To keep the multi-million
  token output compact, `tokens` is the word list; parallel `offsets` entries
  are `[clean_start, clean_end, raw_start, raw_end]`; `labels` contains BIO
  label IDs; and `categories` contains sparse category-ID lists for the
  multi-label targets. `spans` entries are
  `[category_id, clean_start, clean_end, raw_start, raw_end, token_start,
  token_end, text]`; `windows` entries are `[start_token, end_token]`. Cleaned
  contract text is reconstructible by placing token strings at their recorded
  clean offsets; the unmodified source text remains in the original CUAD file.
  Category and BIO IDs use the fixed order in `src/cuad_tagging/config.py`.
- `stats.json`: counts, span length summaries, overlaps, splits, cleaning
  changes, window counts, and token-span round-trip agreement.
- `span_corrections.csv`: audit trail for every repaired or dropped span.
- `results/analysis/`: multilabel class/span tables, priority-resolved BIO
  counts (including O), contract/window split summaries, overlap pairs, and
  slide-readable PNG/PDF figures; tables are also exported as booktabs LaTeX.
- `source_annotation_audit.json` and CSV tables: empty-answer questions,
  malformed/missing answer metadata, valid non-empty answer count, and counts
  by category from the original CUAD source.

Run `python -m unittest discover -s tests` for offset, BIO, window, split,
metric, and end-to-end fixture tests.
