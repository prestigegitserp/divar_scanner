# Human annotation guide — Fatemi rental review

The model is a triage system. Annotators should label **observable evidence**, not make legal accusations of fraud.

## Primary disposition

Choose exactly one value for `human_disposition`:

### `plausible`

Use when the listing is internally coherent and there is no material evidence of a data problem, misleading representation, or unexplained market anomaly.

### `data_error`

Use when the main issue is most plausibly extraction / parsing / structured-data corruption or a field mismatch.

Examples:

- structured area says 65 m² while the body clearly says 140 m²;
- structured one-bedroom while title/body consistently describe three bedrooms;
- crawler/parser dropped or misread a numeric field.

Do **not** use this merely because the property is unusual.

### `market_outlier`

Use when the listing is coherent but its price or property-feature combination is genuinely unusual versus comparable listings.

A low or high price **alone is not evidence of fraud or bait**.

### `misleading_or_bait`

Use only when there is material evidence beyond ordinary price unusualness, such as:

- mutually inconsistent claims that appear promotional rather than a simple parser error;
- near-identical ad copy repeatedly attached to materially different properties/areas/neighborhoods;
- content that misrepresents important property attributes;
- repeated listing patterns whose contradictions cannot reasonably be explained by extraction error.

This is still a review label, not a legal finding.

### `ambiguous_mixed`

Use when two or more explanations are genuinely plausible and no single primary disposition dominates.

## Independent yes/no labels

These labels are intentionally separate from the primary disposition.

### `human_bait`

- `yes`: meaningful evidence of misleading/bait-like representation beyond a simple price outlier.
- `no`: evidence is insufficient.

### `human_data_error`

- `yes`: parser/extraction/structured-data error is meaningfully supported.
- `no`: no such evidence.

### `human_manual_review`

- `yes`: a human should inspect this listing before trusting it.
- `no`: routine use is reasonable on available evidence.

## Consistency score

Set `human_consistency_level` to one integer:

- 0 — major, multiple contradictions
- 1 — several important inconsistencies
- 2 — ambiguous / limited mismatch
- 3 — mostly consistent
- 4 — strongly consistent

## Annotation protocol

1. Review the raw title/body and structured fields before looking at the model's final risk band when possible.
2. Treat statistical anomaly scores as supporting evidence, never as ground truth.
3. Do not infer intent from price alone.
4. When uncertain, use `ambiguous_mixed` and explain why in `human_notes`.
5. For a calibration/evaluation study, keep a held-out set that was **not** used to tune temperatures or thresholds.
6. Prefer two independent annotators for a subset and measure agreement. Resolve recurring disagreements by refining this guide rather than silently forcing consensus.

## Calibration split

A practical starting point after enough reviewed rows:

- calibration set: fit question-specific temperatures;
- evaluation set: measure NLL/Brier score, disposition accuracy, review precision/recall and reliability;
- do not report calibration-set metrics as final model performance.

The project generates the required blank human-label columns automatically in `review_queue_*.csv`.
