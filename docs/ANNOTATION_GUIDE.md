# Human annotation guide — Fatemi rental review v0.4

The scanner is a triage/research system. Annotators label observable evidence, not legal fraud.

## 1. Primary disposition — human_disposition

Choose one:

- `plausible`: coherent listing; no material unresolved issue.
- `data_error`: main issue is extraction/parser/structured-data corruption.
- `market_outlier`: coherent listing but statistically unusual versus comparable properties.
- `misleading_or_bait`: material evidence beyond price unusualness supports misleading/bait-like representation.
- `ambiguous_mixed`: competing explanations remain genuinely unresolved.

Do not label `misleading_or_bait` merely because rent/deposit is cheap, expensive, or statistically extreme.

## 2. Integrity class — human_integrity_class

Choose one:

- `consistent`: text and structured fields materially agree.
- `extraction_error`: mismatch is best explained by crawler/parser/extraction failure.
- `listing_claim_conflict`: listing claims themselves materially conflict with structured representation.
- `insufficient_evidence`: not enough evidence to distinguish the above.

## 3. Duplicate pattern — human_duplicate_pattern

Choose one:

- `no_duplicate_evidence`: similarity is insufficient.
- `normal_template_reuse`: shared boilerplate/template without material property conflict.
- `likely_same_property_repost`: likely same property/listing reposted with limited edits.
- `cross_property_conflict`: highly similar copy attached to materially different neighborhood/price/property facts.

Template reuse by agents is not automatically bait.

## 4. Market status — human_market_status

Choose one:

- `typical`: consistent with peer market.
- `moderate_outlier`: meaningful but not extreme deviation.
- `extreme_outlier`: multiple independent market signals support a strong deviation.
- `insufficient_context`: peer/context evidence is too weak.

This is a market-position label, not a deception label.

## 5. Independent yes/no labels

### human_bait

- yes: meaningful misleading/bait evidence beyond price anomaly.
- no: insufficient evidence.

### human_data_error

- yes: parser/extraction/data corruption is meaningfully supported.
- no: insufficient evidence.

### human_manual_review

- yes: human inspection is warranted before trusting the row.
- no: routine use is reasonable on available evidence.

## 6. Consistency — human_consistency_level

- 0: major multiple contradictions
- 1: several important inconsistencies
- 2: ambiguous/limited mismatch
- 3: mostly consistent
- 4: strongly consistent

## 7. Confidence and notes

`human_label_confidence` should be between 0.5 and 1.0. Use `human_notes` to record the evidence that drove difficult labels.

## Annotation protocol

1. Review title/body and structured fields before model risk bands when possible.
2. Treat statistical anomaly values as evidence, never ground truth.
3. Keep market-outlier and misleading/bait judgments separate.
4. Distinguish parser error from genuine contradiction in the listing.
5. For duplicate clusters, compare at least two cluster members before using `cross_property_conflict`.
6. Use `ambiguous_mixed` / `insufficient_evidence` instead of forcing certainty.
7. Double-annotate a subset and calculate agreement.
8. Freeze an untouched held-out evaluation split before tuning calibration/thresholds.

## Recommended data split

After enough labels, create splits by listing/duplicate cluster so near-duplicates cannot leak across train and evaluation:

- train/domain-adaptation split;
- calibration split for question temperatures;
- held-out evaluation split.

Do not random-split individual rows when near-duplicate cluster members can appear on both sides.

## Evaluation

After labelling:

    divar-scanner evaluate-decisions --input labelled.csv --output outputs/decision_evaluation.json

Report probability quality (NLL, Brier, ECE), coverage after abstention, and selective accuracy — not accuracy alone.