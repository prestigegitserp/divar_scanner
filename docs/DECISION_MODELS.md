# Decision model options — v0.4

## Primary — Laya Multilingual

Backend: `laya-multilingual`
Model: `convaiinnovations/laya`, subfolder `multilingual`
License: Apache-2.0

Why primary:

- native non-autoregressive typed-decision model;
- multilingual encoder rather than autoregressive decoder;
- dynamic caller-supplied options;
- one-pass bounded probability output;
- practical on Colab;
- closest open fit to the observable Jev/System-One contract.

The project wraps Laya with opaque-choice transport, option-permutation pooling, target-domain temperature calibration, cross-question coherence checks, and abstention.

## Baseline — mDeBERTa multilingual NLI

Backend: `mdeberta-nli`
Model: `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`
License: MIT

This is a strong non-generative multilingual NLI baseline, including Persian in its multilingual training. It is not a native System-One decision head; the repo constructs bounded options from entailment/contradiction evidence.

## Persian specialist — ParsBERT ParsiNLU

Backend: `parsbert-parsinlu`
Model: `persiannlp/parsbert-base-parsinlu-entailment`
License: CC-BY-NC-SA-4.0

Persian textual-entailment baseline. Useful for scientific comparison, not the default System-One runtime.

## Persian evidence ensemble

Backend: `persian-ensemble`

Fuses candidate evidence from mDeBERTa and ParsBERT before one closed-set softmax. It provides a disagreement signal but remains an NLI baseline.

## OpenJev

Architecturally very relevant: typed non-generative decisions and closed probabilities. Current public checkpoints use a Qwen-derived backbone and are much larger, so they are excluded as the primary runtime under this project's stricter no-autoregressive-LLM-backbone requirement.

## Lev

Relevant System-One research, but the public model is built on a Qwen decoder backbone. Not selected for the same reason.

## OpenDecider

Compact and promising, but current public evaluation coverage is much more English-centric than the Persian housing target. It remains a future comparison candidate.

## Probability semantics

Even Laya's native probability should not be treated as calibrated Tehran-housing truth out of the box. v0.4 records raw pooled bounded probabilities, option-order stability and coherence, then supports question-specific target-domain temperature scaling.

Final reliability claims require held-out human labels.