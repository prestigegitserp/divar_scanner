# Decision architecture: Jev-style, non-generative

This project deliberately separates **generation** from **decision**. The default decision layer does not ask a language model to write JSON and does not parse generated tokens.

## Observable Jev contract we preserve

TypeSafe Jev publicly exposes three bounded primitives:

- `choice`: one option from a caller-supplied set + option probabilities
- `score`: an ordered level distribution + fractional score
- `noul`: a yes/no probability

The exact proprietary Jev weights/training internals are not public. Therefore this project reproduces the **decision contract**, not TypeSafe's undisclosed RLCD training.

## Default open model

`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`

This is an encoder-style sequence-classification / NLI model, not a text generator. It is based on multilingual DeBERTa and is used only to classify premise/hypothesis pairs into entailment, neutral or contradiction.

For every user-supplied decision option we create one hypothesis. The model evaluates all options, then code converts those bounded NLI outputs into a probability distribution.

### Candidate evidence

For option `i`:

```text
e_i = log P(entailment_i) - log P(contradiction_i)
```

Then:

```text
P(option_i) = softmax(e_i / T)
```

`T` is a temperature that should eventually be fit on labelled Persian examples.

No candidate identifier is generated. If an option is not provided, it cannot be returned.

## Mapping to Jev primitives

### Choice

One premise + N candidate hypotheses.

```text
state + option_1 description -> NLI evidence e1
state + option_2 description -> NLI evidence e2
...
softmax(e1..eN) -> probabilities
argmax -> choice
```

### Noul

Exactly two hypotheses: `no` and `yes`.

```text
P(yes) = softmax(e_no, e_yes)[yes]
```

### Score

Each ordered level is a bounded candidate. After getting probabilities `p_k`:

```text
score = sum(k * p_k)
normalized_score = score / (K - 1)
```

## Confidence

TypeSafe does not disclose the exact proprietary confidence formula. This project therefore labels its confidence honestly as a derived diagnostic:

1. concentration of the option distribution via normalized entropy
2. penalized by weighted NLI neutral mass

This is useful for routing, but is **not** a guarantee of correctness.

## Why this is preferable to an LLM for this project

- no free-form text generation
- no JSON parsing failure mode
- output space is mathematically bounded
- option order is explicit
- probabilities come from classifier logits
- dynamic labels/descriptions are supported without retraining a fixed K-class head
- a ~280M multilingual encoder is practical on Colab CPU/GPU

## Why it is not identical to proprietary Jev

Jev's current public documentation describes RLCD-calibrated typed decisions, but its internal architecture and weights are not public. mDeBERTa NLI was trained for natural-language inference, not specifically for Iranian housing fraud/anomaly decisions.

Therefore the next research milestone should be **Persian calibration / domain adaptation**, not pretending zero-shot probabilities are already calibrated.

## Optional comparison models

### AlexWortega/openjev

This is a genuine open Jev-style sequence-classification project: a cross-encoder produces entailment/contradiction/neutral and exposes typed decisions without generation. It is very relevant for benchmarking Jev-style mechanics. However its current checkpoints use a Qwen3.5-derived backbone, so it is not the default here because this project intentionally wants a non-LLM encoder backbone.

### GLiClass

GLiClass is an efficient zero-shot sequence classifier supporting arbitrary label descriptions. It is a strong future comparison backend, especially for one-pass multi-label classification. The default project stays on NLI first because the probability semantics of entailment/contradiction map cleanly to Jev-style choice/noul/score primitives and are easier to audit.
