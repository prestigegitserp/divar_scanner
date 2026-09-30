# Non-generative decision model options

All models below are used as **sequence classifiers**, not text generators.

## Default — mDeBERTa multilingual NLI

Backend: `mdeberta-nli`  
Model: `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`  
License: MIT

Why it is the default:

- encoder-style NLI / sequence classification;
- Persian (`fa`) is explicitly included in the multilingual NLI training set;
- entailment / neutral / contradiction map naturally to bounded Jev-style options;
- compact enough for Colab;
- permissive license.

## Persian specialist — ParsBERT ParsiNLU

Backend: `parsbert-parsinlu`  
Model: `persiannlp/parsbert-base-parsinlu-entailment`  
License: CC-BY-NC-SA-4.0

The model card explicitly describes Persian textual entailment and gives the label order:

```text
entails, contradicts, neutral
```

The project registry therefore supplies an explicit index map instead of guessing from opaque `LABEL_0` metadata.

Use this backend for non-commercial/public-interest evaluation when the model license is compatible with your deployment.

## Alternative Persian model — multilingual BERT ParsiNLU

Backend: `mbert-parsinlu`  
Model: `persiannlp/mbert-base-parsinlu-entailment`  
License: CC-BY-NC-SA-4.0

Useful as another Persian-domain NLI reference.

## Evidence ensemble

Backend: `persian-ensemble`

Default members:

```text
55% mdeberta-nli
45% parsbert-parsinlu
```

The ensemble does **not** average already-normalized choice probabilities. For candidate `i`, each model first contributes:

```text
e_(m,i) = log P_m(entail_i) - log P_m(contradict_i)
```

Then the engine fuses evidence:

```text
e_i = sum_m w_m * e_(m,i)
P(option_i) = softmax(e_i / T)
```

This preserves one closed-set normalization step over the caller-supplied options.

The weighted standard deviation of model log-odds becomes a disagreement signal and reduces decision confidence.

## Why OpenJev is not the default

`AlexWortega/openjev` is directly relevant: it exposes a Jev-like typed-decision pattern using an NLI cross-encoder and no generation at inference. It is an excellent architectural reference and benchmark.

Its current checkpoints, however, use Qwen3.5-derived sequence-classification backbones. This project intentionally uses encoder NLI backbones as the default because the design requirement is **non-LLM as well as non-generative**.

## Calibration

Neither the mDeBERTa nor ParsiNLU model was trained specifically to output calibrated probabilities for Tehran rental-risk decisions.

Therefore:

- raw option softmax = bounded relative decision score;
- question-specific temperature scaling = calibration layer;
- real reliability claims require held-out human labels from the target domain.
