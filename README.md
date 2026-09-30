# Divar Scanner v0.2 — Fatemi Tehran Rental Risk Lab

A production-minded research pipeline for collecting **public apartment-rental listings around Fatemi, Tehran**, normalizing Persian real-estate data, detecting data-quality errors and local-market anomalies, clustering copy/paste listings, and optionally adding a bounded semantic review from several interchangeable AI providers.

> **Important:** this project generates **review-priority / suspicion signals**, not a legal or factual determination of fraud. A cheap listing, an unusual property, a stale ad, or a parser error can all be legitimate.

## Architecture

```text
Divar public listing/search payloads
              │
              ▼
     polite crawler + disk cache
              │
              ▼
 Persian normalization / phone redaction
              │
              ├───────── deterministic consistency checks
              │
              ├───────── robust peer-group Median/MAD
              │
              ├───────── Isolation Forest
              │
              ├───────── OOF property-price expectation model
              │              (no price→price target leakage)
              │
              ├───────── Local Outlier Factor
              │
              └───────── duplicate similarity graph
                               │
                               ▼
                         cheap prefilter
                               │
                   top suspicious candidates only
                               │
                               ▼
           optional typed/structured semantic backend
        Jev / Groq / Cerebras / Gemini / Cohere Aya /
          OpenRouter / Hugging Face / custom OpenAI API
                               │
                               ▼
          confidence-aware multi-view evidence fusion
                               │
                  ┌────────────┼────────────┐
                  ▼            ▼            ▼
               normal        review        high
```

## The Colab import bug that v0.2 fixes

Do **not** clone this repository to `/content/divar_scanner` inside Colab.

The Python package is also named `divar_scanner`. Colab can keep `/content` on `sys.path`, and a clone directory with the same name can be discovered as a PEP-420 namespace package before the editable `src/divar_scanner` package. The symptom is exactly:

```text
ModuleNotFoundError: No module named 'divar_scanner.pipeline'
```

The notebook now clones to:

```text
/content/divar_scanner_repo
```

and then explicitly places:

```text
/content/divar_scanner_repo/src
```

at the front of `sys.path`, clears stale `divar_scanner.*` modules, and asserts the imported package origin before doing any work.

## Colab — recommended

Open:

```text
colab/Divar_Fatemi_Anomaly_Scanner.ipynb
```

The first cells:

1. clone into the non-conflicting `divar_scanner_repo` directory,
2. install the package with the current kernel's Python,
3. print pandas / NumPy / scikit-learn / PyArrow versions,
4. assert the package origin,
5. run `python -m divar_scanner.doctor --network`,
6. then start the actual crawl.

For a first run use `MAX_LISTINGS=100` or `200`.

## Zero-payment / zero-key local mode

If you cannot or do not want to use a paid API, use local inference in Colab:

```text
SEMANTIC_PROVIDER = local-qwen-small   # Qwen3-1.7B, lightest
SEMANTIC_PROVIDER = local-qwen         # Qwen3-4B, recommended
SEMANTIC_PROVIDER = local-aya          # Aya Expanse 8B, explicitly Persian-capable
```

No API key, billing account, PayPal, or card is used. The model weights are loaded into the notebook runtime and inference happens there.

For `local-qwen` and `local-aya`, select **Runtime → Change runtime type → GPU** in Colab. The notebook installs the optional local stack only after a local provider is selected:

```bash
python -m pip install -e '.[local]'
```

The local stack uses 4-bit NF4 quantization on CUDA. If no GPU is available, `local-qwen-small` has a CPU fallback, but it is much slower.

Public model defaults:

- `Qwen/Qwen3-1.7B`
- `Qwen/Qwen3-4B`
- `CohereLabs/aya-expanse-8b`

You can also point to model files you already have locally:

```bash
export SEMANTIC_PROVIDER=local-qwen
export LOCAL_MODEL_ID=/content/my_model_folder
```

This avoids any inference-provider dependency entirely.

## Semantic provider support

External semantic inference is optional. The statistical/graph pipeline runs without any API key.

Set one provider in the Colab dropdown, or export the corresponding environment variable:

| Provider | Env var | Default model / route | Notes |
|---|---|---|---|
| TypeSafe Jev hosted | `JEV_API_KEY` | `jev-1.13.0` | Native typed choice/score/noul |
| TypeSafe official | `TYPESAFE_API_KEY` | `jev-1.13.0` | Native typed decision endpoint |
| Groq | `GROQ_API_KEY` | `openai/gpt-oss-20b` | OpenAI-compatible, structured JSON |
| Cerebras | `CEREBRAS_API_KEY` | `gpt-oss-120b` | OpenAI-compatible |
| Google Gemini | `GEMINI_API_KEY` | `gemini-3.8-flash` | OpenAI compatibility endpoint |
| Cohere / Aya | `COHERE_API_KEY` | `c4ai-aya-expanse-32b` | Trial keys are useful for prototyping |
| OpenRouter | `OPENROUTER_API_KEY` | `openrouter/free` | Free-model router for low-volume prototypes |
| Hugging Face | `HF_TOKEN` | `openai/gpt-oss-120b:fastest` | Uses Inference Providers credits |
| Custom | `OPENAI_COMPAT_API_KEY` + base/model vars | your model | Any compatible Chat Completions endpoint |

For a custom endpoint:

```bash
export OPENAI_COMPAT_API_KEY='...'
export OPENAI_COMPAT_BASE_URL='https://your-provider.example/v1'
export OPENAI_COMPAT_MODEL='your-model'
export SEMANTIC_PROVIDER='custom-openai-compatible'
```

The compatibility layer first tries strict JSON Schema, then JSON-object mode, then prompt-constrained JSON. Provider failures do not kill the pipeline; the row keeps the deterministic fallback semantic score and the run metadata records the error.

### Provider ensemble

By default:

```yaml
semantic:
  ensemble_max_providers: 1
```

Set it to `2` or more only when you intentionally want multi-provider consensus. Probabilities are averaged and the highest-confidence provider supplies the class label.

## Why the detection is no longer a shallow "outlier = fraud" model

### 1. Data-quality evidence

Cross-field checks detect examples like:

```text
structured: 38 m² / 1 room / no parking
description: 180 m² / 3 rooms / parking
```

This is primarily a **data-error / inconsistency signal**, not automatic fraud.

### 2. Robust local peer statistics

Price views:

```text
deposit_per_m2
rent_per_m2
equivalent_deposit_per_m2
```

are compared hierarchically against:

1. neighborhood + area bucket + rooms,
2. neighborhood + area bucket,
3. neighborhood,
4. global fallback.

Median/MAD is used instead of mean/std because housing prices have heavy tails.

### 3. Isolation Forest

A multivariate Isolation Forest checks unusual combinations of area, room count, age, price views, and text length.

### 4. OOF price expectation model

For enough listings (roughly 40+), the pipeline trains a cross-validated `HistGradientBoostingRegressor` on **property attributes and neighborhood**, not on deposit/rent themselves.

Each listing is predicted by a fold that did not train on that listing:

```text
property attributes + neighborhood
                │
                ▼
       expected equivalent deposit
                │
                ▼
       signed log price residual
                │
                ▼
       price_model_anomaly_score
```

This avoids the common leakage mistake of predicting a price target using variables that algebraically define that target.

### 5. Local Outlier Factor

LOF adds a density view: a point may be far less plausible when its *combination* of characteristics lives in a sparse region even when each field individually looks reasonable.

### 6. Duplicate graph, not just duplicate pairs

Persian-friendly character 3–5-gram TF-IDF creates similarity edges between listings. Connected components become duplicate clusters.

The pipeline records:

```text
duplicate_cluster_size
duplicate_cluster_neighborhoods
duplicate_cluster_price_span
duplicate_bait_score
```

A repeated template is not inherently suspicious. A repeated template that appears with conflicting neighborhoods and large price changes is stronger review evidence.

### 7. Semantic review on a shortlist

The semantic model sees deterministic evidence such as:

- field contradictions,
- local market score,
- OOF price ratio,
- duplicate-cluster signals,
- title and description.

It must choose among:

```text
plausible
data_error
market_outlier
misleading_or_bait
needs_review
```

and returns suspicion/review/consistency/confidence values.

### 8. Confidence-aware fusion + uncertainty

A low-confidence heuristic or LLM opinion is down-weighted. The final system also calculates detector disagreement:

```text
uncertainty_score
```

so cases where market, duplicate, semantic and data-quality channels disagree can be sent to humans rather than auto-labeled.

## Fatemi defaults

`config/fatemi.yaml` uses:

- Tehran city ID `1`
- category `apartment-rent`
- aliases `فاطمی`, `میدان فاطمی`, `دکتر فاطمی`
- runtime district-ID resolution
- conservative request delay + jitter
- response cache
- phone-like string redaction

If Fatemi cannot be resolved to a Divar district, the crawler **refuses to silently fall back to all of Tehran**.

## Local usage

```bash
git clone https://github.com/prestigegitserp/divar_scanner.git divar_scanner_repo
cd divar_scanner_repo
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

divar-scanner-doctor
pytest
divar-scanner run --config config/fatemi.yaml --max-listings 150 --no-jev
```

The legacy `--no-jev` flag disables all external semantic providers; the name is retained for CLI compatibility.

To analyze an already normalized dataset:

```bash
divar-scanner analyze --config config/fatemi.yaml --input listings.parquet
```

## Outputs

Runs write timestamped artifacts to `outputs/fatemi/`:

```text
listings_scored_*.csv
listings_scored_*.parquet
review_queue_*.csv
report_*.html
run_meta_*.json
```

Important columns include:

| Column | Meaning |
|---|---|
| `data_quality_score` | impossible/missing/cross-field mismatch evidence |
| `robust_market_anomaly` | robust local-peer price deviation |
| `isolation_anomaly_score` | Isolation Forest view |
| `price_model_anomaly_score` | OOF property-price residual |
| `price_model_ratio` | actual / OOF expected equivalent deposit |
| `lof_anomaly_score` | local-density anomaly |
| `duplicate_bait_score` | graph-cluster inconsistency |
| `semantic_score` | semantic misleading/bait probability |
| `semantic_confidence` | semantic backend confidence |
| `uncertainty_score` | disagreement across detector families |
| `suspicion_score` | fused evidence score |
| `review_priority_score` | suspicion + uncertainty + review probability |
| `risk_band` | normal / review / high |
| `flag_reasons` | human-readable evidence |

## Diagnostics

```bash
python -m divar_scanner.doctor
python -m divar_scanner.doctor --network
```

The doctor prints the exact imported package path and dependency versions. It exits non-zero if the package is being imported from a wrong path.

## CI

GitHub Actions now tests:

- Python 3.10
- Python 3.11
- Python 3.12
- editable packaging/import
- full pytest suite
- bytecode compilation
- simulated Colab layout under `/tmp/content/divar_scanner_repo`
- end-to-end synthetic dataset analysis including OOF/LOF/duplicate/semantic-fallback layers

## Responsible crawling

The crawler uses public listing/search responses, conservative retry/delay behavior, caching, and stops on 401/403/429 rather than attempting to bypass access controls. It does not use OTP/login flows or contact-info endpoints.

Divar's public web response schema is not a stable contract. If schema or access policy changes, update the adapter rather than increasing request aggression.

## Next high-value upgrades

The biggest remaining improvements require **time and labels**, not another unsupervised trick:

- scheduled snapshots to detect reposting and suspicious price changes,
- advertiser/entity graph if a legitimate public identifier is available,
- image perceptual hashes for reused property photos,
- geospatial peer groups if coordinates become available,
- manual-review labels + LightGBM/XGBoost meta-ranker,
- probability calibration (isotonic / Platt) on held-out human labels,
- conformal review thresholds,
- active learning for uncertain cases,
- drift monitoring by month and neighborhood.

## License

MIT. Verify applicable site terms and local law before large-scale collection or operational deployment.
