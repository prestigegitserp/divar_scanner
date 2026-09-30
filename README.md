# Divar Scanner — Tehran Fatemi Rental Anomaly Lab

A production-minded research pipeline for collecting **public apartment-rental listings in Tehran / Fatemi** from Divar, normalizing Persian real-estate data, detecting data errors and local-market outliers, finding near-duplicate ads, and optionally using **TypeSafe Jev** for a bounded semantic review pass.

> **Important:** the system produces a **suspicion / review score**, not a legal or factual determination of fraud. A cheap listing, an unusual property, or a parser mismatch can all be legitimate. High-scoring rows are a queue for human verification.

## Why this architecture

`Fraud != anomaly` and `anomaly != fraud`. The pipeline intentionally keeps the evidence channels separate:

1. **Data quality** — impossible values and structured-vs-text contradictions.
2. **Market anomaly** — robust peer-group deviations + Isolation Forest.
3. **Duplicate/bait signal** — character n-gram similarity between ads.
4. **Semantic review** — optional Jev `choice` + `noul` + `score` decisions only on the prefiltered shortlist.
5. **Final triage** — weighted score with explainable `flag_reasons` and `normal / review / high` bands.

```text
Divar public listing/search responses
              │
              ▼
     polite crawler + cache
              │
              ▼
 Persian normalization / PII redaction
              │
      ┌───────┼───────────┐
      ▼       ▼           ▼
 sanity   local peer   near-duplicate
 checks   statistics     detector
      └───────┼───────────┘
              ▼
       cheap prefilter
              │
              ▼  top-K only
        Jev semantic pass
              │
              ▼
  explainable suspicion score
              │
      CSV / Parquet / HTML
```

## Fatemi-specific defaults

`config/fatemi.yaml` ships with:

- city: Tehran (`city_id=1`)
- category: `apartment-rent`
- district query/aliases: `فاطمی`, `میدان فاطمی`, `دکتر فاطمی`
- runtime district-ID resolution from Divar's city/district data
- maximum 500 listings by default
- 1.2 s request delay + jitter
- phone-number redaction in description text
- equivalent-deposit conversion factor of `30x` monthly rent (configurable assumption)

The crawler **refuses to fall back to all of Tehran** if it cannot resolve Fatemi to a Divar district ID. This prevents accidental broad crawling.

## Colab — recommended start

Open `colab/Divar_Fatemi_Anomaly_Scanner.ipynb` in Google Colab. It:

1. clones this repository,
2. installs the package,
3. lets you choose the crawl size,
4. optionally asks for a Jev key without writing it to disk,
5. runs the full pipeline,
6. displays the highest-risk rows and generated report paths.

No Jev key is required. Without one, the project still runs end-to-end and uses deterministic semantic heuristics for the semantic component.

## Local quick start

```bash
git clone https://github.com/prestigegitserp/divar_scanner.git
cd divar_scanner
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e '.[dev]'
pytest

divar-scanner run --config config/fatemi.yaml --max-listings 200
```

Disable Jev explicitly:

```bash
divar-scanner run --config config/fatemi.yaml --max-listings 200 --no-jev
```

### Jev configuration

Hosted gateway:

```bash
export JEV_API_KEY='jv_live_...'
```

Official TypeSafe endpoint:

```bash
export TYPESAFE_API_KEY='...'
```

You can override either endpoint with `JEV_ENDPOINT`. The implementation follows the documented typed-decision interface: one `state`, multiple `questions`, and `choice / score / noul` answers. Secrets are read only from environment variables and are never committed.

## What gets collected

The crawler reads public listing/search payloads and detail payloads and normalizes common fields including:

- listing token / URL
- title and description
- neighborhood
- area (m²)
- rooms
- Shamsi year built
- deposit (Toman)
- monthly rent (Toman)
- parking / elevator / storage when present
- floor and the raw structured field map

It intentionally does **not** call contact-info endpoints, login/OTP flows, or phone-number APIs. Phone-like strings in descriptions are redacted by default.

## Market model

Three price views are created:

```text
deposit_per_m2
rent_per_m2
equivalent_deposit_per_m2 = (deposit + rent * multiplier) / area
```

The default `multiplier=30` is an explicit modeling assumption, not a universal market truth. Change it in the YAML or evaluate alternatives.

Peer comparisons are hierarchical:

1. neighborhood + area bucket + rooms
2. neighborhood + area bucket
3. neighborhood
4. global fallback

For each price view, a robust deviation based on median/MAD is calculated. That signal is blended with an Isolation Forest trained on numeric listing features.

## Cross-field consistency

The deterministic checker catches examples such as:

- structured `65 m²` but description says `140 متری`
- structured `1 room` but text says `سه خواب`
- structured `parking=False` but description says `دارای پارکینگ`
- impossible area / room / construction-year ranges
- extreme room-to-area combinations

These are treated primarily as **data-quality evidence**, not automatically as fraud.

## Duplicate detection

Titles + descriptions are transformed with Persian-friendly character `3–5`-gram TF-IDF. A nearest-neighbor search estimates similarity to the most similar other listing. This catches copy/paste listings even when spacing, punctuation, or a few words differ.

The review queue records the nearest similar listing token for manual inspection.

## Jev semantic pass

Only candidates above `semantic.min_prefilter_score` are sent to Jev, capped at `semantic.top_k`. Each request asks independent typed questions:

- `classification`: plausible / data_error / market_outlier / misleading_or_bait / needs_review
- `semantic_suspicion`: calibrated yes/no probability
- `manual_review`: calibrated yes/no probability
- `consistency`: ordered consistency score

The prompt explicitly tells the model that market price alone is not proof of fraud.

Current Jev API documentation: https://www.jevtypesafeai.com/jev/api

## Outputs

Each run creates timestamped files under `outputs/fatemi/`:

- `listings_scored_*.csv`
- `listings_scored_*.parquet`
- `review_queue_*.csv`
- `report_*.html`
- `run_meta_*.json`

The HTML report contains score distribution, a market scatter plot and the top review queue with links back to Divar.

### Key columns

| Column | Meaning |
|---|---|
| `data_quality_score` | missing/impossible/contradictory fields |
| `market_anomaly_score` | unusual versus local peers |
| `duplicate_similarity` | similarity to nearest other ad |
| `semantic_score` | Jev result, or deterministic fallback |
| `suspicion_score` | final blended triage score |
| `risk_band` | `normal`, `review`, `high` |
| `flag_reasons` | human-readable evidence summary |

## Re-analyze an existing dataset

If you already have a normalized CSV/Parquet with the expected columns:

```bash
divar-scanner analyze --config config/fatemi.yaml --input my_listings.parquet
```

## Tests and CI

```bash
pytest
python -m compileall -q src
```

GitHub Actions runs both checks on pushes and pull requests.

## Responsible crawling and schema drift

Divar's public web response schema is not a stable public contract and may change. The project uses conservative retries, caching, a low request rate, and stops on HTTP 401/403/429 instead of attempting to bypass controls. If Divar changes its web schema or access policy, update the adapter rather than increasing request aggression.

Divar also maintains the **Kenar/Open Platform** APIs for authorized integrations; its SDK documents search and district assets. For long-lived production use, prefer an official authorized API when your use case and access allow it.

## Ideas for the next iteration

- time-series snapshots for price-change and repost behavior
- learned ranking model from human review labels (LightGBM / XGBoost)
- image perceptual-hash duplicate detection
- geospatial peer groups using coordinates rather than district labels
- conformal thresholds for review queues
- active learning: sample uncertain cases for manual labeling
- drift dashboards for monthly rent/deposit distributions

## License

MIT. Use responsibly and verify applicable site terms and local law before large-scale collection or operational deployment.
