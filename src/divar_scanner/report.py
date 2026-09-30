from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.io as pio


def _fmt_money(v: Any) -> str:
    try:
        if pd.isna(v):
            return "—"
        n = float(v)
    except Exception:
        return "—"
    if abs(n) >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f} B"
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.1f} M"
    return f"{n:,.0f}"


def build_html_report(df: pd.DataFrame, meta: dict[str, Any], output_path: str | Path) -> Path:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        p.write_text("<html><body><h1>No listings collected</h1></body></html>", encoding="utf-8")
        return p

    hist = px.histogram(
        df,
        x="review_priority_score" if "review_priority_score" in df else "suspicion_score",
        nbins=25,
        title="Review-priority distribution",
        hover_data=[c for c in ["token", "title", "risk_band", "flag_reasons"] if c in df],
    )
    scatter = px.scatter(
        df,
        x="area_m2",
        y="equivalent_deposit_per_m2",
        color="review_priority_score" if "review_priority_score" in df else "suspicion_score",
        hover_data=[
            c for c in [
                "title", "neighborhood", "deposit_toman", "rent_monthly_toman",
                "price_model_ratio", "market_anomaly_score", "duplicate_cluster_size",
                "semantic_provider", "semantic_classification", "risk_band", "flag_reasons"
            ] if c in df
        ],
        title="Market map: area vs equivalent deposit / m²",
    )
    scatter.update_yaxes(type="log")

    if "price_model_expected_equiv_deposit" in df and df["price_model_expected_equiv_deposit"].notna().any():
        price_df = df[df["price_model_expected_equiv_deposit"].notna()].copy()
        price_fig = px.scatter(
            price_df,
            x="price_model_expected_equiv_deposit",
            y="equivalent_deposit_toman",
            color="price_model_anomaly_score",
            hover_data=[c for c in ["title", "neighborhood", "price_model_ratio", "flag_reasons"] if c in price_df],
            title="OOF price model: expected vs observed equivalent deposit",
        )
        price_fig.update_xaxes(type="log")
        price_fig.update_yaxes(type="log")
        price_html = pio.to_html(price_fig, include_plotlyjs=False, full_html=False)
    else:
        price_html = "<p>OOF price model requires enough valid listings (roughly 40+).</p>"

    sort_col = "review_priority_score" if "review_priority_score" in df else "suspicion_score"
    top = df.nlargest(min(80, len(df)), sort_col).copy()
    show_cols = [
        "review_priority_score", "suspicion_score", "uncertainty_score", "risk_band",
        "title", "neighborhood", "area_m2", "rooms", "deposit_toman", "rent_monthly_toman",
        "price_model_ratio", "price_model_anomaly_score", "lof_anomaly_score",
        "market_anomaly_score", "data_quality_score", "duplicate_similarity",
        "duplicate_cluster_size", "duplicate_bait_score", "semantic_score",
        "semantic_provider", "semantic_classification", "semantic_confidence",
        "flag_reasons", "url",
    ]
    top = top[[c for c in show_cols if c in top.columns]]
    for c in ("deposit_toman", "rent_monthly_toman"):
        if c in top:
            top[c] = top[c].map(_fmt_money)
    if "url" in top:
        top["url"] = top["url"].map(
            lambda u: f'<a href="{html.escape(str(u))}" target="_blank">open</a>'
        )
    table_html = top.to_html(index=False, escape=False, classes="risk-table")

    district = ", ".join(x.get("name", "") for x in meta.get("district_matches", []))
    semantic = meta.get("semantic", {})
    used = ", ".join(semantic.get("providers_used", [])) or "heuristic"
    summary = f"""
    <div class="cards">
      <div class="card"><b>{len(df)}</b><span>listings</span></div>
      <div class="card"><b>{int((df['risk_band'] == 'high').sum())}</b><span>high priority</span></div>
      <div class="card"><b>{int((df['risk_band'] == 'review').sum())}</b><span>manual review</span></div>
      <div class="card"><b>{html.escape(district or 'unknown')}</b><span>resolved district</span></div>
      <div class="card"><b>{semantic.get('evaluated', 0)}</b><span>semantic-evaluated</span></div>
      <div class="card"><b>{html.escape(used)}</b><span>semantic backend</span></div>
    </div>
    """

    hist_html = pio.to_html(hist, include_plotlyjs="cdn", full_html=False)
    scatter_html = pio.to_html(scatter, include_plotlyjs=False, full_html=False)
    doc = f"""<!doctype html>
<html lang="fa" dir="rtl">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Divar Fatemi Rental Scanner</title>
<style>
body {{ font-family: system-ui,-apple-system,Segoe UI,Tahoma,sans-serif; margin:24px; background:#f7f7f8; color:#171717; }}
.wrap {{ max-width:1600px; margin:auto; }}
h1 {{ margin-bottom:4px; }}
.note {{ background:#fff7df; border:1px solid #ead28a; padding:12px 16px; border-radius:10px; line-height:1.8; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; margin:16px 0; }}
.card {{ background:white; border:1px solid #e6e6e6; padding:16px; border-radius:12px; display:flex; flex-direction:column; gap:6px; }}
.card b {{ font-size:20px; direction:ltr; overflow-wrap:anywhere; }}
.panel {{ background:white; border:1px solid #e6e6e6; border-radius:12px; padding:12px; margin:14px 0; overflow:auto; }}
.risk-table {{ border-collapse:collapse; width:100%; font-size:12px; direction:rtl; }}
.risk-table th,.risk-table td {{ border-bottom:1px solid #eee; padding:7px; text-align:right; vertical-align:top; }}
.risk-table th {{ position:sticky; top:0; background:#fafafa; }}
a {{ color:#0759c7; }}
</style>
</head>
<body><div class="wrap">
<h1>اسکنر چندلایهٔ آگهی اجاره — فاطمی تهران</h1>
<p>crawl → normalize → rules → robust peers → Isolation Forest → OOF price model → LOF → duplicate graph → semantic provider → evidence fusion</p>
<div class="note"><b>تفسیر:</b> این سیستم تقلب را «اثبات» نمی‌کند. خروجی، اولویت بازبینی مبتنی بر چند منبع شواهد مستقل است؛ خطای parser، ملک واقعاً خاص و آگهی گمراه‌کننده از هم جدا نگه داشته می‌شوند.</div>
{summary}
<div class="panel">{hist_html}</div>
<div class="panel">{scatter_html}</div>
<div class="panel">{price_html}</div>
<div class="panel"><h2>صف بازبینی</h2>{table_html}</div>
</div></body></html>"""
    p.write_text(doc, encoding="utf-8")
    return p
