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
        x="suspicion_score",
        nbins=25,
        title="Suspicion score distribution",
        hover_data=["token", "title", "risk_band"],
    )
    scatter = px.scatter(
        df,
        x="area_m2",
        y="equivalent_deposit_per_m2",
        color="suspicion_score",
        hover_data=[
            "title",
            "neighborhood",
            "deposit_toman",
            "rent_monthly_toman",
            "risk_band",
            "flag_reasons",
        ],
        title="Market map: area vs equivalent deposit / m²",
    )
    scatter.update_yaxes(type="log")

    top = df.nlargest(min(60, len(df)), "suspicion_score").copy()
    show_cols = [
        "suspicion_score",
        "risk_band",
        "title",
        "neighborhood",
        "area_m2",
        "rooms",
        "deposit_toman",
        "rent_monthly_toman",
        "market_anomaly_score",
        "data_quality_score",
        "duplicate_similarity",
        "semantic_score",
        "jev_classification",
        "flag_reasons",
        "url",
    ]
    show_cols = [c for c in show_cols if c in top.columns]
    top = top[show_cols]
    for c in ("deposit_toman", "rent_monthly_toman"):
        if c in top:
            top[c] = top[c].map(_fmt_money)
    if "url" in top:
        top["url"] = top["url"].map(lambda u: f'<a href="{html.escape(str(u))}" target="_blank">open</a>')
    table_html = top.to_html(index=False, escape=False, classes="risk-table")

    district = ", ".join(x.get("name", "") for x in meta.get("district_matches", []))
    jev = meta.get("semantic", {})
    summary = f"""
    <div class="cards">
      <div class="card"><b>{len(df)}</b><span>listings</span></div>
      <div class="card"><b>{int((df['risk_band'] == 'high').sum())}</b><span>high-risk review</span></div>
      <div class="card"><b>{int((df['risk_band'] == 'review').sum())}</b><span>manual review</span></div>
      <div class="card"><b>{html.escape(district or 'unknown')}</b><span>resolved district</span></div>
      <div class="card"><b>{jev.get('jev_evaluated', 0)}</b><span>Jev-evaluated</span></div>
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
body {{ font-family: system-ui, -apple-system, Segoe UI, Tahoma, sans-serif; margin: 24px; background: #f7f7f8; color: #171717; }}
.wrap {{ max-width: 1500px; margin: auto; }}
h1 {{ margin-bottom: 4px; }}
.note {{ background: #fff7df; border: 1px solid #ead28a; padding: 12px 16px; border-radius: 10px; line-height: 1.8; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; margin:16px 0; }}
.card {{ background:white; border:1px solid #e6e6e6; padding:16px; border-radius:12px; display:flex; flex-direction:column; gap:6px; }}
.card b {{ font-size:22px; direction:ltr; }}
.panel {{ background:white; border:1px solid #e6e6e6; border-radius:12px; padding:12px; margin:14px 0; overflow:auto; }}
.risk-table {{ border-collapse:collapse; width:100%; font-size:13px; direction:rtl; }}
.risk-table th,.risk-table td {{ border-bottom:1px solid #eee; padding:8px; text-align:right; vertical-align:top; }}
.risk-table th {{ position:sticky; top:0; background:#fafafa; }}
a {{ color:#0759c7; }}
</style>
</head>
<body><div class="wrap">
<h1>اسکنر آگهی اجاره — فاطمی تهران</h1>
<p>Pipeline: crawl → normalize → consistency checks → local market anomaly → duplicate detection → optional Jev semantic review.</p>
<div class="note"><b>تفسیر مهم:</b> امتیاز بالا به معنی «اثبات تقلب» نیست. خروجی برای triage و بازبینی انسانی است و بین خطای داده، outlier واقعی بازار و آگهی مشکوک تمایز می‌گذارد.</div>
{summary}
<div class="panel">{hist_html}</div>
<div class="panel">{scatter_html}</div>
<div class="panel"><h2>صف بازبینی</h2>{table_html}</div>
</div></body></html>"""
    p.write_text(doc, encoding="utf-8")
    return p
