from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from divar_scanner.config import Config
from divar_scanner.pipeline import _finalize_scores


def _cfg():
    return Config(
        raw={
            'anomaly': {'duplicate_similarity_threshold': 0.88},
            'scoring': {
                'data_rule_weight': 0.72,
                'data_decision_weight': 0.28,
                'market_stat_weight': 0.82,
                'market_decision_weight': 0.18,
                'misleading_duplicate_weight': 0.42,
                'misleading_decision_weight': 0.58,
                'abstain_review_uplift': 0.08,
                'review_threshold': 0.55,
                'high_risk_threshold': 0.75,
            },
        },
        source=Path('test.yaml'),
    )


def test_market_outlier_does_not_leak_into_misleading_risk():
    df = pd.DataFrame([
        {
            'duplicate_similarity': 0.10, 'duplicate_bait_score': 0.05,
            'data_quality_score': 0.10, 'market_anomaly_score': 0.10,
            'decision_evaluated': False,
        },
        {
            'duplicate_similarity': 0.10, 'duplicate_bait_score': 0.05,
            'data_quality_score': 0.10, 'market_anomaly_score': 0.95,
            'decision_evaluated': False,
        },
    ])
    out = _finalize_scores(df, _cfg())
    by_market = out.sort_values('market_outlier_score')
    low, high = by_market.iloc[0], by_market.iloc[1]
    assert high['market_outlier_score'] > low['market_outlier_score']
    assert np.isclose(high['misleading_risk_score'], low['misleading_risk_score'])
    assert np.isclose(high['suspicion_score'], low['suspicion_score'])


def _decision_row(abstain: bool):
    return {
        'duplicate_similarity': 0.20,
        'duplicate_bait_score': 0.10,
        'data_quality_score': 0.15,
        'market_anomaly_score': 0.20,
        'decision_evaluated': True,
        'decision_data_error_probability': 0.10,
        'decision_integrity_probs_json': json.dumps({
            'consistent': 0.8, 'extraction_error': 0.05,
            'listing_claim_conflict': 0.05, 'insufficient_evidence': 0.1,
        }),
        'decision_market_status_probs_json': json.dumps({
            'typical': 0.8, 'moderate_outlier': 0.1,
            'extreme_outlier': 0.05, 'insufficient_context': 0.05,
        }),
        'decision_bait_probability': 0.20,
        'decision_disposition_probs_json': json.dumps({
            'plausible': 0.7, 'data_error': 0.05, 'market_outlier': 0.05,
            'misleading_or_bait': 0.15, 'ambiguous_mixed': 0.05,
        }),
        'decision_duplicate_pattern_probs_json': json.dumps({
            'no_duplicate_evidence': 0.7, 'normal_template_reuse': 0.15,
            'likely_same_property_repost': 0.1, 'cross_property_conflict': 0.05,
        }),
        'decision_effective_confidence': 0.65,
        'decision_manual_review_probability': 0.45,
        'decision_coherence_score': 0.70,
        'decision_abstain': abstain,
    }


def test_abstention_raises_review_priority_not_misleading_probability():
    out = _finalize_scores(pd.DataFrame([_decision_row(False), _decision_row(True)]), _cfg())
    non = out[out['decision_abstain'] == False].iloc[0]  # noqa: E712
    abst = out[out['decision_abstain'] == True].iloc[0]  # noqa: E712
    assert np.isclose(non['misleading_risk_score'], abst['misleading_risk_score'])
    assert abst['review_priority_score'] > non['review_priority_score']