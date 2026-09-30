from __future__ import annotations

import json

import pandas as pd

from divar_scanner.evaluation import evaluate_frame


def _raw(disposition_probs):
    return json.dumps({
        'disposition': {'probabilities': disposition_probs},
        'bait_evidence': {'probabilities': {'no': 0.9, 'yes': 0.1}},
    })


def test_evaluator_reports_probability_and_selective_metrics():
    df = pd.DataFrame([
        {
            'decision_evaluated': True,
            'decision_abstain': False,
            'decision_raw_json': _raw({'plausible': 0.9, 'data_error': 0.1}),
            'human_disposition': 'plausible',
            'human_bait': 'no',
        },
        {
            'decision_evaluated': True,
            'decision_abstain': True,
            'decision_raw_json': _raw({'plausible': 0.2, 'data_error': 0.8}),
            'human_disposition': 'plausible',
            'human_bait': 'no',
        },
    ])
    result = evaluate_frame(df)
    disp = result['questions']['disposition']
    assert disp['n'] == 2
    assert disp['accuracy'] == 0.5
    assert disp['coverage_non_abstained'] == 0.5
    assert disp['selective_accuracy_non_abstained'] == 1.0
    assert 0 <= disp['ece_10bin'] <= 1
    assert result['decision_abstention_rate'] == 0.5