from __future__ import annotations

from typing import Any

from .decision import listing_questions


CASES: list[dict[str, Any]] = [
    {
        'name': 'disposition_clear_data_mismatch',
        'question': 'disposition',
        'state': (
            'عنوان ۱۴۰ متر سه خواب؛ فیلد ساختاریافته ۶۵ متر، یک اتاق، بدون پارکینگ؛ '
            'متن دوباره ۱۴۰ متر سه خواب و پارکینگ را می‌گوید. '
            'market evidence معمولی است و duplicate evidence وجود ندارد.'
        ),
        'expected': 'data_error',
    },
    {
        'name': 'disposition_coherent_normal',
        'question': 'disposition',
        'state': (
            '۸۵ متر دو خواب فاطمی؛ متن و فیلدها سازگار؛ قیمت نزدیک peerها؛ '
            'هیچ duplicate conflict یا mismatch معناداری وجود ندارد.'
        ),
        'expected': 'plausible',
    },
    {
        'name': 'disposition_market_only',
        'question': 'disposition',
        'state': (
            'آگهی از نظر متن و فیلدها کاملاً سازگار است و duplicate ندارد. '
            'چند مدل مستقل بازار انحراف قیمت بسیار شدید را تأیید می‌کنند. '
            'هیچ شاهد مستقلی برای گمراه‌کنندگی وجود ندارد.'
        ),
        'expected': 'market_outlier',
    },
    {
        'name': 'disposition_duplicate_conflict',
        'question': 'disposition',
        'state': (
            'متن تقریباً یکسان در چند آگهی با محله و مشخصات اصلی متناقض استفاده شده؛ '
            'این تفاوت با parser error توضیح داده نمی‌شود.'
        ),
        'expected': 'misleading_or_bait',
    },
    {
        'name': 'integrity_consistent',
        'question': 'integrity_class',
        'state': 'عنوان، توضیحات و فیلد ساختاریافته همگی ۸۰ متر دو خواب با پارکینگ را گزارش می‌کنند.',
        'expected': 'consistent',
    },
    {
        'name': 'integrity_parser_like',
        'question': 'integrity_class',
        'state': (
            'عنوان و متن چند بار ۱۲۰ متر سه خواب را تکرار می‌کنند اما یک فیلد استخراج‌شده '
            '۱۲ متر و صفر خواب ثبت شده و سایر فیلدهای استخراجی نیز ناقص‌اند.'
        ),
        'expected': 'extraction_error',
    },
    {
        'name': 'duplicate_template_reuse',
        'question': 'duplicate_pattern',
        'state': (
            'چند آگهی از عبارت عمومی «فول امکانات، دسترسی عالی» استفاده می‌کنند اما متن اصلی، '
            'محله و مشخصات ملک متفاوت است و similarity کلی پایین است.'
        ),
        'expected': 'normal_template_reuse',
    },
    {
        'name': 'duplicate_cross_property_conflict',
        'question': 'duplicate_pattern',
        'state': (
            'duplicate_similarity=0.98؛ متن تقریباً یکسان؛ cluster size=6؛ همان متن برای '
            'سه محله و متراژهای ناسازگار استفاده شده است.'
        ),
        'expected': 'cross_property_conflict',
    },
    {
        'name': 'market_typical',
        'question': 'market_status',
        'state': (
            'متراژ ۹۰، دو خواب؛ robust market anomaly=0.08؛ OOF actual/expected=1.03؛ '
            'LOF=0.10؛ peer sample کافی است.'
        ),
        'expected': 'typical',
    },
    {
        'name': 'market_extreme',
        'question': 'market_status',
        'state': (
            'متراژ ۹۰، دو خواب؛ robust anomaly=0.93؛ OOF price anomaly=0.95؛ '
            'actual/expected=0.42؛ LOF=0.88؛ چند سیگنال مستقل هم‌جهت‌اند.'
        ),
        'expected': 'extreme_outlier',
    },
    {
        'name': 'bait_negative_control_price_only',
        'question': 'bait_evidence',
        'state': (
            'متن و فیلدها سازگارند و هیچ duplicate conflict وجود ندارد. '
            'تنها نکته این است که قیمت از بازار بسیار پایین‌تر است. قیمت پرت به تنهایی شاهد bait نیست.'
        ),
        'expected_binary': False,
    },
    {
        'name': 'bait_positive_duplicate_conflict',
        'question': 'bait_evidence',
        'state': (
            'همان متن تقریباً بدون تغییر برای چند ملک با محله، متراژ و مشخصات متناقض منتشر شده '
            'و این تفاوت‌ها با extraction error توضیح داده نمی‌شوند.'
        ),
        'expected_binary': True,
    },
]


def run_diagnostic(engine) -> dict[str, Any]:
    questions = listing_questions()
    requests = [
        (case['state'], {case['question']: questions[case['question']]})
        for case in CASES
    ]
    answers = engine.decide_many(requests)

    rows = []
    correct = 0
    for case, answer_map in zip(CASES, answers):
        qid = case['question']
        answer = answer_map[qid]
        if 'expected' in case:
            predicted = answer['choice']
            ok = predicted == case['expected']
            expected = case['expected']
            observed = predicted
        else:
            p_yes = float(answer['noul'])
            predicted_bool = p_yes >= 0.5
            ok = predicted_bool == bool(case['expected_binary'])
            expected = str(bool(case['expected_binary']))
            observed = f'{predicted_bool} (Pyes={p_yes:.3f})'
        correct += int(ok)
        rows.append({
            'case': case['name'],
            'question': qid,
            'expected': expected,
            'observed': observed,
            'correct': ok,
            'confidence': answer.get('effective_confidence', answer.get('answer_confidence', answer.get('confidence'))),
            'order_stability': answer.get('diagnostics', {}).get('order_stability'),
        })

    return {
        'model': engine.model_name,
        'backend': engine.backend_name,
        'correct': correct,
        'total': len(CASES),
        'accuracy_on_diagnostic_only': correct / max(len(CASES), 1),
        'rows': rows,
        'warning': (
            'This is a hand-written Persian wiring/shortcut diagnostic, not a real-world accuracy estimate. '
            'Use held-out human-labelled Fatemi data for performance or calibration claims.'
        ),
    }