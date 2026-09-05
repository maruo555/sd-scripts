"""The user's eight arithmetic fixtures, exercised through production aggregation."""
import json
import math
from pathlib import Path

import pytest

from dq_profile.dataset_diagnostics import aggregate, gradient_metrics, image_equal_weights, improvement, summarize_samples, weighted_quantile
from test_dq_dataset_diagnostics import fixture_rows

CASES = json.loads((Path(__file__).parent / 'fixtures/dq_dataset_diagnostics_golden_cases.json').read_text(encoding='utf-8'))['cases']


@pytest.mark.parametrize('case',CASES,ids=lambda c:c['id'])
def test_user_golden_case(case):
    op,values=case['operation'],case['input']
    if op=='pre_post':
        actual=improvement(values['pre'],values['post'])
    elif op=='weighted_quantile':
        actual={'weighted_quantile':weighted_quantile(values['values'],values['weights'],values['p'])}
    elif op in ('gradient_metrics','gradient_metrics_zero'):
        g0,gm=values['grad_norm_noquant'],values['grad_norm_candidate']
        # The legacy fixture supplies cosine only; live collection uses direct ExactGradient GD.
        gd=values.get('grad_diff_norm',math.sqrt(max(0,g0*g0+gm*gm-2*g0*gm*values.get('cosine',0))))
        m=gradient_metrics(dict(reference_norm=g0,candidate_norm=gm,difference_norm=gd,cosine=values.get('cosine'),topology_matches=True))
        actual=dict(relative_gradient_distance=m['d'],grad_diff_norm=m['grad_diff_norm'],symmetric_gradient_distance=m['symmetric_d'],gradient_cosine=m['cosine'],gradient_norm_ratio=m['norm_ratio'],reference_near_zero=m['reference_near_zero'])
    else:
        n=1 if op=='paired_quant_loss' else len(values['pre']) if op=='group_improvement' else sum(len(v) for v in values.values())
        inv,refs,qs,muls=fixture_rows(n)
        if op=='paired_quant_loss':
            for r in refs:r['raw_mse']=values['noquant_by_noise'][r['noise']]
            for q in qs:q['raw_mse']=values['candidate_by_noise_and_repeat'][q['noise']][q['quant_repeat']]
        elif op=='group_improvement':
            for r in refs:r['raw_mse']=values['pre' if r['snapshot']=='pre' else 'post'][int(r['sample_id'][1:])]
        else:
            flat=[(image,x) for image,views in values.items() for x in views]
            for i,s in enumerate(inv):s['image_id']=flat[i][0]
            for r in refs:r['raw_mse']=flat[int(r['sample_id'][1:])][1]
        a=aggregate(summarize_samples(inv,refs,qs,muls),muls)
        if op=='paired_quant_loss':
            q=a['quant'][0];actual=dict(quant_loss_delta=q['delta'],matched_noquant_mean=q['matched'],quant_loss_relative=q['relative'],reference_noise_count=2)
        elif op=='group_improvement':actual=a
        else:actual=dict(image_equal_mean=a['loss_post'],view_weights=image_equal_weights(inv))
    for key,expected in case['expected'].items():
        if expected is None or isinstance(expected,bool):assert actual[key] is expected
        else:assert actual[key]==pytest.approx(expected,rel=1e-12,abs=1e-12)
