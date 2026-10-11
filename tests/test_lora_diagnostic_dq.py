"""Synthetic CPU regressions for scope-aware training DQ reports."""
import copy
import csv
import json
import math
from pathlib import Path
import tempfile
import unittest

from tools.lora_dq_report import aggregate_rows, build_auto_monitor, group_dq_rows, mul_settings
from tools.make_lora_diagnostic_report import parse_dq_logs, build_chart_payload, build_html


def row(scope, step=100, **changes):
    result = dict(Scope=scope, TrainStep=step, Epoch=1, Target="delta", Module="", Numel=9 if scope == "unet" else 1,
                  RMS=2 if scope == "unet" else 1, QuantErrRMSRaw=1 if scope == "unet" else .1,
                  QuantErrRatioRaw=.5 if scope == "unet" else .1, ClipRateRaw=.02 if scope == "unet" else .01,
                  Bits=8, RangeMul=2.7, QuantErrRatioEMA=.4, QuantErrRMSEMA=.7)
    result.update(changes)
    return result


def parsed(rows, settings=None):
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "synthetic.csv"
        with path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            writer.writeheader()
            writer.writerows(rows)
        return parse_dq_logs(str(path), None, None, settings)


class DQReportTests(unittest.TestCase):
    def test_scope_is_retained_and_same_step_never_interleaves(self):
        data = parsed([row("unet"), row("te"), row("unet", 200), row("te", 200)],
                      {"values": {"dq_delta_auto_ema": .5}})
        self.assertEqual(len(data["rows"]), 4)
        self.assertEqual(data["rows"][0]["Scope"], "unet")
        self.assertEqual(data["summary"]["steps"], 2)
        self.assertEqual(data["summary"]["final_quant_err_ratio_ema"], .4)
        charts = build_chart_payload(None, data, None, None, None)["dq"]
        for chart in charts:
            self.assertEqual(len(set(chart["x"])), len(chart["x"]), chart["id"])
        ratios = {c["id"]: c for c in charts if c["id"].startswith("dq_qerr_ratio")}
        self.assertEqual(len(ratios), 3)
        self.assertEqual(ratios["dq_qerr_ratio_te"]["series"][0]["y"], [.1, .1])
        self.assertEqual(ratios["dq_qerr_ratio_te"]["series"][1]["y"], [.1, .1])
        self.assertEqual(ratios["dq_qerr_ratio"]["series"][1]["y"], [.4, .4])
        self.assertEqual(len({c["y_max_fixed"] for c in ratios.values()}), 1)

    def test_pooled_ratio_uses_sum_of_squares_not_average_of_ratios(self):
        merged = aggregate_rows([row("unet"), row("te")])
        self.assertAlmostEqual(merged["QuantErrRMSRaw"], math.sqrt(.901))
        self.assertAlmostEqual(merged["RMS"], math.sqrt(3.7))
        self.assertAlmostEqual(merged["QuantErrRatioRaw"], math.sqrt(.901) / (math.sqrt(3.7) + 1e-12))
        self.assertAlmostEqual(merged["ClipRateRaw"], .019)
        self.assertIsNone(merged["Range"])

    def test_per_module_aggregation_does_not_double_count_shared_ema(self):
        rows = [row("unet", Module="block_a"), row("unet", Module="block_b", Numel=1, RMS=1, QuantErrRMSRaw=.1)]
        grouped = group_dq_rows(rows)
        self.assertEqual(len(grouped["plot_rows"]), 1)
        self.assertAlmostEqual(grouped["plot_rows"][0]["QuantErrRMSRaw"], math.sqrt(.901))
        self.assertEqual(grouped["plot_rows"][0]["QuantErrRatioEMA"], .4)

    def test_missing_weights_and_partial_scopes_are_gaps_not_zero(self):
        rows = [row("unet", Numel=None), row("te"), row("unet", 200)]
        grouped = group_dq_rows(rows)
        self.assertTrue(grouped["warnings"])
        self.assertTrue(all(r["QuantErrRatioRaw"] is None for r in grouped["plot_rows"]))
        self.assertEqual(grouped["scope_rows"]["te"][-1], {"TrainStep": 200, "Scope": "te"})

    def test_duplicate_summary_and_mixed_targets_are_not_silently_averaged(self):
        for rows in ([row("unet"), row("unet")], [row("unet"), row("te", Target="z")]):
            data = group_dq_rows(rows)
            self.assertIsNone(data["plot_rows"][0]["QuantErrRatioRaw"])
            self.assertTrue(data["warnings"])

    def test_conflicting_shared_ema_is_missing(self):
        data = group_dq_rows([row("unet"), row("te", QuantErrRatioEMA=.8)])
        self.assertIsNone(data["plot_rows"][0]["QuantErrRatioEMA"])

    def test_legacy_single_scope_and_empty_auto_only_inputs(self):
        data = parsed([row("", 100), row("", 200)])
        self.assertEqual(data["plot_rows"][1]["QuantErrRatioRaw"], .1)
        charts = build_chart_payload(None, data, None, None, None)["dq"]
        self.assertFalse(any(c["id"].endswith("_te") for c in charts))
        self.assertEqual(group_dq_rows([])["plot_rows"], [])
        self.assertEqual(parse_dq_logs(None, None, None)["summary"]["rows"], 0)

    def test_no_ema_coefficient_does_not_invent_a_scope_ema(self):
        data = parsed([row("unet"), row("te")])
        charts = build_chart_payload(None, data, None, None, None)["dq"]
        self.assertEqual(len(next(c for c in charts if c["id"] == "dq_qerr_ratio_te")["series"]), 1)

    def test_unlogged_te_does_not_gain_a_curve_from_its_mul_setting(self):
        data = parsed([row("unet"), row("unet", 200)], {"values": {
            "dq_delta_scope": "unet", "dq_delta_range_mul": 2.7, "dq_delta_range_mul_te": 4.05}})
        charts = build_chart_payload(None, data, None, None, None)
        self.assertEqual(list(data["scope_rows"]), ["unet"])
        self.assertFalse(any(c["id"].endswith("_te") for cs in charts.values() for c in cs))

    def test_auto_monitor_uses_controller_scope_and_not_pooled_log_error_ema(self):
        data = parsed([row("unet"), row("te")])
        data['auto_rows'] = [dict(TrainStep=50, Scope='unet', ClipRateRaw=.001,
            ClipRateEMA=.002, QErrPerClip=140., QErrPerClipClipFloor=.0001,
            ActiveClipLow=.0005, ActiveClipHigh=.0022,
            ClipRateLowAutoThresholdQErrRatio=.25, ClipRateLowAutoThresholdQErrPerClip=130.)]
        monitor = build_auto_monitor(data, {"values": {"dq_delta_scope": "both"}})
        self.assertEqual(monitor['scopes'], ['unet'])
        qerr = next(c for c in monitor['charts'] if c['id'] == 'dq_auto_qerr_unet')
        curve = next(s for s in qerr['series'] if s['name'].startswith('AutoQuantErrRatioEMA'))
        self.assertEqual(curve['y'], [.28])  # .002 * 140; pooled log EMA is .4
        self.assertTrue(any('復元' in s for s in monitor['notes']))

    def test_new_auto_error_columns_win_over_rounded_reconstruction(self):
        data = {'auto_rows': [dict(TrainStep=50, Scope='unet', ClipRateEMA=.002,
                QErrPerClip=140., QErrPerClipClipFloor=.0001, AutoQuantErrRatioEMA=.279999,
                AutoQuantErrRatioRaw=.32)]}
        monitor = build_auto_monitor(data)
        qerr = next(c for c in monitor['charts'] if c['id'] == 'dq_auto_qerr_unet')
        self.assertEqual(qerr['series'][1]['y'], [.279999])
        self.assertFalse(any('復元' in s for s in monitor['notes']))

    def test_partial_log_scope_cannot_stand_in_for_combined_controller_raw(self):
        data = parsed([row('unet', ClipRateEMA=.002)],
                      {'values': {'dq_delta_scope': 'both', 'dq_delta_log_scope': 'unet'}})
        monitor = data['auto_monitor']
        self.assertEqual(monitor['mode'], 'sampled')
        clip = next(c for c in monitor['charts'] if c['id'] == 'dq_auto_clip_both')
        self.assertEqual(clip['series'][0]['y'], [None])
        self.assertEqual(clip['series'][1]['y'], [.002])

    def test_auto_off_keeps_reference_band_without_fabricating_control_history(self):
        data = parsed([row('unet', ActiveClipLow=.0005, ActiveClipHigh=.003),
                       row('te', ActiveClipLow=.0005, ActiveClipHigh=.003)],
                      {'values': {'dq_delta_auto_range_mul': False, 'dq_delta_auto_ema': .95}})
        monitor = data['auto_monitor']
        self.assertEqual(monitor['mode'], 'reference')
        self.assertEqual(len(monitor['charts']), 2)
        self.assertTrue(all('clip_' in c['id'] for c in monitor['charts']))
        self.assertTrue(all(any(s['name'] == '参考上限（OFF）' for s in c['series']) for c in monitor['charts']))
        html = build_html({'dq': data, 'charts': build_chart_payload(None, data, None, None, None)})
        self.assertIn("id='section-dq_auto'", html)
        self.assertIn('QErrPerClip・判定用の誤差EMA・mul変更履歴は未記録', html)

    def test_parser_retains_new_auto_columns(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'auto.csv'
            path.write_text('TrainStep,Scope,ClipRateRaw,AutoQuantErrRatioRaw,AutoQuantErrRatioEMA\n50,unet,.003,.2,.25\n', encoding='utf-8')
            data = parse_dq_logs(None, str(path), None)
        self.assertEqual(data['auto_rows'][0]['AutoQuantErrRatioEMA'], .25)
        self.assertEqual(data['auto_monitor']['scopes'], ['unet'])

    def test_inactive_auto_band_is_not_presented_as_a_limit(self):
        rows = [row("unet", ActiveClipLow=.0005, ActiveClipHigh=.003)]
        for enabled in (False, None, True):
            data = parsed(rows, {"values": {"dq_delta_auto_range_mul": enabled}})
            charts = build_chart_payload(None, data, None, None, None)["dq"]
            clip = next(c for c in charts if c["id"] == "dq_clip")
            self.assertEqual(any(s["name"] == "ActiveClipHigh" for s in clip["series"]), enabled is not False)

    def test_mul_cli_precedence_even_with_unet_scope(self):
        result = mul_settings({"values": {"dq_delta_scope": "unet", "dq_delta_range_mul": 2.7,
                             "dq_delta_range_mul_attn2": 4.05, "dq_delta_range_mul_te": 4.05, "dq_delta_range_mul_te2": 3.75}})
        self.assertEqual([r["mul"] for r in result["rows"]], [2.7, 4.05, 4.05, 3.75])
        self.assertTrue(all(r["enabled"] for r in result["rows"]))

    def test_saved_policy_is_used_without_reading_mutable_policy_file(self):
        policy = dict(base_mul=3, components={"te1": 4}, group_overrides=[], te_quantized=False)
        settings = {"values": {"dq_delta_policy_file": "missing.json", "dq_delta_range_mul": 8},
                    "resolved": {"metadata": {"ss_dq_mul_policy": json.dumps(policy)}}}
        result = mul_settings(settings)
        self.assertEqual(result["base_mul"], 3)
        self.assertFalse(result["rows"][1]["enabled"])
        settings.pop("resolved")
        self.assertFalse(mul_settings(settings)["rows"])
        self.assertFalse(mul_settings({"raw_metadata": {"ss_dq_mul_policy": "bad"}})["rows"])

    def test_raw_inputs_unchanged_and_html_exposes_correct_scope_controls(self):
        rows = [row("unet"), row("te")]
        before = copy.deepcopy(rows)
        group_dq_rows(rows)
        self.assertEqual(rows, before)
        data = parsed(rows, {"values": {"dq_delta_range_mul": 2.7, "dq_delta_range_mul_te": 4.05}})
        report = {"base_name": "synthetic", "dq": data,
                  "charts": build_chart_payload(None, data, None, None, None)}
        html = build_html(report)
        self.assertIn("部位別mul（固定配分）", html)
        self.assertIn("id='dq-scope-view'", html)
        self.assertIn("基準mul（ログ列）", html)
        self.assertIn("TE1：4.05", html)
        self.assertIn("summaryログのTEはTE1＋TE2合算", html)


if __name__ == "__main__":
    unittest.main()
