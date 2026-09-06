"""Display regressions: missing values must not erase observations or warnings."""
import copy
import json
import math
import unittest

from tools.lora_diagnostic_display import prepare_display, render_body
from tools.make_lora_diagnostic_report import build_html


def chart(chart_id, values, name="observed", **extra):
    return {"id": chart_id, "title": chart_id, "x_label": "TrainStep",
            "x": list(range(len(values))), "series": [{"name": name, "y": values}], **extra}


def report(charts, section="grad", **extra):
    return {"base_name": "test", "charts": {section: charts},
            "diagnostics": {"score": 60, "overall_status": "注意", "overall_class": "warn", "checks": []}, **extra}


class DiagnosticDisplayTests(unittest.TestCase):
    def test_missing_and_reference_only_charts_are_omitted(self):
        c = chart("grad_norm", [None, math.nan])
        c["series"].append({"name": "Threshold", "y": [10, 10]})
        view, notes = prepare_display(report([c]))
        self.assertEqual(view["charts"]["grad"], [])
        self.assertTrue(any("参考線のみ" in n for n in notes["grad"]))

    def test_zero_is_data_and_gaps_keep_their_positions(self):
        c = chart("future_chart", [0, None, 2])
        c["series"].append({"name": "empty", "y": [None] * 3})
        c["legend_rows"] = [{"name": "observed"}, {"name": "empty"}]
        original = report([c])
        snapshot = copy.deepcopy(original)
        view, notes = prepare_display(original)
        shown = view["charts"]["grad"][0]
        self.assertEqual(shown["series"][0]["y"], [0, None, 2])
        self.assertEqual(len(shown["series"]), 1)
        self.assertEqual(shown["legend_rows"], [{"name": "observed"}])
        self.assertTrue(shown["display_missing"])
        self.assertEqual(original, snapshot)

    def test_finite_y_with_no_valid_x_is_not_an_observation(self):
        c = chart("unknown", [2, 3], x=[None, math.inf])
        view, _ = prepare_display(report([c]))
        self.assertFalse(view["charts"]["grad"])

    def test_series_specific_x_is_supported(self):
        c = chart("unknown", [])
        c["series"] = [{"name": "valid", "x": [5, 10], "y": [0, 2]}]
        view, _ = prepare_display(report([c]))
        self.assertEqual(len(view["charts"]["grad"]), 1)

    def test_only_selected_constants_collapse(self):
        ids = ["rank_dim", "dq_rank_dim", "dq_bits", "dq_range_mul", "thresh_off", "rank_sat", "dq_guard_gain", "future_chart"]
        view, _ = prepare_display(report([chart(i, [0, 0]) for i in ids]))
        modes = {c["id"]: c["display_mode"] for c in view["charts"]["grad"]}
        self.assertEqual([i for i in ids if modes[i] == "constant"], ids[:5])

    def test_nonzero_threshoff_and_unrounded_changes_stay_visible(self):
        view, _ = prepare_display(report([chart("thresh_off", [1, 1]), chart("dq_range_mul", [1.0000001, 1.0000002])]))
        self.assertTrue(all(c["display_mode"] == "chart" for c in view["charts"]["grad"]))

    def test_partial_constant_does_not_claim_complete_coverage(self):
        source = report([chart("rank_dim", [4, None, 4])], section="rank")
        view, notes = prepare_display(source)
        html = render_body(view, notes)
        self.assertIn("有効な記録では一定・一部欠測", html)
        self.assertNotIn("記録期間中一定", html)

    def test_single_observation_is_not_a_constant(self):
        view, _ = prepare_display(report([chart("rank_dim", [None, 4])]))
        self.assertEqual(view["charts"]["grad"][0]["display_mode"], "single")
        self.assertEqual(view["charts"]["grad"][0]["display_values"][0]["x"], 1)

    def test_empty_sections_cards_and_tables_are_not_rendered(self):
        html = build_html(report([chart("cosine", [None, None])], lora={"summary_cards": {}, "module_summary": [], "unet_block_summary": []}))
        body = html.split("<script>")[0]
        self.assertNotIn("id='section-grad'", body)
        self.assertNotIn("id='section-lora'", body)
        self.assertNotIn("class='card'", body)
        self.assertNotIn("表示できるデータがありません", body)
        self.assertIn("入力データ・解析状況", body)
        self.assertIn("cosine", body)

    def test_error_reason_remains_visible_and_escaped(self):
        html = build_html(report([], lora_error="LoRA解析に失敗しました: <bad>"))
        body = html.split("<script>")[0]
        self.assertIn("class='analysis-errors'", body)
        self.assertIn("LoRA解析に失敗しました", body)
        self.assertIn("&lt;bad&gt;", body)
        self.assertNotIn("<bad>", body)

    def test_diagnostic_score_and_warning_are_preserved(self):
        source = report([])
        source["diagnostics"]["checks"] = [
            {"section": "DQ", "name": "risk", "value": "-", "status": "warn", "note": "注意 (reason)"},
            {"section": "DQ", "name": "missing", "value": "-", "status": "info", "note": "データ不足"},
        ]
        original = copy.deepcopy(source)
        html = build_html(source)
        self.assertEqual(source, original)
        self.assertIn("60点", html)
        self.assertIn("<td>reason</td>", html)
        self.assertIn("missing：データ不足", html)
        self.assertNotIn("<td>missing</td>", html)

    def test_reference_observations_remain_with_single_sample(self):
        c = chart("grad_norm", [3])
        c["series"].append({"name": "Threshold", "y": [10]})
        view, _ = prepare_display(report([c]))
        self.assertEqual([v["name"] for v in view["charts"]["grad"][0]["display_values"]], ["observed", "Threshold"])
        c["series"][1] = {"name": "Threshold", "x": [0, 1], "y": [10, 20]}
        view, _ = prepare_display(report([c]))
        self.assertEqual(view["charts"]["grad"][0]["display_mode"], "chart")

    def test_recorded_warnings_remain_even_without_charts(self):
        html = build_html(report([], section="dq_guard", dq_guard={"warnings": ["profile is provisional"], "path": "guard.jsonl"}))
        body = html.split("<script>")[0]
        self.assertIn("id='section-dq_guard'", body)
        self.assertIn("profile is provisional", body)
        self.assertIn("guard.jsonl", body)

    def test_unknown_chart_and_critical_caption_survive(self):
        c = chart("future_guard", [1, 1], subtitle="Actual / Shadow: important")
        view, _ = prepare_display(report([c], section="future_section"))
        self.assertEqual(view["charts"]["future_section"][0]["subtitle"], c["subtitle"])
        html = build_html(report([c], section="future_section"))
        self.assertIn("id='charts-future_section'", html)

    def test_heatmap_is_independent_of_rank_timeseries(self):
        heatmap = {"role_labels": ["q"], "rows": [{"label": "down", "cells": [{"value": 0}]}]}
        html = build_html(report([], rank={"final_heatmaps": {"rank_sat_wmean": heatmap}}))
        self.assertIn("id='section-rank'", html)
        self.assertIn("0.000", html)
        heatmap["rows"][0]["cells"][0]["value"] = None
        html = build_html(report([], rank={"final_heatmaps": {"rank_sat_wmean": heatmap}}))
        self.assertNotIn("id='section-rank'", html)

    def test_html_embedded_json_cannot_close_script(self):
        source = report([chart("future", [1, 2], subtitle="</script><script>alert(1)</script>")])
        html = build_html(source)
        payload = html.split("const reportData = ", 1)[1].split(";\n", 1)[0]
        self.assertNotIn("</script>", payload)
        self.assertEqual(json.loads(payload)["charts"]["grad"][0]["subtitle"], source["charts"]["grad"][0]["subtitle"])


if __name__ == "__main__":
    unittest.main()
