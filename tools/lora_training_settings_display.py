"""Compact settings presentation; shared JSON remains the source for comparison."""
import json
from html import escape

LABELS = {
    "learning_rate": "共通 学習率", "unet_lr": "UNet 学習率の指定", "text_encoder_lr": "共通TE 学習率の指定",
    "text_encoder_lr1": "TE1 学習率の指定", "text_encoder_lr2": "TE2 学習率の指定",
    "network_dim": "Rank", "network_alpha": "Alpha", "seed": "Seed", "optimizer": "Optimizer（記録値）",
    "optimizer_type": "Optimizerの指定", "mixed_precision": "Mixed precision", "lr_scheduler": "Scheduler",
    "fp16_safe_norms_mode": "FP16 safe norms", "fp16_safe_norms_mode_resolved": "FP16 safe norms（確定）",
    "max_train_steps": "予定step数", "gradient_accumulation_steps": "Gradient accumulation",
    "dq_delta_auto_preset": "DQ auto presetの指定", "avg_cp": "チェックポイント平均化",
    "pretrained_model_name_or_path": "ベースモデル", "sd_model_name": "ベースモデル名",
}


def _value(value):
    if value is None:
        return "None（指定なし）"
    if isinstance(value, bool):
        return "true" if value else "false"
    return json.dumps(value, ensure_ascii=False, indent=2) if isinstance(value, (dict, list)) else str(value)


def render_settings(settings):
    if not settings:
        return ""
    def esc(value):
        return escape(str(value), quote=True)

    def rows(values, labels=None):
        return "<div class='table-scroll'><table><thead><tr><th>項目</th><th>記録値</th></tr></thead><tbody>" + "".join(
            f"<tr><td>{esc((labels or {}).get(key, key))}</td><td style='white-space:pre-wrap;overflow-wrap:anywhere'>{esc(_value(value))}</td></tr>"
            for key, value in values.items()
        ) + "</tbody></table></div>"

    def fold(title, content):
        return f"<details class='help'><summary>{esc(title)}</summary>{content}</details>"

    source = {"record": "設定記録ファイル", "metadata": "チェックポイントのメタデータ", "none": "記録なし"}.get(settings.get("source"), "未対応")
    status = {"recorded": "初期化後の設定", "partial": "一部のみ", "error": "読み取りエラー", "ambiguous": "対象未確定", "mismatch": "対応不一致", "unavailable": "未取得"}.get(settings.get("status"), "未対応")
    body = "".join(f"<p class='sub'>{esc(note)}</p>" for note in settings.get("notes", []))
    values = settings.get("values") or {}
    if values:
        major = {key: values[key] for key in LABELS if key in values}
        if major:
            body += rows(major, LABELS)
        body += "<p class='sub'>Noneは指定なし、未記録は不明です。学習中の変化はグラフの記録値で確認できます。</p>"
        resolved = settings.get("resolved") or {}
        groups = resolved.get("optimizer_groups_created")
        if isinstance(groups, list):
            group_rows = []
            for group in groups:
                options = group.get("options", {})
                group_rows.append(f"<tr><td>{esc(group.get('label', group.get('index', '')))}</td><td>{esc(_value(options.get('lr'))) if 'lr' in options else '未記録'}</td></tr>")
            body += "<h3>Optimizer作成時の学習率</h3><div class='table-scroll'><table><thead><tr><th>グループ</th><th>LR</th></tr></thead><tbody>" + "".join(group_rows) + "</tbody></table></div>"
            body += "<p class='sub'>Scheduler適用・resume復元前の値です。開始時の値は下の確定値に収録しています。</p>"
        body += fold("すべての引数・記録項目", rows(values))
        if resolved:
            body += fold("初期化で確定した値・グループ設定", rows({k: v for k, v in resolved.items() if k not in ("args", "schema_version", "run_id", "metadata")}))
            body += fold("データセット・モデル等の記録", rows(resolved.get("metadata", {})))
        if settings.get("requested_args") is not None:
            body += fold("初期化前の引数（CLI・設定ファイル・既定値の統合後）", rows(settings["requested_args"]))
    provenance = {key: settings[key] for key in ("source_path", "run_id", "association", "candidates", "unreadable_records", "error") if key in settings}
    provenance["checkpoint"] = settings.get("checkpoint", {})
    body += fold("取得元・チェックポイント情報", rows(provenance))
    return "<section class='panel' id='training-settings'>" + fold(f"学習設定 · {source} · {status}", body) + "</section>"
