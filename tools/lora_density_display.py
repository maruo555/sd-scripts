from html import escape
import json
import math
from typing import Any, Dict, List, Optional, Tuple

from tools.analyze_lora_density import module_label, format_unet_block_label, format_text_encoder_block_label


def build_html_report(data: Dict[str, Any]) -> str:
    report_json = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    generated_at = escape(str(data.get('generated_at', '')))
    model_path = escape(str(data.get('model_path', '')))
    summary = data.get('summary', {})
    module_summary = data.get('module_summary', [])
    module_rankings = data.get('module_rankings', [])
    unet_block_summary = data.get('unet_block_summary', [])
    te1_block_summary = data.get('te1_block_summary', [])
    te2_block_summary = data.get('te2_block_summary', [])
    sparse_blocks = data.get('sparse_blocks', [])
    blocks = data.get('blocks', [])
    settings = data.get('settings', {})
    checkpoint_history = data.get('checkpoint_history', {})

    module_density_bounds: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    for entry in module_summary:
        density_stats = entry.get('density', {}) or {}
        module_density_bounds[entry.get('module', '-') ] = (
            density_stats.get('min'),
            density_stats.get('max'),
        )

    def format_float(val: Any, precision: int = 4) -> str:
        if val is None:
            return '-'
        if isinstance(val, (int, float)):
            if not math.isfinite(val):
                return str(val)
            return f"{val:.{precision}f}"
        return str(val)

    def format_int(val: Any) -> str:
        if val is None:
            return '-'
        if isinstance(val, (int, float)):
            return f"{int(val):,}"
        return str(val)

    def density_color(norm: Optional[float]) -> str:
        if norm is None:
            return ''
        norm = max(0.0, min(1.0, norm))
        low = (252, 165, 165)    # red-300
        mid = (255, 255, 255)    # white
        high = (74, 222, 128)    # green-400

        def blend(color_a, color_b, t: float) -> Tuple[int, int, int]:
            return tuple(int(round(a + (b - a) * t)) for a, b in zip(color_a, color_b))

        if norm < 0.5:
            ratio = norm / 0.5
            rgb = blend(low, mid, ratio)
        else:
            ratio = (norm - 0.5) / 0.5
            rgb = blend(mid, high, ratio)
        return '#%02x%02x%02x' % rgb

    def render_cards(cards: List[Tuple[str, str, str]]) -> str:
        parts: List[str] = []
        for title, value, subtitle in cards:
            parts.append(
                f'<div class="card"><h3>{title}</h3><p class="value">{value}</p><p class="sub">{subtitle}</p></div>'
            )
        return ''.join(parts)

    def render_module_table(items: List[Dict[str, Any]]) -> str:
        rows: List[str] = []
        for item in items:
            density = item.get('density', {}) or {}
            rms = item.get('rms', {}) or {}
            entropy = item.get('entropy_norm', {}) or {}
            cells = (
                f"<td>{module_label(item.get('module', '-'))}</td>"
                f'<td class="num">{format_int(item.get("block_count"))}</td>'
                f'<td class="num">{format_int(item.get("total_params"))}</td>'
                f'<td class="num">{format_float(density.get("mean"))}</td>'
                f'<td class="num">{format_float(density.get("median"))}</td>'
                f'<td class="num">{format_float(rms.get("median"))}</td>'
                f'<td class="num">{format_float(entropy.get("median"))}</td>'
            )
            rows.append(f'<tr>{cells}</tr>')
        if not rows:
            rows.append("<tr><td colspan='7' class='empty'>データがありません</td></tr>")
        return ''.join(rows)

    def render_block_table(
        items: List[Dict[str, Any]],
        columns: List[Tuple[str, str]],
        module_bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
        highlight_density: bool = False,
    ) -> str:
        header = ''.join(f'<th>{label}</th>' for _, label in columns)
        rows: List[str] = []
        for item in items:
            row_style = ''
            if highlight_density and module_bounds:
                norm = item.get('module_density_norm')
                if norm is None:
                    bounds = module_bounds.get(item.get('module')) if module_bounds else None
                    if bounds:
                        low, high = bounds
                        value = item.get('density_score')
                        if value is not None and low is not None and high is not None:
                            if math.isclose(high, low):
                                norm = 0.5
                            else:
                                norm = (value - low) / (high - low)
                if norm is not None:
                    color = density_color(norm)
                    if color:
                        row_style = f' style="background-color: {color};"'
            cells: List[str] = []
            for key, _ in columns:
                value = item.get(key)
                if key in {'total_params', 'tensor_count'}:
                    cells.append(f"<td class='num'>{format_int(value)}</td>")
                elif key in {
                    'density_score',
                    'rms',
                    'entropy_norm',
                    'non_zero_ratio',
                    'svd_top_energy_ratio_max',
                    'svd_top_energy_ratio_mean',
                    'svd_effective_rank_mean',
                    'mean',
                    'std',
                    'skewness',
                    'kurtosis',
                    'alpha',
                }:
                    cells.append(f"<td class='num'>{format_float(value)}</td>")
                else:
                    display = '-' if value is None else value
                    cells.append(f"<td>{escape(str(display))}</td>")
            rows.append(f"<tr{row_style}>" + ''.join(cells) + "</tr>")
        if not rows:
            rows.append(f"<tr><td colspan='{len(columns)}' class='empty'>データがありません</td></tr>")
        return (
            '<table>'
            '<thead><tr>' + header + '</tr></thead>'
            '<tbody>' + ''.join(rows) + '</tbody>'
            '</table>'
        )

    def render_unet_block_table(items: List[Dict[str, Any]]) -> str:
        if not items:
            return "<p class='empty'>UNetブロックの統計はありません。</p>"
        rows: List[str] = []
        for item in items:
            density = item.get('density', {}) or {}
            rms = item.get('rms', {}) or {}
            cells = (
                f"<td>{format_unet_block_label(item.get('label', '-'))}</td>"
                f"<td class='num'>{format_int(item.get('block_count'))}</td>"
                f"<td class='num'>{format_int(item.get('total_params'))}</td>"
                f"<td class='num'>{format_float(density.get('mean'))}</td>"
                f"<td class='num'>{format_float(density.get('median'))}</td>"
                f"<td class='num'>{format_float(rms.get('median'))}</td>"
            )
            rows.append(f"<tr>{cells}</tr>")
        return (
            "<table>"
            "<thead><tr>"
            "<th>UNetブロック</th>"
            "<th class='num'>LoRA数</th>"
            "<th class='num'>総パラメータ</th>"
            "<th class='num'>情報密度平均</th>"
            "<th class='num'>情報密度中央値</th>"
            "<th class='num'>RMS中央値</th>"
            "</tr></thead>"
            "<tbody>"
            + ''.join(rows)
            + "</tbody></table>"
        )

    def render_text_encoder_block_table(items: List[Dict[str, Any]], module: str) -> str:
        if not items:
            return f"<p class='empty'>{module_label(module)}のブロック統計はありません。</p>"
        rows: List[str] = []
        for item in items:
            density = item.get('density', {}) or {}
            rms = item.get('rms', {}) or {}
            cells = (
                f"<td>{format_text_encoder_block_label(item.get('label', '-'))}</td>"
                f"<td class='num'>{format_int(item.get('block_count'))}</td>"
                f"<td class='num'>{format_int(item.get('total_params'))}</td>"
                f"<td class='num'>{format_float(density.get('mean'))}</td>"
                f"<td class='num'>{format_float(density.get('median'))}</td>"
                f"<td class='num'>{format_float(rms.get('median'))}</td>"
            )
            rows.append(f"<tr>{cells}</tr>")
        heading = "TE1" if module == "te1" else "TE2" if module == "te2" else module.upper()
        return (
            "<table>"
            "<thead><tr>"
            f"<th>{heading}ブロック</th>"
            "<th class='num'>LoRA数</th>"
            "<th class='num'>総パラメータ</th>"
            "<th class='num'>情報密度平均</th>"
            "<th class='num'>情報密度中央値</th>"
            "<th class='num'>RMS中央値</th>"
            "</tr></thead>"
            "<tbody>"
            + ''.join(rows)
            + "</tbody></table>"
        )

    density_stats = summary.get('density', {}) or {}
    rms_stats = summary.get('rms', {}) or {}
    entropy_stats = summary.get('entropy_norm', {}) or {}

    overview_cards = [
        ('総ブロック数', format_int(summary.get('total_blocks')), f"総パラメータ: {format_int(summary.get('total_params'))}"),
        ('情報密度中央値', format_float(density_stats.get('median')), f"範囲 {format_float(density_stats.get('min'))} – {format_float(density_stats.get('max'))}"),
        ('RMS中央値', format_float(rms_stats.get('median')), f"範囲 {format_float(rms_stats.get('min'))} – {format_float(rms_stats.get('max'))}"),
        ('エントロピー中央値', format_float(entropy_stats.get('median')), f"範囲 {format_float(entropy_stats.get('min'))} – {format_float(entropy_stats.get('max'))}"),
    ]

    block_columns = [
        ('name', 'ブロック'),
        ('density_score', '情報密度'),
        ('rms', 'RMS'),
        ('entropy_norm', 'エントロピー'),
        ('non_zero_ratio', '非ゼロ率'),
        ('alpha', 'α'),
        ('total_params', 'パラメータ数'),
        ('svd_top_energy_ratio_max', 'SVDトップ比'),
    ]
    shortlist_columns = block_columns[:5]

    module_sections: List[str] = []
    for entry in module_rankings:
        module = entry.get('module', '-')
        label = module_label(module)
        top_blocks = entry.get('top', [])
        bottom_blocks = entry.get('bottom', [])
        top_html = render_block_table(top_blocks, shortlist_columns, module_density_bounds)
        bottom_html = render_block_table(bottom_blocks, shortlist_columns, module_density_bounds)
        module_sections.append(
            f"""
            <article class='module-ranking'>
                <h3>{label}</h3>
                <div class='module-tables'>
                    <div>
                        <h4>上位 {entry.get('top_count', len(top_blocks))} ブロック</h4>
                        {top_html}
                    </div>
                    <div>
                        <h4>下位 {entry.get('bottom_count', len(bottom_blocks))} ブロック</h4>
                        {bottom_html}
                    </div>
                </div>
            </article>
            """
        )
    module_rankings_html = ''.join(module_sections) if module_sections else "<p class='empty'>データがありません</p>"

    if sparse_blocks:
        sparse_table = render_block_table(
            sparse_blocks,
            block_columns,
            module_density_bounds,
            highlight_density=True,
        )
    else:
        sparse_table = "<p>しきい値未満のスパースブロックはありません。</p>"

    all_blocks_table = render_block_table(
        blocks,
        block_columns,
        module_density_bounds,
        highlight_density=True,
    )

    html = f"""<!DOCTYPE html>
<html lang='ja'>
<head>
    <meta charset='utf-8'>
    <title>LoRA 情報密度レポート</title>
    <style>
        body {{
            font-family: "Segoe UI", "Hiragino Kaku Gothic ProN", Meiryo, sans-serif;
            margin: 0 auto;
            max-width: 1080px;
            padding: 0 1.5rem 4rem;
            background-color: #f8fafc;
            color: #1f2937;
        }}
        header.page-header {{
            padding: 1.5rem 0;
            border-bottom: 1px solid #d9dde5;
            margin-bottom: 1.5rem;
        }}
        h1 {{
            margin: 0 0 0.5rem;
        }}
        .meta {{
            color: #4b5563;
            font-size: 0.95rem;
        }}
        .cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .card {{
            background: #ffffff;
            padding: 1rem 1.2rem;
            border-radius: 12px;
            box-shadow: 0 6px 16px rgba(15, 23, 42, 0.08);
        }}
        .card h3 {{
            margin: 0;
            font-size: 0.95rem;
            color: #4b5563;
        }}
        .card p.value {{
            margin: 0.4rem 0 0.2rem;
            font-size: 1.6rem;
            font-weight: 600;
            color: #1f2937;
        }}
        .card p.sub {{
            margin: 0;
            color: #6b7280;
            font-size: 0.9rem;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            background: #ffffff;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
            margin-bottom: 1.5rem;
        }}
        th, td {{
            border-bottom: 1px solid #e5e7eb;
            padding: 0.45rem 0.6rem;
            text-align: left;
        }}
        th {{
            font-weight: 600;
            color: #374151;
        }}
        td.num, th.num {{
            text-align: right;
            font-variant-numeric: tabular-nums;
        }}
        section {{
            margin-bottom: 2.5rem;
        }}
        section > h2 {{
            margin-bottom: 0.75rem;
            color: #111827;
        }}
        .module-ranking {{
            background: #ffffff;
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
            margin-bottom: 1.2rem;
        }}
        .module-ranking h3 {{
            margin: 0 0 0.6rem;
            color: #1f2937;
        }}
        .module-tables {{
            display: grid;
            gap: 1rem;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        }}
        .module-tables > div {{
            background: #f9fafb;
            border-radius: 10px;
            padding: 0.6rem;
            box-shadow: inset 0 0 0 1px #e5e7eb;
            overflow-x: auto;
        }}
        .module-tables table {{
            box-shadow: none;
            margin-bottom: 0;
        }}
        .module-tables table th, .module-tables table td {{
            border-bottom: 1px solid #e5e7eb;
        }}
        .module-tables h4 {{
            margin: 0 0 0.5rem;
            font-size: 1rem;
            color: #1f2937;
        }}
        #checkpoint-history {{
            margin-top: 2.5rem;
        }}
        #history-meta {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }}
        .epoch-tag {{
            display: inline-flex;
            align-items: center;
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            background: #e0e7ff;
            color: #3730a3;
            font-size: 0.85rem;
            font-weight: 600;
        }}
        .history-module {{
            background: #ffffff;
            border-radius: 12px;
            padding: 1rem 1.2rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
        }}
        .history-module h3 {{
            margin: 0 0 0.7rem;
            font-size: 1.1rem;
            color: #1f2937;
        }}
        .history-metric {{
            margin: -0.2rem 0 0.6rem;
            color: #4b5563;
            font-size: 0.9rem;
        }}
        .history-chart {{
            margin-bottom: 0.8rem;
            background: #f9fafb;
            border-radius: 10px;
            border: 1px solid #e5e7eb;
            padding: 0.5rem;
        }}
        .history-chart canvas {{
            width: 100%;
            height: 260px;
            display: block;
        }}
        .history-selected {{
            margin: 0 0 0.4rem;
            font-weight: 600;
            color: #111827;
        }}
        .history-values {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin: 0 0 0.8rem;
            font-size: 0.85rem;
            color: #4b5563;
        }}
        .history-values span {{
            background: #eef2ff;
            border-radius: 8px;
            padding: 0.2rem 0.5rem;
        }}
        .history-table-wrapper {{
            overflow-x: auto;
        }}
        table.history-table {{
            box-shadow: none;
            margin-bottom: 0;
        }}
        table.history-table thead th {{
            font-size: 0.85rem;
            color: #4b5563;
        }}
        table.history-table tbody td {{
            font-size: 0.86rem;
        }}
        table.history-table tbody tr {{
            cursor: pointer;
            transition: background-color 0.15s ease;
        }}
        table.history-table tbody tr:hover {{
            background: #f3f4f6;
        }}
        table.history-table tbody tr.selected {{
            background: #dbeafe;
        }}
        .history-delta {{
            font-weight: 600;
        }}
        .history-delta.positive {{
            color: #047857;
        }}
        .history-delta.negative {{
            color: #b91c1c;
        }}
        .empty {{
            text-align: center;
            padding: 1rem 0;
            color: #6b7280;
        }}
        footer {{
            margin-top: 3rem;
            color: #6b7280;
            font-size: 0.85rem;
            text-align: center;
        }}
    </style>
</head>
<body>
    <header class='page-header'>
        <h1>LoRA 情報密度レポート</h1>
        <p class='meta'>モデル: {model_path} | 生成日時: {generated_at}</p>
        <p class='meta'>ヒストグラムビン数: {settings.get('bins')} | 全体Top/Bottom: {settings.get('top_n')} | モジュール別Top/Bottom: {settings.get('module_top_n')}</p>
    </header>
    <section id='overview'>
        <h2>サマリー</h2>
        <div class='cards'>
            {render_cards(overview_cards)}
        </div>
    </section>
    <section id='modules'>
        <h2>モジュール別統計</h2>
        <table>
            <thead>
                <tr>
                    <th>モジュール</th>
                    <th class='num'>ブロック数</th>
                    <th class='num'>総パラメータ</th>
                    <th class='num'>情報密度平均</th>
                    <th class='num'>情報密度中央値</th>
                    <th class='num'>RMS中央値</th>
                    <th class='num'>エントロピー中央値</th>
                </tr>
            </thead>
            <tbody>
                {render_module_table(module_summary)}
            </tbody>
        </table>
    </section>
    <section id='unet-blocks'>
        <h2>UNetブロック別概要</h2>
        {render_unet_block_table(unet_block_summary)}
    </section>
    <section id='te1-blocks'>
        <h2>TE1ブロック別概要</h2>
        {render_text_encoder_block_table(te1_block_summary, 'te1')}
    </section>
    <section id='te2-blocks'>
        <h2>TE2ブロック別概要</h2>
        {render_text_encoder_block_table(te2_block_summary, 'te2')}
    </section>
    <section id='checkpoint-history'>
        <h2>エポック推移</h2>
        <div id='history-meta'></div>
        <div id='history-content'></div>
    </section>
    <section id='module-rankings'>
        <h2>モジュール別 上位/下位ブロック</h2>
        {module_rankings_html}
    </section>
    <section id='sparse-blocks'>
        <h2>スパースブロック（非ゼロ率 &lt; {format_float(settings.get('sparse_threshold'))}）</h2>
        {sparse_table}
    </section>
    <section id='all-blocks'>
        <h2>全ブロック一覧（主要指標）</h2>
        {all_blocks_table}
    </section>
    <footer>
        <p>このレポートは sd-scripts の analyze_lora_density ツールで生成されました。</p>
    </footer>

<script>
    window.__LORA_DENSITY_REPORT__ = {report_json};
    (function() {{
        const report = window.__LORA_DENSITY_REPORT__;
        if (!report) {{
            return;
        }}
        const history = report.checkpoint_history || {{}};
        const files = Array.isArray(history.files)
            ? history.files.slice().sort((a, b) => (a.epoch || 0) - (b.epoch || 0))
            : [];
        const container = document.getElementById('history-content');
        const meta = document.getElementById('history-meta');
        if (!container) {{
            return;
        }}
        if (files.length <= 1) {{
            container.innerHTML = "<p class='empty'>途中のチェックポイントは見つかりませんでした。</p>";
            if (meta) {{
                meta.innerHTML = '';
            }}
            return;
        }}
        if (meta) {{
            meta.innerHTML = files
                .map((file) => `<span class="epoch-tag">${{file.label}}</span>`)
                .join('');
        }}

        const epochIndex = new Map();
        const epochLabels = new Map();
        files.forEach((file, idx) => {{
            epochIndex.set(file.epoch, idx);
            epochLabels.set(file.epoch, file.label);
        }});

        const COLOR_PALETTE = ['#2563eb', '#f97316', '#10b981', '#ec4899', '#6366f1', '#14b8a6', '#f59e0b', '#8b5cf6', '#ef4444', '#0ea5e9'];

        function createColorMap(seriesList) {{
            const map = new Map();
            if (!Array.isArray(seriesList)) {{
                return map;
            }}
            seriesList.forEach((entry, idx) => {{
                const key = entry.label || entry.display_label || `series_${{idx}}`;
                entry._colorKey = key;
                const color = COLOR_PALETTE[idx % COLOR_PALETTE.length];
                map.set(key, color);
            }});
            return map;
        }}

        function applyOpacity(hex, alpha) {{
            const rgb = hexToRgb(hex);
            return `rgba(${{rgb.r}}, ${{rgb.g}}, ${{rgb.b}}, ${{alpha}})`;
        }}

        function hexToRgb(hex) {{
            let normalized = hex ? hex.toString().trim() : '#000000';
            if (normalized.startsWith('#')) {{
                normalized = normalized.slice(1);
            }}
            if (normalized.length === 3) {{
                normalized = normalized.split('').map((ch) => ch + ch).join('');
            }}
            const intVal = parseInt(normalized, 16);
            if (Number.isNaN(intVal)) {{
                return {{ r: 37, g: 99, b: 235 }};
            }}
            return {{
                r: (intVal >> 16) & 255,
                g: (intVal >> 8) & 255,
                b: intVal & 255,
            }};
        }}



        const moduleLabels = history.module_labels || {{}};
        const seriesByModule = history.series || {{}};
        const modulesInOrder = ['unet', 'te1', 'te2'];
        let rendered = false;

        modulesInOrder.forEach((moduleKey) => {{
            const seriesList = seriesByModule[moduleKey];
            if (!Array.isArray(seriesList) || seriesList.length === 0) {{
                return;
            }}
            rendered = true;

            const section = document.createElement('article');
            section.className = 'history-module';

            const title = document.createElement('h3');
            title.textContent = moduleLabels[moduleKey] || moduleKey.toUpperCase();
            section.appendChild(title);

            const metricLabel = document.createElement('p');
            metricLabel.className = 'history-metric';
            metricLabel.textContent = '情報密度平均の推移';
            section.appendChild(metricLabel);

            const chartWrapper = document.createElement('div');
            chartWrapper.className = 'history-chart';
            const canvas = document.createElement('canvas');
            chartWrapper.appendChild(canvas);
            section.appendChild(chartWrapper);

            const selectionLabel = document.createElement('p');
            selectionLabel.className = 'history-selected';
            section.appendChild(selectionLabel);

            const valueList = document.createElement('div');
            valueList.className = 'history-values';
            section.appendChild(valueList);

            const tableWrapper = document.createElement('div');
            tableWrapper.className = 'history-table-wrapper';
            const table = document.createElement('table');
            table.className = 'history-table';
            const thead = document.createElement('thead');
            thead.innerHTML = '<tr><th>ブロック</th><th class="num">開始</th><th class="num">最新</th><th class="num">変化量</th><th class="num">データ数</th></tr>';
            table.appendChild(thead);
            const tbody = document.createElement('tbody');
            table.appendChild(tbody);
            tableWrapper.appendChild(table);
            section.appendChild(tableWrapper);

            container.appendChild(section);

            const colorMap = createColorMap(seriesList);
            const state = {{
                canvas,
                selectionLabel,
                valueList,
                tbody,
                files,
                epochLabels,
                epochIndex,
                seriesList,
                colorMap,
                selectedRow: null,
            }};

            seriesList.forEach((entry) => {{
                const row = document.createElement('tr');
                row.dataset.displayLabel = entry.display_label || entry.label || '-';
                row._seriesEntry = entry;
                row._seriesValues = Array.isArray(entry.values) ? entry.values : [];

                const labelCell = document.createElement('td');
                labelCell.textContent = row.dataset.displayLabel;
                row.appendChild(labelCell);

                const startCell = document.createElement('td');
                startCell.className = 'num';
                startCell.textContent = formatDensity(entry.start_density);
                row.appendChild(startCell);

                const endCell = document.createElement('td');
                endCell.className = 'num';
                endCell.textContent = formatDensity(entry.end_density);
                row.appendChild(endCell);

                const deltaCell = document.createElement('td');
                deltaCell.className = 'num';
                const deltaSpan = document.createElement('span');
                deltaSpan.textContent = formatDelta(entry.delta_density);
                deltaSpan.className = 'history-delta';
                if (typeof entry.delta_density === 'number') {{
                    if (entry.delta_density > 0) {{
                        deltaSpan.classList.add('positive');
                    }} else if (entry.delta_density < 0) {{
                        deltaSpan.classList.add('negative');
                    }}
                }}
                deltaCell.appendChild(deltaSpan);
                row.appendChild(deltaCell);

                const countCell = document.createElement('td');
                countCell.className = 'num';
                const count = entry.values ? entry.values.length : 0;
                countCell.textContent = count ? String(count) : '-';
                row.appendChild(countCell);

                tbody.appendChild(row);
            }});

            tbody.addEventListener('click', (event) => {{
                const row = event.target.closest('tr');
                if (!row) {{
                    return;
                }}
                selectRow(row, state);
            }});

            state.selectionLabel.textContent = '選択中: なし（情報密度平均 全系列表示）';
            renderHistoryChart(state, null);
            renderValueList(state, null);
        }});

        if (!rendered) {{
            container.innerHTML = "<p class='empty'>チェックポイント統計を生成できませんでした。</p>";
        }}

        function formatDensity(value) {{
            if (typeof value !== 'number' || !Number.isFinite(value)) {{
                return '-';
            }}
            return value.toFixed(4);
        }}

        function formatDelta(value) {{
            if (typeof value !== 'number' || !Number.isFinite(value)) {{
                return '-';
            }}
            const sign = value > 0 ? '+' : value < 0 ? '' : '';
            return sign + value.toFixed(4);
        }}

        function selectRow(row, state) {{
            if (state.selectedRow === row) {{
                row.classList.remove('selected');
                state.selectedRow = null;
                state.selectionLabel.textContent = '選択中: なし（情報密度平均 全系列表示）';
                renderHistoryChart(state, null);
                renderValueList(state, null);
                return;
            }}
            if (state.selectedRow) {{
                state.selectedRow.classList.remove('selected');
            }}
            state.selectedRow = row;
            row.classList.add('selected');
            const entry = row._seriesEntry || null;
            state.selectionLabel.textContent = `選択中: ${{row.dataset.displayLabel || '-'}}（情報密度平均）`;
            renderHistoryChart(state, entry);
            renderValueList(state, entry);
        }}

        function renderValueList(state, entry) {{
            const container = state.valueList;
            if (!container) {{
                return;
            }}
            if (!entry) {{
                container.innerHTML = '<span>情報密度平均の全系列を表示中です。表から系列を選択すると詳細を表示します。</span>';
                return;
            }}
            const values = Array.isArray(entry.values) ? entry.values : [];
            if (!values.length) {{
                container.innerHTML = '<span>データがありません</span>';
                return;
            }}
            container.innerHTML = values
                .map((point) => {{
                    const label = state.epochLabels.get(point.epoch) || `Epoch ${{String(point.epoch).padStart(2, '0')}}`;
                    const valueText = formatDensity(point.density_mean);
                    return `<span>${{label}}: ${{valueText}}</span>`;
                }})
                .join('');
        }}

        function prepareCanvas(canvas) {{
            const dpr = window.devicePixelRatio || 1;
            const rect = canvas.getBoundingClientRect();
            const width = rect.width || 640;
            const height = rect.height || 260;
            const scaledWidth = width * dpr;
            const scaledHeight = height * dpr;
            if (canvas.width !== scaledWidth || canvas.height !== scaledHeight) {{
                canvas.width = scaledWidth;
                canvas.height = scaledHeight;
            }}
            const ctx = canvas.getContext('2d');
            ctx.setTransform(1, 0, 0, 1, 0, 0);
            ctx.scale(dpr, dpr);
            ctx.clearRect(0, 0, width, height);
            return {{ ctx, width, height }};
        }}

        function renderHistoryChart(state, selectedEntry) {{
            const {{ canvas, files, seriesList, colorMap }} = state;
            const {{ ctx, width, height }} = prepareCanvas(canvas);
            const margin = {{ top: 20, right: 20, bottom: 40, left: 60 }};
            const chartWidth = Math.max(width - margin.left - margin.right, 10);
            const chartHeight = Math.max(height - margin.top - margin.bottom, 10);

            ctx.fillStyle = '#f9fafb';
            ctx.fillRect(0, 0, width, height);

            ctx.strokeStyle = '#d1d5db';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(margin.left, margin.top);
            ctx.lineTo(margin.left, margin.top + chartHeight);
            ctx.lineTo(margin.left + chartWidth, margin.top + chartHeight);
            ctx.stroke();

            const allNumericPoints = [];
            seriesList.forEach((entry) => {{
                const values = Array.isArray(entry.values) ? entry.values : [];
                values.forEach((point) => {{
                    if (typeof point.density_mean === 'number' && Number.isFinite(point.density_mean)) {{
                        allNumericPoints.push(point.density_mean);
                    }}
                }});
            }});

            if (!allNumericPoints.length) {{
                ctx.fillStyle = '#6b7280';
                ctx.font = '14px sans-serif';
                ctx.fillText('有効なデータがありません', margin.left + 10, margin.top + chartHeight / 2);
                return;
            }}

            let minVal = Math.min(...allNumericPoints);
            let maxVal = Math.max(...allNumericPoints);
            if (Math.abs(maxVal - minVal) < 1e-9) {{
                const delta = Math.abs(maxVal) * 0.05 + 1e-4;
                minVal -= delta;
                maxVal += delta;
            }}

            const epochPositions = new Map();
            const fileCount = files.length;
            files.forEach((file, idx) => {{
                const ratio = fileCount > 1 ? idx / (fileCount - 1) : 0.5;
                const x = margin.left + ratio * chartWidth;
                epochPositions.set(file.epoch, x);
                ctx.beginPath();
                ctx.moveTo(x, margin.top + chartHeight);
                ctx.lineTo(x, margin.top + chartHeight + 6);
                ctx.stroke();
                if (idx === 0 || idx === fileCount - 1) {{
                    ctx.fillStyle = '#4b5563';
                    ctx.font = '12px sans-serif';
                    ctx.fillText(file.label, x - 30, margin.top + chartHeight + 22, 60);
                }}
            }});

            const scaleY = chartHeight / (maxVal - minVal);

            seriesList.forEach((entry) => {{
                const values = Array.isArray(entry.values) ? entry.values : [];
                const numericValues = values.filter(
                    (point) => typeof point.density_mean === 'number' && Number.isFinite(point.density_mean)
                );
                if (!numericValues.length) {{
                    return;
                }}

                const key = entry._colorKey || entry.label || entry.display_label || `series_${{seriesList.indexOf(entry)}}`;
                const baseColor = colorMap.get(key) || '#2563eb';
                const isSelected = !!(selectedEntry && entry === selectedEntry);
                const strokeColor = selectedEntry
                    ? (isSelected ? baseColor : applyOpacity(baseColor, 0.35))
                    : baseColor;
                const pointColor = selectedEntry
                    ? (isSelected ? baseColor : applyOpacity(baseColor, 0.45))
                    : baseColor;
                const lineWidth = selectedEntry
                    ? (isSelected ? 2.4 : 1.2)
                    : 1.8;
                const pointRadius = selectedEntry
                    ? (isSelected ? 4.2 : 3.0)
                    : 3.5;

                ctx.strokeStyle = strokeColor;
                ctx.lineWidth = lineWidth;
                ctx.beginPath();
                let moved = false;
                numericValues.forEach((point) => {{
                    const x = epochPositions.get(point.epoch);
                    if (typeof x !== 'number') {{
                        return;
                    }}
                    const y = margin.top + chartHeight - (point.density_mean - minVal) * scaleY;
                    if (!moved) {{
                        ctx.moveTo(x, y);
                        moved = true;
                    }} else {{
                        ctx.lineTo(x, y);
                    }}
                }});
                ctx.stroke();

                ctx.fillStyle = pointColor;
                numericValues.forEach((point) => {{
                    const x = epochPositions.get(point.epoch);
                    if (typeof x !== 'number') {{
                        return;
                    }}
                    const y = margin.top + chartHeight - (point.density_mean - minVal) * scaleY;
                    ctx.beginPath();
                    ctx.arc(x, y, pointRadius, 0, Math.PI * 2);
                    ctx.fill();
                }});
            }});

            ctx.fillStyle = '#4b5563';
            ctx.font = '12px sans-serif';
            ctx.fillText(formatDensity(maxVal), margin.left - 50, margin.top + 6);
            ctx.fillText(formatDensity(minVal), margin.left - 50, margin.top + chartHeight);
        }}

    }})();
</script>

</body>
</html>
"""
    return html
