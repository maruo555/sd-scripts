# `make_lora_diagnostic_report.py` 使い方

## 概要
`tools/make_lora_diagnostic_report.py` は、LoRA学習時のログとチェックポイントをまとめて解析し、診断用のHTML/JSONレポートを生成するツールです。

- `grad_norm` ログ可視化
- `dq_delta` ログ可視化
- LoRA重みの統計（任意）
- LoRA情報密度のエポック推移（任意）

出力先は、`--output_dir` 未指定時に `--input_dir/diagnostic_report` です。

## `--input_dir` 内で使うログファイル名（固定）
このツールは、`--input_dir` 配下の以下ファイル名を前提に読み込みます。

- `gradient_logs+<base_name>.txt`
- `dq_delta_logs+<base_name>.txt`
- `dq_delta_auto+<base_name>.txt`

LoRAチェックポイントは `--input_dir/<base_name>.safetensors` を使用します。  
`--lora_epoch_trend`オプション指定時は`--input_dir/<base_name>-000001.safetensors`などの途中のチェックポイントも使用します。

## オプション一覧
| オプション | 必須 | 既定値 | 説明 |
|---|---|---|---|
| `--base_name` | 必須 | なし | LoRAのベース名 |
| `--input_dir` | 任意 | `.` | ログ/チェックポイントのあるフォルダ |
| `--loss_ma_window` | 任意 | `100` | Loss移動平均の窓サイズ |
| `--lora_bins` | 任意 | `128` | LoRA解析ヒストグラムのビン数 |
| `--skip_lora_analysis` | 任意 | `False` | LoRA重み解析をスキップ |
| `--lora_epoch_trend` | 任意 | `False` | 情報密度のエポック推移解析を有効化 |
| `--output_dir` | 任意 | `<input_dir>/diagnostic_report` | 出力フォルダ |
| `--output_html` | 任意 | 自動生成 | HTML出力先を個別指定 |
| `--output_json` | 任意 | 自動生成 | JSON出力先を個別指定 |

## コマンド例
普段使いの想定コマンド:

```powershell
python tools\make_lora_diagnostic_report.py --base_name loraname --input_dir ..\lora_output --lora_epoch_trend
```

## ファイルが無い場合の動作
`gradient_logs` と `dq_delta_logs` は必須です。見つからない場合はエラー終了します。

- `gradient_logs+<base_name>.txt` が無い  
  `FileNotFoundError: grad log が見つかりません: ...`
- `dq_delta_logs+<base_name>.txt` が無い  
  `FileNotFoundError: dq log が見つかりません: ...`

`dq_delta_auto` は任意です。無い場合は自動でスキップして続行します。

- `dq_delta_auto+<base_name>.txt` が無い  
  auto系の解析だけ省略し、HTML/JSONは生成されます。

LoRAチェックポイント（`.safetensors`）が無い場合は、LoRA重み解析セクションのみ警告表示になり、レポート生成は続行します。
