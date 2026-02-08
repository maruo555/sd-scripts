# `make_lora_diagnostic_report.py` 使い方

## 概要
`tools/make_lora_diagnostic_report.py` は、LoRA学習時のログとチェックポイントをまとめて解析し、診断用のHTML/JSONレポートを生成するツールです。

- `grad_norm` ログ可視化
- `dq_delta` ログ可視化
- `group_loss` ログ可視化（任意）
- LoRA重みの統計（任意）
- LoRA情報密度のエポック推移（任意）

出力先は、`--output_dir` 未指定時に `--input_dir/diagnostic_report` です。

## `--input_dir` 内で使うログファイル名（固定）
このツールは、`--input_dir` 配下の以下ファイル名を前提に読み込みます。

- `gradient_logs+<base_name>.txt`
- `dq_delta_logs+<base_name>.txt`
- `dq_delta_auto+<base_name>.txt`
- `group_loss_logs+<base_name>.csv`（任意）
- `group_loss_epoch+<base_name>.csv`（任意）

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

`group_loss_logs` / `group_loss_epoch` も任意です。存在する場合のみ `Group Loss Dashboard` が追加されます。  
- `group_loss_logs` がある場合: `global_step` をX軸に `ema_loss_group` をgroup別系列で表示  
- `group_loss_epoch` がある場合: `epoch` をX軸に `ema_loss_end` をgroup別系列で表示  
両方ある場合は、それぞれ別グラフとして表示します。

LoRAチェックポイント（`.safetensors`）が無い場合は、LoRA重み解析セクションのみ警告表示になり、レポート生成は続行します。
