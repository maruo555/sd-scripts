# SDXL LoRA 学習のおすすめ設定

単一GPU・fp16・rank 4で使っている5つの設定例です。DQ（Delta Quantization）は、学習中にLoRAが元の層に加える出力差分（Δ）を量子化する機能です。5例とも保存するLoRA重みはfp16です。

表は **1行が1つのCLI引数、1列が1つの学習例** です。横に読むと構成ごとの違いが分かり、縦に読むとその構成で指定する引数を拾えます。グループ名のリンクから詳しい説明に移れます。

## 5つの設定例で使う全64引数

**値**は実際に渡す値、**指定**は値を取らないフラグを付けること、**-** はその引数を省略することを表します。省略時はCLIの既定値または読み込んだ設定が適用されます。`--no-avg_reset_stats` の「指定」は、平均重みの採用時にoptimizerの統計をリセットしない指定です。

全例とも `sdxl_train_network.py` を使用します。`--num_cpu_threads_per_process` は起動側の `accelerate launch` に渡します。表中の「モデル」は `noobaiXLNAIXL_epsilonPred075.safetensors` のパス、「入力TOML」は手元のデータ設定ファイルのパス、「出力名」は各学習につける名前です。bucketの有効化と最小・最大解像度は、入力TOMLに同じ項目があればそちらの値を使います。

※ 全例で `AdamW8bitFast` を使います。「独自機能なし」で使う独自処理は、標準の `AdamW8bit` のGPU同期待ちをまとめる高速化だけです。パラメーターの更新計算は同じです。[optimizerの仕組み](maruo_training_features_ja.md#optimizer)

| グループ | CLI引数 | 独自機能なし※ | DQのみなし | DQ mul固定 | DQ A方式 | DQ mul自動 |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **[起動・入力](sdxl_train_network.md)** | `--num_cpu_threads_per_process` | `8` | `8` | `8` | `8` | `8` |
|  | `--pretrained_model_name_or_path` | モデル | モデル | モデル | モデル | モデル |
|  | `--dataset_config` | 入力TOML | 入力TOML | 入力TOML | 入力TOML | 入力TOML |
|  | `--max_data_loader_n_workers` | `1` | `1` | `1` | `1` | `1` |
| **[LoRA・dropout](train_network_README-ja.md)** | `--network_module` | `networks.lora` | `networks.lora` | `networks.lora` | `networks.lora` | `networks.lora` |
|  | `--network_dim` | `4` | `4` | `4` | `4` | `4` |
|  | `--network_dropout` | `0.3` | `0.3` | `0.3` | `0.3` | `0.3` |
|  | `--network_args` | `rank_dropout=0.2` | `rank_dropout=0.2` | `rank_dropout=0.2` | `rank_dropout=0.2` | `rank_dropout=0.2` |
| **[データ・bucket](config_README-ja.md)** | `--enable_bucket` | 指定 | 指定 | 指定 | 指定 | 指定 |
|  | `--min_bucket_reso` | `384` | `384` | `384` | `384` | `384` |
|  | `--max_bucket_reso` | `1024` | `1024` | `1024` | `1024` | `1024` |
|  | `--cache_latents` | 指定 | 指定 | 指定 | 指定 | 指定 |
| **[ノイズ・loss](train_README-ja.md)** | `--noise_offset` | `0.15` | `0.15` | `0.15` | `0.15` | `0.15` |
|  | `--adaptive_noise_scale` | `0.1` | `0.1` | `0.1` | `0.1` | `0.1` |
|  | `--prior_loss_weight` | `1.0` | `1.0` | `1.0` | `1.0` | `1.0` |
| **[学習期間・更新](maruo_training_features_ja.md#optimizer)** | `--max_train_epochs` | `40` | `40` | `40` | `40` | `40` |
|  | `--seed` | `39` | `39` | `39` | `39` | `39` |
|  | `--optimizer_type` | `AdamW8bitFast` | `AdamW8bitFast` | `AdamW8bitFast` | `AdamW8bitFast` | `AdamW8bitFast` |
|  | `--learning_rate` | `3.5e-4` | `3.5e-4` | `3.5e-4` | `3.5e-4` | `3.5e-4` |
|  | `--lr_scheduler` | - | `constant_with_warmup` | `constant_with_warmup` | `constant_with_warmup` | `constant_with_warmup` |
|  | `--lr_warmup_steps` | - | `0.05` | `0.05` | `0.05` | `0.05` |
| **[テキストエンコーダー](maruo_training_features_ja.md#encoders)** | `--text_encoder_lr` | `2e-4` | `2e-4` | `2e-4` | `2e-4` | `2e-4` |
|  | `--text_encoder_lr1` | - | `3e-4` | `3e-4` | `3e-4` | `3e-4` |
|  | `--text_encoder_lr2` | - | `2e-4` | `2e-4` | `2e-4` | `2e-4` |
|  | `--te_mlp_fc_only` | - | 指定 | 指定 | 指定 | 指定 |
| **[UNet・演算精度](maruo_training_features_ja.md#unet)** | `--sdpa` | 指定 | 指定 | 指定 | 指定 | 指定 |
|  | `--mixed_precision` | `fp16` | `fp16` | `fp16` | `fp16` | `fp16` |
|  | `--downscale_freq_shift` | - | 指定 | 指定 | 指定 | 指定 |
|  | `--fp16_safe_norms_mode` | - | - | `strict` | `strict` | `strict` |
| **[Guardian](maruo_training_features_ja.md#guardian)** | `--grad_norm_mode` | - | `stable_no_threshoff` | `stable_no_threshoff` | `stable_no_threshoff` | `stable_no_threshoff` |
| **[重みの平均・promote](maruo_training_features_ja.md#averaging)** | `--avg_cp` | - | 指定 | 指定 | 指定 | 指定 |
|  | `--avg_cp_mode` | - | `promote` | `promote` | `promote` | `promote` |
|  | `--avg_begin` | - | `0.6` | `0.6` | `0.6` | `0.6` |
|  | `--avg_window` | - | `4` | `4` | `4` | `4` |
|  | `--avg_mode` | - | `ema` | `ema` | `ema` | `ema` |
|  | `--avg_shadow_bank_size` | - | `12` | `12` | `12` | `12` |
|  | `--no-avg_reset_stats` | - | 指定 | 指定 | 指定 | 指定 |
|  | `--avg_save_final_raw` | - | 指定 | 指定 | 指定 | 指定 |
| **[DQ・量子化](maruo_training_features_ja.md#dq)** | `--dq_delta_bits` | - | - | `8` | `8` | `8` |
|  | `--dq_delta_granularity` | - | - | `channel` | `channel` | `channel` |
|  | `--dq_delta_stat` | - | - | `rms` | `rms` | `rms` |
|  | `--dq_delta_mode` | - | - | `stoch` | `stoch` | `stoch` |
|  | `--dq_delta_scope` | - | - | `both` | `both` | `both` |
|  | `--dq_delta_begin_after_lr_warmup` | - | - | 指定 | 指定 | 指定 |
| **[DQ・範囲倍率](maruo_training_features_ja.md#dq)** | `--dq_delta_range_mul` | - | - | `4.05` [※1](#fixed-mul-note) | `2.70` | `3.0` |
|  | `--dq_delta_range_mul_attn2` | - | - | - | `3.75` | - |
|  | `--dq_delta_range_mul_te` | - | - | - | `3.75` | - |
| **[DQ・自動倍率](maruo_training_features_ja.md#dq-auto)** | `--dq_delta_auto_range_mul` | - | - | - | - | 指定 |
|  | `--dq_delta_auto_preset` | - | - | - | - | `clip_rate_high` [※2](#auto-preset-note) |
|  | `--dq_delta_auto_init_range_mul_from_band` | - | - | - | - | 指定 |
|  | `--dq_delta_auto_use_raw` | - | - | - | - | 指定 |
| **[DQ・Triton](maruo_training_features_ja.md#triton)** | `--dq_delta_use_triton` | - | - | 指定 | 指定 | 指定 |
|  | `--dq_delta_triton_stats` | - | - | 指定 | 指定 | 指定 |
| **[学習ログ](maruo_training_features_ja.md#logging)** | `--rank_log` | - | 指定 | 指定 | 指定 | 指定 |
|  | `--rank_log_mode` | - | `per_module` | `per_module` | `per_module` | `per_module` |
|  | `--dq_delta_log` | - | - | 指定 | 指定 | 指定 |
|  | `--dq_delta_log_detail` | - | - | `basic` | `basic` | `basic` |
| **[出力・保存](maruo_training_features_ja.md#run-output)** | `--output_dir` | `..\lora_output` | `..\lora_output` | `..\lora_output` | `..\lora_output` | `..\lora_output` |
|  | `--output_name` | 出力名 | 出力名 | 出力名 | 出力名 | 出力名 |
|  | `--save_model_as` | `safetensors` | `safetensors` | `safetensors` | `safetensors` | `safetensors` |
|  | `--save_precision` | `fp16` | `fp16` | `fp16` | `fp16` | `fp16` |
|  | `--save_every_n_epochs` | `1` | `1` | `1` | `1` | `1` |
|  | `--training_comment` | `test` | `test` | `test` | `test` | `test` |
| **[終了時診断](maruo_tools_ja.md#report)** | `--report_after_training` | - | - | 指定 | 指定 | 指定 |

## 差し替えるときの目安

「DQのみなし」で他の3列と学習条件を揃える場合は、`--fp16_safe_norms_mode strict` を追加します。

<a id="fixed-mul-note"></a>

**※1　DQ mul固定**　`4.05` は一例です。手元では `--dq_delta_range_mul` を **2.70〜4.05程度**の範囲で変えて試しています。小さくするとclipが増えやすく、大きくするとclipは減りやすい一方で丸めの刻みが粗くなります。「DQ A方式」では基本を `2.70` に保ち、UNetのattn2と両TEだけ `3.75` に広げます。[量子化と倍率](maruo_training_features_ja.md#dq)

<a id="auto-preset-note"></a>

**※2　DQ mul自動**　`--dq_delta_auto_preset` の値を選んで、目標とするclip率を変えます。よく使うのは次のうち `clip_rate_high` と `clip_rate_low` です。

- `clip_rate_high`（表の例）なら目標は **0.3〜0.5%**。
- `clip_rate_low` なら目標は **0.05〜0.22%**。
- `clip_rate_low_auto` はlowの帯から始まり、学習の進捗と量子化誤差が条件を満たすと、一度だけmidの帯に移れます。

この列では `--dq_delta_auto_init_range_mul_from_band` を指定しているので、実際の初期倍率も選んだ目標帯から計算されます。[自動倍率の判定](maruo_training_features_ja.md#dq-auto)・[プリセット全一覧](maruo_cli_ja.md#training-options)

各引数の既定値と組み合わせの制約は[独自CLIリファレンス](maruo_cli_ja.md#training-options)にあります。Windowsで表のTriton経路を使う場合は、学習用venvに `requirements-triton-windows.txt` の依存パッケージを追加します。[Tritonの依存ファイル](../requirements-triton-windows.txt)
