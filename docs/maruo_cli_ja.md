# SDXL LoRA 独自CLIリファレンス

SDXL LoRAの学習・診断・生成・比較・評価で、どの入口に何を指定するかをまとめました。学習の独自引数49種とその否定形3種、本家引数に追加した使い方、同梱した独自ツールの公開引数を収録しています。ベースモデルや解像度など、本家から引き継いだ引数の全一覧は各スクリプトの `--help` で確認できます。設定の組み合わせを選ぶ場合は[SDXL LoRA 学習のおすすめ設定](maruo_features_ja.md)をご覧ください。

**このページの索引**　[独自学習引数49種](#training-options) ／ [本家引数への拡張・dataset設定](#extension-points) ／ [ツール別の公開CLI](#tool-options)

## 変更箇所の見取り図

| やりたいこと | 入口・目印 | 詳しい説明 |
| --- | --- | --- |
| TE1・TE2の学習層と学習率を分ける | `sdxl_train_network.py` の `--te_mlp_fc_only`、TE別LR | [TEの対象と学習率](maruo_training_features_ja.md#encoders) |
| UNetのfp16演算を安定させる、拡散時刻の埋め込みを変える | `--fp16_safe_norms_mode`、`--downscale_freq_shift` | [SDXL UNet](maruo_training_features_ja.md#unet) |
| optimizerを選び、勾配に応じて更新を見送る | `--optimizer_type`、`--grad_norm_mode` | [optimizer](maruo_training_features_ja.md#optimizer)・[Guardian](maruo_training_features_ja.md#guardian) |
| epoch末の重みを平均し、候補を採点する | `--avg_cp`、`--avg_cp_mode` | [平均とpromote](maruo_training_features_ja.md#averaging) |
| LoRAの出力差分（Δ）を量子化する | `--dq_delta_bits`、場所別倍率、bit schedule、自動倍率、CUDA高速経路 | [固定DQ](maruo_training_features_ja.md#dq)・[自動倍率](maruo_training_features_ja.md#dq-auto)・[Triton](maruo_training_features_ja.md#triton) |
| 学習を観測し、保存された重みと結び付ける | rank・group・DQログ、学習1回ごとのrun、途中のサンプルと保存 | [ログ](maruo_training_features_ja.md#logging)・[runの出力](maruo_training_features_ja.md#run-output)・[run control](maruo_training_features_ja.md#control) |
| 学習結果を読む・見比べる | 診断CLI、density CLI、診断比較HTML | [診断レポート](maruo_tools_ja.md#report)・[density](maruo_tools_ja.md#density)・[診断比較](maruo_tools_ja.md#viewer) |
| LoRAの強度やブロックを変えて生成・採点する | `sdxl_gen_img.py`、画像比較GUI/CUI、blindレポート | [画像生成](maruo_tools_ja.md#generate)・[画像比較とblind評価](maruo_tools_ja.md#compare) |
| 評価と履歴を管理する、タグの分割を調べる | 研究台帳CLI/GUI、tokenizer CLI | [研究台帳](maruo_tools_ja.md#ledger)・[tokenizer](maruo_tools_ja.md#tokenizer) |

<a id="training-options"></a>

## 学習CLIの全一覧

対象は `sdxl_train_network.py --network_module networks.lora`。表の「省略時」は、ほかの引数を指定して値が決まる場合も含む**実効値**です。フラグはCLIでは引数名だけを書きます。学習TOMLでは先頭の `--` を外したキーに数値・文字列・真偽値を設定し、同じ設定をCLIで明示するとCLIを優先します。実行ごとの入力・実効値はrun内の `training_settings.json` で確認できます。[学習の適用条件](maruo_training_features_ja.md#scope)・[引数定義と解決処理](../library/maruo_training_config.py)

学習挙動を変える独自設定では、1プロセス、勾配蓄積1、各datasetの実効 `batch_size=1`・`network_multiplier=1.0` が前提です。途中状態の `--resume`、TE出力のキャッシュ、inpainting、subsetの `diff_output_preservation` と組み合わせた学習は対象外です。観測だけを追加した構成の再開と、以前のLoRAを初期重みにする方法は[学習の適用条件](maruo_training_features_ja.md#scope)にあります。

### TEとUNet

| 引数 | 型・選択肢 | 省略時 | 効果・条件 |
| --- | --- | --- | --- |
| `--text_encoder_lr1` | 実数 ≥ 0 | 共通LRを継承 | TE1の学習率。明示した0はこのTEのoptimizer groupを省く。 |
| `--text_encoder_lr2` | 実数 ≥ 0 | 共通LRを継承 | TE2の学習率。接続済みTEのforward・DQの対象は学習率と個別に決まる。 |
| `--te_mlp_fc_only` | フラグ | OFF | TE1・TE2の各MLP直下の `fc1`・`fc2` をLoRA対象にする。 |
| `--downscale_freq_shift` | フラグ | OFF（shift 0） | UNetの拡散timestep埋め込みをshift 1にする。サイズ・cropの埋め込みは本家の設定を使う。 |
| `--fp16_safe_norms_mode` | `off` / `strict` | `off` | `strict` は対象UNetのGroupNorm・LayerNorm・通常attentionのsoftmaxをfp16入力時にfp32で計算し、元のdtypeに戻す。SDPAなどの専用attention経路は各backendに委ねる。 |

独自のTE別LRを使うときは `--learning_rate` を両TEの出発点にし、`--text_encoder_lr` 1値で両者を上書きし、最後に各TEの個別値を適用します。`--text_encoder_lr` に本家の2値形式で各TEのLRを与えた場合、同じTEに異なる個別値を重ねるとエラーになります。[TEの対象と学習率](maruo_training_features_ja.md#encoders)・[SDXL UNet](maruo_training_features_ja.md#unet)

### Guardianとepoch末の平均

| 引数 | 型・選択肢 | 省略時 | 効果・条件 |
| --- | --- | --- | --- |
| `--grad_norm_mode` | `stable` / `stable_no_threshoff` / `gamble` | Guardian無効 | 勾配ノルムでoptimizerとLR schedulerの更新を見送る。`stable` は非有限値も窓に含め、`stable_no_threshoff` は有限値だけを窓に入れる。`gamble` は非有限値を即時棄却する。 |
| `--avg_cp` | フラグ | OFF | epoch末の平均を有効化。対応optimizerは `AdamW8bit`、`AdamW8bitFast`、`AdamWBnb`。 |
| `--avg_cp_mode` | `live` / `promote` | `live` | `live` は窓が揃うと平均を適用。`promote` は同じ固定入力で現重みと候補を採点し、条件を満たした候補を採用。 |
| `--avg_window` | 整数 ≥ 1 | `4` | 平均に使うepoch末スナップショット数。 |
| `--avg_begin` | 実数 0–1 | `0.6` | `現在のepoch / 全epoch` が指定値に達したepochから窓に加える。 |
| `--avg_mode` | `ema` / `uniform` | `ema` | 窓内で近いepochに大きく重み付けするか、等重みで平均するかを選ぶ。 |
| `--avg_shadow_bank_size` | 整数 ≥ 1 | `12` | `promote` の比較用bankに保持する採点バッチの目標数。 |
| `--avg_shadow_margin` | 実数 ≥ 0 | `0.003` | `候補loss < 現重みloss × (1 − margin)` で勝ちと判定。既定は0.3%を超える改善。 |
| `--avg_shadow_patience` | 整数 ≥ 1 | `2` | `promote` で採用に必要な連続勝利数。 |
| `--avg_reset_stats` | フラグ | ON | 平均採用時に既存の `exp_avg`・`exp_avg_sq`・`exp_avg_max` をゼロ化。bnbの `state1`・`state2` は保持。`--no-avg_reset_stats` でこの処理を省く。 |
| `--avg_save_final_raw` | フラグ | OFF | 最終epochの判定直前のスナップショットを `final_raw` として別保存。無効を明示する形は `--no-avg_save_final_raw`。 |

本家引数 `--optimizer_type` には、`AdamW8bitFast`（bitsandbytesの更新を用いて同期を集約）と `AdamWBnb`（同じ高速経路で `min_8bit_size=2147483647` に固定し、通常のLoRAで32bit状態を使う）を追加しています。optimizer状態の精度と、次節のLoRA Δの量子化ビット数は独立です。[optimizer](maruo_training_features_ja.md#optimizer)・[Guardian](maruo_training_features_ja.md#guardian)・[平均・保存の順序](maruo_training_features_ja.md#averaging)

### DQの対象・bit数・固定倍率

| 引数 | 型・選択肢 | 省略時 | 効果・条件 |
| --- | --- | --- | --- |
| `--dq_delta_bits` | 整数 ≥ 2 | 指定なし | LoRAのforwardで加算するΔを指定bit数で疑似量子化。scheduleも省略するとDQは無効。 |
| `--dq_delta_bits_sched` | `進捗:bits` のカンマ区切り | 指定なし | 例 `0.0:8,0.5:4`。進捗0–1、各bitsは2以上。最初の境界までは固定bitsを使い、固定値もなければ量子化を待つ。 |
| `--dq_delta_begin` | 実数 0–1 | `0.0` | 完了した同期試行数 / 予定試行数がこの値に達した後のforwardから量子化。 |
| `--dq_delta_begin_after_lr_warmup` | フラグ | OFF | `--dq_delta_begin` より優先して予定LR warmup終了後から量子化。正かつ有限の `--lr_warmup_steps` を要する。2,000試行・5%なら101試行目から。 |
| `--dq_delta_mode` | `stoch` | `stoch` | 浮動小数のΔを近接する格子点へ確率丸め。 |
| `--dq_delta_granularity` | `channel` | `channel` | NC、NLC、NCHWのΔはチャネル別に尺度を計算。それ以外の形状は全体のRMSを使う。 |
| `--dq_delta_stat` | `rms` | `rms` | Δの二乗平均平方根を量子化尺度の基準にする。 |
| `--dq_delta_scope` | `unet` / `te` / `both` | `both` | 量子化する側。`te` は接続されているTE1・TE2。 |
| `--dq_delta_range_mul` | 正の実数 | `3.0` | RMSに掛ける基本倍率。自動倍率を使う場合は、その開始値にも使う。 |
| `--dq_delta_range_mul_attn2` | 正の実数 | 基本倍率を継承 | UNetのattn2だけ上書き。 |
| `--dq_delta_range_mul_te` | 正の実数 | 基本倍率を継承 | TE1・TE2共通の上書き。 |
| `--dq_delta_range_mul_te1` | 正の実数 | TE共通 → 基本倍率 | TE1個別の上書き。 |
| `--dq_delta_range_mul_te2` | 正の実数 | TE共通 → 基本倍率 | TE2個別の上書き。 |

場所別倍率4種には固定の `--dq_delta_bits` が必要です。bit scheduleや自動倍率との同時指定、上書きする部位に接続済みLoRAが1件もない指定はエラーになります。固定倍率の例では基本2.70、attn2・両TEを3.75にします。[固定DQと割当](maruo_training_features_ja.md#dq)

### DQの自動倍率・高速経路

| 引数 | 型・選択肢 | 省略時 | 効果・条件 |
| --- | --- | --- | --- |
| `--dq_delta_auto_range_mul` | フラグ | OFF | bitsまたはbits scheduleで有効にしたDQの共通倍率を自動調整。判定後の倍率は次のforwardに適用。 |
| `--dq_delta_auto_scope` | `unet` / `te` / `both` | 通常 `unet` | 判定に使う観測元。TEのみの学習、またはUNet専用学習ではなく量子化対象が `te` の場合は `te`。量子化対象を増やす引数ではない。 |
| `--dq_delta_auto_preset` | 下表の6種 | auto有効時 `default` | クリップ率の目標帯を選ぶ。自動倍率の例では `clip_rate_high`。 |
| `--dq_delta_auto_every` | 整数 ≥ 1 | `50` | 自動判定の周期（同期試行）。 |
| `--dq_delta_auto_use_raw` | フラグ | OFF | EMAが目標帯の上限／下限を外れた際、raw値も同じ側に外れたときだけ増減。 |
| `--dq_delta_auto_init_range_mul_from_band` | フラグ | OFF | autoの目標帯中央から初期倍率を計算。OFFなら基本倍率から開始。 |
| `--dq_delta_auto_warmup` | フラグ | ON | autoの観測を蓄積してから倍率調整を始める。`--no-dq_delta_auto_warmup` で待機を省く。 |
| `--dq_delta_auto_warmup_updates` | 整数 ≥ 0 | `0`（自動） | warmup中に待つ有効観測数。0なら現在40回。帯内のEMAが3回続くと早期終了。 |
| `--dq_delta_qerr_per_clip_floor` | 正の実数 | `0.001` | `clip_rate_low_auto` の誤差判定で、clip率を分母に使うときの下限。 |
| `--dq_delta_use_triton` | フラグ | OFF | 対応するCUDA入力の量子化にTritonを使う。対象外ではPyTorch経路を使用。 |
| `--dq_delta_triton_stats` | フラグ | OFF | Triton量子化時、対応する観測で統計の融合を試す。単独指定ではTorchの統計処理を使う。 |

| `--dq_delta_auto_preset` の値 | 目標clip率（割合） |
| --- | --- |
| `default` | 0.0005–0.003 |
| `clip_rate_high` | 0.003–0.005 |
| `clip_rate_high_narrow` | 0.0038–0.0048 |
| `clip_rate_mid` | 0.002–0.004 |
| `clip_rate_low` | 0.0005–0.0022 |
| `clip_rate_low_auto` | lowから開始し、進捗と誤差の条件を満たすとmidへ一度だけ移行可能 |

DQの開始時期を決めるLR warmupと、自動倍率を決めるための観測warmupは別々に指定できます。[自動倍率の判定](maruo_training_features_ja.md#dq-auto)・[Tritonの適用条件](maruo_training_features_ja.md#triton)

### ログと学習後の診断

| 引数 | 型・選択肢 | 省略時 | 効果・条件 |
| --- | --- | --- | --- |
| `--dq_delta_log` | フラグ | OFF | 有効なDQの観測を `dq.csv` に記録。自動判定の `dq_auto.csv` はautoの有効化で別に生成。 |
| `--dq_delta_log_every` | 整数 ≥ 1 | `100` | DQの通常ログを観測する周期（同期試行）。 |
| `--dq_delta_log_scope` | `unet` / `te` / `both` | `--dq_delta_scope` を継承 | 通常ログに集約する観測元。`both` は観測元ごとの行を出す。 |
| `--dq_delta_log_mode` | `summary` | `summary` | 通常のDQ統計を集約して記録。 |
| `--dq_delta_log_detail` | `basic` | `basic` | 通常ログの詳細レベル。 |
| `--rank_log` | フラグ | OFF | 同期試行番号が100の倍数でGuardianを通過したとき、**UNet**のLoRAモジュール単位のrank統計を `rank.csv` に書く。 |
| `--rank_log_mode` | `per_module` | `per_module` | `--rank_log` の集計形式。学習対象のUNetモジュールを個別に識別する。 |
| `--group_loss_log` | フラグ | OFF | datasetの実画像・subset・groupとlossを `group_loss.csv` に対応付ける。datasetのbatch size 1が必要。 |
| `--report_after_training` | フラグ | OFF | 学習・最終保存・ログ確定後、当該runでepoch推移付きの診断レポートを作る。 |

`gradient.csv` はGuardian、`promote.csv` はpromoteが有効にします。観測・棄却・optimizer更新は別々に数えるため、進捗バーの `skipped` はGuardianの棄却回数として読みます。学習開始前の初期化時間は自動表示され、進捗の速度計時は初回バッチから始まります。[ログの記録条件](maruo_training_features_ja.md#logging)・[重みの役割と保存先](maruo_training_features_ja.md#run-output)

<a id="extension-points"></a>

## 本家引数・設定と独自機能の接点

| 入力 | このブランチで関係する挙動 |
| --- | --- |
| 学習 `--output_dir`、`--output_name` | 指定した出力ルートに学習1回の `日時_base_name` ディレクトリを作成。重み、設定、6種類のうち有効なCSVがまとまる。[runの出力](maruo_training_features_ja.md#run-output) |
| 学習 `--network_weights`、`--resume` | 既存LoRAを初期値にして新規runを開始する方法と、optimizer途中状態を読み戻す方法。観測だけの本家学習で後者を使ったときも、新しいrunとして記録する。[適用条件](maruo_training_features_ja.md#scope) |
| 学習 `--run_control`、本家の定期・手動サンプル | 学習の独自機能と併用可能。定期・手動サンプルは出力ルートの `sample/` に保存。各コマンドとstep境界は[run controlとの関係](maruo_training_features_ja.md#control)・[本家のrun control](run_control.md)を参照。 |
| dataset TOMLの `group` | dataset／subset等の設定からgroup名を解決し、group lossで参照。名前の例と無指定時の扱いは[ログ](maruo_training_features_ja.md#logging)を参照。 |
| 生成 `--network_module`、`--network_weights` | 通常LoRAは `networks.lora`、SDXLのLBWを指定するslotには `networks.lora_lbw` を使用。[画像生成](maruo_tools_ja.md#generate) |
| 生成 `--network_mul`（実数列） | N本のslotに対し、0～N値はslot順の共通強度（不足分1.0）、2N値は各slotのTE共通・UNetの組、3N値は各slotのTE1・TE2・UNetの組。例はLoRA 1本で `--network_mul 0.8 0.25 0.5`。分割強度は `--network_merge`、`--network_merge_n_models` によるマージ、`--network_pre_calc` と併用不可。[画像生成](maruo_tools_ja.md#generate) |
| 生成 `--network_lbw`（文字列列） | 省略でブロック倍率1。各slotに `ALL`、`NONE`、`XLMIDD`、`XLMLT1` またはカンマ区切りの有限な12値を指定する。slot数より指定が少ないと最後の指定を引き継ぐ。先頭の係数は両TEに作用する。[画像生成](maruo_tools_ja.md#generate) |
| 生成prompt行内の `--am` | 行単位の強度上書き。`--am 0.8,0.25,0.5` のようにカンマ区切りで指定し、その行の共通強度がslot数に満たない場合は最後の値を反復。[画像生成](maruo_tools_ja.md#generate) |

<a id="tool-options"></a>

## 独自ツールの公開CLI一覧

以下のパスはリポジトリのルートからの相対パスです。GUIで渡す設定、内部workerの引数やキューの受領・取消用フラグ、旧版からの互換性確認用の非公開引数は利用者向けCLIの一覧から除いています。

### 診断、density、比較ビューア

| 入口 | 引数 | 省略時・用途 |
| --- | --- | --- |
| `tools/make_lora_diagnostic_report.py` | `--run_dir PATH` | 対象runを直接指定。省略時は `--base_name` が必要。 |
| 同上 | `--input_dir PATH` | 名前で探す出力ルート。省略時はカレントディレクトリ。`--run_dir` と併用時はそのrunの出力ルートに一致する必要がある。 |
| 同上 | `--base_name NAME` | 学習名で出力ルート直下の索引を検索。1件なら直接、複数なら対話で選択。非対話で複数なら `--run_dir` を明示。 |
| 同上 | `--loss_ma_window N` | loss移動平均の窓幅。整数、既定100。1以下は平滑化しない。 |
| 同上 | `--lora_bins N` | LoRA密度解析のヒストグラムのビン数。整数、既定128、密度解析時は2以上。 |
| 同上 | `--skip_lora_analysis` | 既定OFF。LoRA重みの単体密度・epoch推移を省く。 |
| 同上 | `--lora_epoch_trend` | 既定OFF。ONで登録済みepochと標準finalの密度推移を含める。`--skip_lora_analysis` が優先。 |
| `tools/analyze_lora_density.py` | 位置引数 `model` | 必須。LoRA重みのパス。run登録重みはrun内の `density/` に、索引のない単体重みはカレントの `analyze_lora_result/` に保存。 |
| 同上 | `--bins N` | 整数、既定128、2以上。単独densityのヒストグラムのビン数。 |
| 同上・本家共通ログ引数 | `--console_log_level LEVEL`、`--console_log_file PATH`、`--console_log_simple` | 実効レベルは `INFO`（ほかに `DEBUG`、`WARNING`、`ERROR`、`CRITICAL`）。既定はコンソール出力、simpleはOFF。ファイル指定時はそのパスへ書き出す。 |
| `tools/lora_comparison.html` | 引数なし | ブラウザで開き、比較するrunの診断JSONを選ぶ。 |

[診断レポートを作る](maruo_tools_ja.md#report)・[densityを保存する](maruo_tools_ja.md#density)・[複数runの診断比較](maruo_tools_ja.md#viewer)

### 画像比較と研究台帳

| 入口 | 引数 | 省略時・用途 |
| --- | --- | --- |
| `sdxl_lora_report_gui.py` | 独自CLIなし | 比較条件の設定・キュー・実行はGUIで行う。 |
| `sdxl_lora_report_cui.py` | `--config PATH` | 必須。比較条件・モデル・生成設定を含むJSON。 |
| 同上 | `--dry-run` | 既定OFF。生成計画を組み立てて確認する。ONでは生成器を起動しない。 |
| `tools/lora_ledger.py` | 位置引数 `command` | 必須。`init` は台帳作成、`scan` は登録候補の確認、`register` は登録反映、`verify` は登録情報の検証、`export` は選択範囲の出力。 |
| 同上 | `--ledger PATH` | 全コマンドで必須。新規作成は出力ルート直下の `ledger` を指定する。 |
| 同上 | `--full` | 既定OFF。scan／register／verify時に重みの全内容のSHAを調べる。 |
| 同上 | `--run-id UUID` | 省略で対象全体。複数は引数を繰り返す。scan／register／verify／exportの対象を限定。 |
| 同上 | `--selection PATH` | export専用。`{"run_ids":["UUID", "UUID"]}` 形式の選択JSON。`--run-id` を併用するとそちらを優先。 |
| `lora_ledger_gui.py` | `--ledger PATH` | 任意。省略時は記録済みの前回台帳があれば再表示し、画面でも選べる。`lora_ledger_gui.bat` からも起動できる。 |

[生成設定とslotの強度](maruo_tools_ja.md#generate)・[比較・blind採点](maruo_tools_ja.md#compare)・[研究台帳の登録・評価・export](maruo_tools_ja.md#ledger)

### SDXL tokenizer

`tools/sdxl_tokenize.py` は同じ文字列をTE1とTE2で分割します。`--text` または `--search-core` が必要です。両方を指定すると検索を実行します。[タグの分割と候補探索](maruo_tools_ja.md#tokenizer)

| 引数 | 省略時 | 用途 |
| --- | --- | --- |
| `--text TEXT` | 入力なし | タグやpromptのトークンID・分割境界を表示。 |
| `--tokenizer-cache-dir PATH` | 専用ディレクトリ指定なし | モデル別のローカルキャッシュから読み込み、なければ取得・保存する。 |
| `--add-special-tokens` | OFF | 分割と検索にBOS/EOSを含める。 |
| `--search-core TEXT` | 検索しない | この文字列へ文字を追加してTE1・TE2の分割を検査。 |
| `--search-side prefix/suffix/both` | `both` | 前付け、後付け、両方向を個別に列挙する。 |
| `--search-min-add N` | `1` | 追加文字数の下限。0なら元の語も試す。 |
| `--search-max-add N` | `3` | 追加文字数の上限。 |
| `--search-min-tokens N` | `2` | 合格に必要なトークン数の下限。各分割片には1文字だけのものがないことも条件。 |
| `--search-alphabet TEXT` | `abcdefghijklmnopqrstuvwxyz` | 追加文字の候補集合。 |
| `--search-limit N` | `20` | 表示する同率最良候補の上限。 |
| `--search-either` | OFF | ONならTEのどちらかが条件を満たせば採用。既定では両方で条件を満たす必要がある。 |
