# SDXL LoRA 学習設定の使用例

以下は単一GPU・fp16・rank 4のSDXL LoRAを、Windows PowerShellから学習する例です。リポジトリ直下で実行してください。GUIを使う場合は、このリポジトリの仮想環境に `venv\Scripts\python -m pip install -r requirements-gui.txt` で追加の依存パッケージを入れます。

## WindowsでTritonを追加する（任意）

Windows x64でCUDA対応のPyTorch 2.9系と本家の `requirements.txt` を導入した後、Tritonを使う場合は同じ仮想環境で次を実行します。検証した組み合わせは `torch==2.9.1+cu130` と `triton-windows==3.5.1.post24` です。[Tritonの任意依存ファイル](../requirements-triton-windows.txt)・[検証時の環境記録](migration/baseline/windows-cu130.lock.txt)

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-triton-windows.txt
```

以下の固定倍率・自動倍率の例では、`--dq_delta_use_triton` で量子化にTritonを使い、`--dq_delta_triton_stats` で統計の集計にもTritonを使います。Torchだけで量子化する場合は、この導入手順と2つの引数を省いてください。量子化なしの例でもTritonの追加は任意です。

## 量子化なし・固定倍率・自動倍率で学習する

### 共通の学習設定

モデルとデータセットTOMLには手元のファイルを指定します。出力名は学習ごとに選べます。以下を同じPowerShellで上から実行します。

```powershell
$modelPath = Read-Host 'SDXLベースモデルのファイルパス'
$datasetPath = Read-Host 'データセットTOMLのファイルパス'
$trainingPython = '.\venv\Scripts\python.exe'
$outputName = 'example_r4'
$common = @(
    '--pretrained_model_name_or_path', $modelPath, '--dataset_config', $datasetPath,
    '--output_dir', '..\lora_output', '--output_name', $outputName,
    '--network_module', 'networks.lora', '--network_dim', '4', '--network_alpha', '1',
    '--network_dropout', '0.3', '--network_args', 'rank_dropout=0.2',
    '--learning_rate', '3.5e-4', '--text_encoder_lr', '2e-4', '--text_encoder_lr1', '3e-4', '--text_encoder_lr2', '2e-4',
    '--optimizer_type', 'AdamW8bitFast', '--lr_scheduler', 'constant_with_warmup', '--lr_warmup_steps', '0.05',
    '--mixed_precision', 'fp16', '--save_precision', 'fp16', '--save_model_as', 'safetensors', '--sdpa',
    '--seed', '39', '--max_train_epochs', '40', '--save_every_n_epochs', '1', '--max_data_loader_n_workers', '1',
    '--prior_loss_weight', '1.0', '--noise_offset', '0.15', '--adaptive_noise_scale', '0.1',
    '--enable_bucket', '--min_bucket_reso', '384', '--max_bucket_reso', '1024', '--cache_latents', '--training_comment', 'test',
    '--downscale_freq_shift', '--te_mlp_fc_only', '--grad_norm_mode', 'stable_no_threshoff',
    '--avg_cp', '--avg_cp_mode', 'promote', '--avg_window', '4', '--avg_begin', '0.6', '--avg_mode', 'ema',
    '--avg_shadow_bank_size', '12', '--no-avg_reset_stats', '--avg_save_final_raw',
    '--fp16_safe_norms_mode', 'strict', '--rank_log', '--rank_log_mode', 'per_module'
)
$dq = @()
```

### 量子化方式を選ぶ

**量子化なし**は上の空の `$dq` を使います。そのまま「学習を開始する」へ進んでください。

**固定倍率・自動倍率**は、まず次の共通引数を設定します。UNetと両TEに8bitの確率丸めを使用します。

```powershell
$dq = @(
    '--dq_delta_bits', '8', '--dq_delta_granularity', 'channel', '--dq_delta_stat', 'rms',
    '--dq_delta_mode', 'stoch', '--dq_delta_begin_after_lr_warmup', '--dq_delta_scope', 'both',
    '--dq_delta_log', '--dq_delta_log_detail', 'basic', '--dq_delta_use_triton', '--dq_delta_triton_stats'
)
```

続いて、次の2つのうち使う方式のブロックを **1つだけ** 実行します。

**固定倍率**は全体に2.70、UNetのattn2と両TEに3.75を割り当てます。

```powershell
$dq += @('--dq_delta_range_mul', '2.70', '--dq_delta_range_mul_attn2', '3.75', '--dq_delta_range_mul_te', '3.75')
```

**自動倍率**は目標帯から初期倍率を求め、学習中のclip率に応じて倍率を更新します。

```powershell
$dq += @('--dq_delta_range_mul', '3.0', '--dq_delta_auto_range_mul', '--dq_delta_auto_preset', 'clip_rate_high',
    '--dq_delta_auto_init_range_mul_from_band', '--dq_delta_auto_use_raw')
```

### 学習を開始する

選んだ引数を通常のSDXL学習コマンドへ渡します。各実行で新しいrunフォルダが作られます。

```powershell
& $trainingPython -m accelerate.commands.launch --num_cpu_threads_per_process 8 --num_processes 1 --num_machines 1 --mixed_precision fp16 --dynamo_backend no sdxl_train_network.py @common @dq
```

学習TOMLで有効にした指定を個別に解除する場合は、`--no-avg_reset_stats`、`--no-avg_save_final_raw`、`--no-dq_delta_auto_warmup` を使えます。

通常設定のまま終了後の診断も作る場合は、最後の起動行に `--report_after_training` を加えます。group別に見たい場合は `--group_loss_log` も加えます。上の3構成はどちらも省略した設定です。

## 学習の設定と進捗を読む

|指定（省略時）|実際の作用|
|---|---|
|`--te_mlp_fc_only`（OFF）|TE1とTE2の各Transformer層のMLP `fc1` / `fc2` にLoRAを置きます。TE1が12ブロック・TE2が32ブロックのモデルでは、それぞれ24・64モジュールです。|
|`--text_encoder_lr1` / `--text_encoder_lr2`（個別指定なし）|`--learning_rate` を出発点に、共通 `--text_encoder_lr` を両TEへ、個別指定を該当TEへ適用します。本家の2値形式で両TEをすでに指定した場合、同じTEへの異なる個別値はエラーになります。|
|`--downscale_freq_shift`（OFF）|SDXL UNetの拡散timestep埋め込みにshift 1を適用します。解像度・crop等のサイズ埋め込みは本家の条件を使います。|
|`--fp16_safe_norms_mode off/strict`（`off`）|`strict` ではSDXL UNetの対象演算への入力がfp16のとき、GroupNorm・LayerNormと通常attentionのsoftmaxをfp32で計算してfp16へ戻します。`--sdpa` のattentionは本家のSDPAに委譲します。|
|`--optimizer_type AdamW8bitFast` / `AdamWBnb`|FastはbitsandbytesのAdamW8bit更新と状態を用い、同期回数を減らします。AdamWBnbは8bit状態に切り替わるサイズ閾値を高く固定し、常用のLoRAパラメーターでは32bit状態を使います。本家の `AdamW8bit` も選択できます。|
|`--grad_norm_mode`（Guardian OFF）|`stable_no_threshoff` は有限な勾配ノルムを窓に入れて閾値を更新。`stable` は非有限の観測も窓に保持。`gamble` は非有限値を即時棄却します。閾値超過を棄却するとoptimizerとLR schedulerの更新を見送ります。|
|`--avg_cp`（OFF）、`--avg_cp_mode live/promote`（`live`）|`live` は窓が揃った平均重みを適用します。`promote` は固定bankの同じ入力で現重みと候補を採点し、改善幅と連勝条件を満たした候補を採用します。|
|`--avg_begin 0.6`、`--avg_window 4`、`--avg_mode ema/uniform`（`ema`）|平均を始めるepochの割合、窓の長さ、重み付けです。`--avg_shadow_bank_size 12`、`--avg_shadow_margin 0.003`、`--avg_shadow_patience 2` がpromoteの既定値です。|
|`--avg_reset_stats`（ON）、`--avg_save_final_raw`（OFF）|平均採用時に対応するoptimizerの状態バッファをリセットします。常用では `--no-avg_reset_stats` で維持。`--avg_save_final_raw` は最終epoch判定前の重みも保存します。|
|`--rank_log` / `--rank_log_mode per_module`、`--group_loss_log`（両ログともOFF）|rankは100同期試行ごとのUNetモジュール別。groupは学習の実画像とlossを対応付けます。Guardianの棄却や固定bankの採点を、更新後のrankや通過したgroup観測へ加えません。|

group別のlossを記録するときは、学習に `--group_loss_log` を加え、dataset TOMLの各subsetに名前を付けます。すでにdataset TOMLがある場合は対応するsubsetに `group` の行を加えます。

```toml
[[datasets]]
batch_size = 1

[[datasets.subsets]]
image_dir = "path/to/images"
num_repeats = 15
group = "subset_a"
```

同じ名前のsubsetは、groupのEMAと観測回数を共有します。名前を省略したsubsetと空白だけの名前は `__ungrouped__` にまとまります。この観測にはdatasetのbatch size 1が必要です。

初期化完了時に「学習開始前の初期化時間」が表示されます。進捗バーの速度は初期化後から計測します。進捗バーの分母は予定した同期試行数、`skipped` はその時点までにGuardianが更新を見送った累計です。AMP側で更新を見送る場合もあるため、`試行数 − skipped` をoptimizerの実更新回数とみなす用途には使えません。

## DQの対象、開始時期と倍率を選ぶ

|指定（省略時）|使い方|
|---|---|
|`--dq_delta_bits`、`--dq_delta_bits_sched`（両方省略でDQ無効）|固定値は2以上。scheduleは `0.0:8,0.5:4` のように学習進捗とbit数を並べます。初期bitsも開始点0の指定もない場合、最初の境界までは素通しです。1bitは開始前に拒否します。|
|`--dq_delta_mode stoch`、`--dq_delta_granularity channel`、`--dq_delta_stat rms`|DQ有効時はこの3値が既定です。LoRAのdeltaをチャンネル別RMSから決めた尺度で確率丸めします。|
|`--dq_delta_scope`（`both`）|`unet`、`te`、`both` から量子化対象を選びます。旧版で `unet` と書いてTEにも量子化が掛かっていた設定には `both` を指定します。|
|`--dq_delta_begin`（`0.0`）、`--dq_delta_begin_after_lr_warmup`（OFF）|前者は予定試行に対する開始割合です。後者を指定する場合は、`--lr_warmup_steps` に0より大きい有限値を設定します。たとえば予定が2,000試行でwarmupが5%なら、100試行を終えた101試行目から量子化します。Guardianの棄却でLR更新が遅れてもDQの試行時計は進みます。|
|`--dq_delta_range_mul`（`3.0`）、`--dq_delta_range_mul_attn2`、`--dq_delta_range_mul_te`、`--dq_delta_range_mul_te1` / `--dq_delta_range_mul_te2`（上書きなし）|固定倍率の例では全体2.70、UNet attn2と両TEを3.75。TE個別を明示すると共通TEより優先します。場所別上書きは固定bits専用です。実割当はrunの設定に残ります。|
|`--dq_delta_auto_range_mul`（OFF）、`--dq_delta_auto_preset`（auto有効時 `default`）|自動倍率の例は `clip_rate_high`。ほかに `clip_rate_high_narrow`、`clip_rate_mid`、`clip_rate_low`、`clip_rate_low_auto` を選べます。autoが決めた倍率は次のforwardから使います。|
|`--dq_delta_auto_every`（50）、`--dq_delta_auto_scope`（通常 `unet`）|指定間隔の実測で制御。TEだけを学習または量子化する通常構成では `te`。制御対象・量子化対象・ログ対象は設定に別々に残ります。|
|`--dq_delta_auto_use_raw`、`--dq_delta_auto_init_range_mul_from_band`（両方OFF）|raw clip率を判定に使う設定と、目標帯から初期倍率を求める設定です。自動倍率の例は両方を指定します。|
|`--dq_delta_auto_warmup`（ON）、`--dq_delta_auto_warmup_updates`（`0` で自動）、`--dq_delta_qerr_per_clip_floor`（`0.001`）|auto自身の観測待ちと、誤差をclip率で割る際の分母下限です。autoのwarmupとoptimizerのLR warmupは別の時計です。|
|`--dq_delta_use_triton`、`--dq_delta_triton_stats`（両方OFF）|前者は対応形状のCUDA量子化、後者は統計との融合を試します。対応外の演算はTorchへ戻ります。|
|`--dq_delta_log`（OFF）、`--dq_delta_log_every`（100）、`--dq_delta_log_scope`（量子化scopeを継承）|観測間隔と保存対象。`--dq_delta_log_mode summary`、`--dq_delta_log_detail basic` が公開形式です。|

## runの成果物と診断

`--output_dir ..\lora_output --output_name example_r4` の場合は、次のように保存します。同じ秒・同じ名前なら連番が付きます。`run.json` のUUIDが学習回を識別します。

```text
lora_output/
  YYYYMMDD-HHMMSS_example_r4/
    run.json  training_settings.json
    *.safetensors
    gradient.csv  rank.csv  group_loss.csv
    dq.csv  dq_auto.csv  promote.csv
    diagnostic_report/report.html  diagnostic_report/report.json
    density/
  sample/
  ledger/
```

重みとログは同じrunに並びます。設定の正本は `training_settings.json` の `requested` と `resolved` です。`run.json` は重みの役割・epoch・同期試行・各ログの有効／無効／部分出力を示します。6 CSVは機能に応じて作られ、量子化なしの例ではDQログが無効です。上記3例ではgroupログも無効です。`gradient.csv` は1マイクロバッチ、`dq.csv` / `dq_auto.csv` / `rank.csv` / `group_loss.csv` は該当する同期試行、`promote.csv` はepoch判定と窓の内訳を記録します。空欄と数値0は別の値です。

40 epochを毎epoch保存する上の例では、使う目的に合わせて `run.json` の `role` から重みを選べます。

|役割|ファイルの例|保存する時点|
|---|---|---|
|epoch| `example_r4-000001.safetensors` |1〜39 epochの保存周期ごとに、そのepochの平均・promote判定前を保存します。|
|final| `example_r4.safetensors` |40 epochの判定後に保存する標準LoRAです。最終候補を採用した場合はその重みが入ります。|
|final_raw| `example_r4_final_raw.safetensors` |平均を有効にして `--avg_save_final_raw` を指定すると、40 epochの判定直前も保存します。最終判定の前後を比較できます。|

本家の定期・手動サンプル画像は共有の `lora_output\sample\` に入ります。run controlの操作は本家の境界で処理します。サンプル生成中の時間・一時的なVRAM増加は学習条件と生成サイズに左右されます。

```powershell
& .\venv\Scripts\python.exe tools\make_lora_diagnostic_report.py --input_dir ..\lora_output --base_name example_r4 --lora_epoch_trend
$runDir = Read-Host "操作するrunのフルパス"
& .\venv\Scripts\python.exe tools\make_lora_diagnostic_report.py --run_dir $runDir --skip_lora_analysis
& .\venv\Scripts\python.exe tools\analyze_lora_density.py (Join-Path $runDir "example_r4.safetensors") --bins 128
```

最初のコマンドは同名runが複数あれば対話で選べます。自動実行では `--run_dir` で指定します。2行目には操作するrunのパスを入力します。複数候補を表示したときは、選んだ候補の `--run_dir` に続くパスを使えます。`--skip_lora_analysis` は密度を省いた診断を同じ保存先に再生成します。通常のepoch推移に戻すときは `--lora_epoch_trend` で再生成します。診断失敗後も正常に完了した学習重みをそのまま使えます。重みの単体解析は `density`、複数runの診断比較はHTMLに各 `report.json` を読み込みます。[オフライン診断比較](../tools/lora_comparison.html)

## 画像生成からblind評価・研究台帳まで

LoRA 1本の生成例です。次の3値はTE1、TE2、UNetの順です。UNetにはさらにLBWのブロック係数が掛かります。`prompts.txt` の例は `paper lantern, still life --d 39`。生成時のパスには準備した実ファイルを指定します。

```powershell
& .\venv\Scripts\python.exe sdxl_gen_img.py --ckpt $modelPath --network_module networks.lora_lbw --network_weights (Join-Path $runDir "example_r4.safetensors") --network_mul 0.8 0.25 0.5 --network_lbw XLMLT1 --from_file prompts.txt --outdir ..\generated --bf16 --sdpa
& .\venv\Scripts\python.exe sdxl_lora_report_gui.py
$reportDir = Read-Host "GUIで作成した比較フォルダのパス"
Copy-Item (Join-Path $reportDir "config.json") .\study.json
& .\venv\Scripts\python.exe sdxl_lora_report_cui.py --config study.json --dry-run
& .\venv\Scripts\python.exe sdxl_lora_report_cui.py --config study.json
```

GUIでモデル・LoRA・prompt・seedを決めて「Run report」を押すと、その比較フォルダに `config.json` が保存されます。上の例はこの設定を `study.json` にコピーしてCUIで使います。`--dry-run` は条件とjobを組み立てます。画像生成は同じコマンドから `--dry-run` を外して開始します。N本のLoRAならN値までが各slotの共通強度、2N値が各slotのTE共通とUNet、3N値がTE1・TE2・UNetです。prompt内の `--am 0.4,0.3,0.7` はその行だけ上書きします。部位分割指定とmerge／precalcの組合せは対象外です。GUI初期設定と手書きJSONの省略時既定は異なるため、比較では保存した設定を共用します。

1. 比較フォルダの `report.html` で生成できた画像を確認します。blindで採点するときは `blind_report.html` または `blind_report_detailed.html` を先に開きます。
2. 詳細blindでは開示後に「集計CSVを保存」で採点を書き出せます。未採点・同順位・画像欠落の扱いが区別されます。比較フォルダには `metadata.json` と対応する画像を一緒に保管します。
3. 台帳GUIで `lora_output\ledger` を開き、学習runを登録します。「比較の判断を記録」→「比較レポートから取り込む」で `metadata.json` のある比較フォルダを選び、実際に見た候補・実用性・理由を確認して保存します。ブラウザの採点CSVは別に保管できます。評価を修正すると履歴に新しい改訂が残ります。
4. 同じ選択範囲でexportすると、その時点のrun・重み・評価・指標をひとまとめに再読できます。旧台帳の引継ぎと保存済みexportの開き方は[ツールの操作ガイド](maruo_tools_ja.md#ledger)を参照してください。

```powershell
& .\venv\Scripts\python.exe tools\lora_ledger.py init --ledger ..\lora_output\ledger
& .\venv\Scripts\python.exe tools\lora_ledger.py scan --ledger ..\lora_output\ledger
& .\venv\Scripts\python.exe tools\lora_ledger.py register --ledger ..\lora_output\ledger
& .\venv\Scripts\python.exe lora_ledger_gui.py --ledger ..\lora_output\ledger
& .\venv\Scripts\python.exe tools\lora_ledger.py export --ledger ..\lora_output\ledger
& .\venv\Scripts\python.exe tools\sdxl_tokenize.py --text "paper lantern, still life"
```

## 旧環境から移るとき

旧LoRAの重みを `--network_weights` で読み込むと新しい学習runとして開始します。独自学習機能は1プロセス・勾配蓄積1・datasetの実効倍率1で使用します。独自学習を伴うoptimizer途中状態からの `--resume` は初回移植の対象外です。本家学習＋観測だけの構成は本家の再開経路を使い、再開後の記録には新しいrun UUIDが付きます。

新しいログを読む診断と台帳は、run索引を起点にします。旧 `.txt` やJSONLの診断には旧環境を利用できます。現在使える引数と既定値は[独自CLIリファレンス](maruo_cli_ja.md)、移植時に選んだ機能と理由は[移植の記録](migration/MIGRATION_PLAN.md)にまとめています。
