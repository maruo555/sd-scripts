# グループ別Loss(EMA)ログ機能

`sdxl_train_network.py`（`train_network.py` 共通ループ）に追加された、学習中の `loss` をグループ単位で可視化する機能です。

## 概要

- 目的:
  - 複数キャラ同時学習時に、キャラごとの学習進み具合の偏りを観測する
- 機能:
  - subsetごとに設定した `group` で、学習lossのオンラインEMAを集計
  - stepログCSV（全step記録）とepochサマリCSVを出力
  - 任意で、epoch境界で group別LR自動補正（boost-only）を適用

## 制約条件

- `--group_loss_log` は **batch_size=1 専用**です。
  - データセット定義の `batch_size` が 1 以外ならエラーで停止します。
  - 実行中バッチサイズが 1 以外でもエラーで停止します。
- 想定利用は DreamBooth の `class_tokens` 方式です（最低限この方式をサポート）。
- 分散学習時（DDP）は **main process のみCSV出力**します（rank0のローカルバッチに基づくログ）。
- `skip_grad_norm` で更新がスキップされたstepは、EMA集計にもCSVにも含みません。
- `loss` が `NaN` / `inf` のstepは、EMA集計にもCSVにも含みません。
- `--group_lr_auto` は以下を満たさないとエラーで停止します。
  - `--group_loss_log` と `--group_loss_epoch_summary` を同時指定
  - `gradient_accumulation_steps = 1`
  - 単一process実行（multi-process/DDP不可）
  - optimizerの全 `param_group["lr"]` が数値であること（`lr=None` は非対応）
    - 例: Adafactor の既定（`relative_step=True`）は `lr=None` になるため非対応
- `--group_lr_auto` は resume/state保存系オプションと併用できません（指定時はエラー）。
  - `--resume`
  - `--save_state`
  - `--save_state_on_train_end`
  - `--resume_from_huggingface`
  - `--save_state_to_huggingface`

## dataset_config.toml 拡張

`[[datasets.subsets]]` に以下の任意キーを追加できます。

- `group = "hyuzu"`
  - グループ識別子（文字列）

`group` 未指定（または空文字）のsubsetは、ログ上 `__ungrouped__` として扱われます。

## 参考情報（重複サブセットとキャッシュ）

- 同一 `[[datasets]]` 内で、DreamBooth方式の同一 `image_dir` 重複subsetは **後続が無視**されます。
  - この重複判定は `image_dir` ベースです。`class_tokens` が違っていても別扱いにはなりません。
- 同一 `[[datasets]]` 内で、FineTuning方式の同一 `metadata_file` 重複subsetも後続が無視されます。
- 同じ画像フォルダを別設定で使いたい場合は、`[[datasets]]` を分ければ併用可能です。
- ただし、`[[datasets]]` を分けても同じ実画像パスを再利用する場合は、次のディスクキャッシュ系オプションは避けてください。
  - `--cache_latents_to_disk`
  - `--cache_text_encoder_outputs_to_disk`
  - 同じキャッシュファイルパスを共有して衝突・上書きが発生する可能性があります。
- 参考:
  - `--cache_latents`（メモリキャッシュのみ）は通常この制約に当たりません。

## CLIオプション一覧

### ログ出力オプション

| オプション | 既定値 | 説明 |
|---|---:|---|
| `--group_loss_log` | `False` | グループ別Loss(EMA)ログ機能を有効化 |
| `--group_loss_ema_beta <float>` | `0.98` | EMA係数（`ema = ema*beta + loss*(1-beta)`） |
| `--group_loss_log_every_n_steps <int>` | `100` | stepログCSVのバッファを書き出す間隔（global step）。記録自体は全stepで行う |
| `--group_loss_epoch_summary` | `False` | epoch末サマリCSVを追記出力 |

### LR自動調整オプション

| オプション | 既定値 | 説明 |
|---|---:|---|
| `--group_lr_auto` | `False` | group別LR自動補正（boost-only）を有効化 |
| `--group_lr_auto_warmup_epochs <int>` | `3` | warmup中は全group scale=1.0固定 |
| `--group_lr_auto_min_count <int>` | `20` | epoch更新対象にする最小 `count_epoch` |
| `--group_lr_auto_ratio_ema_beta <float>` | `0.85` | `ratio_ema` 平滑化のEMA係数 |
| `--group_lr_auto_power <float>` | `0.5` | `ratio_ema**power` の指数 |
| `--group_lr_auto_boost_threshold <float>` | `1.10` | `ratio_ema` の補正判定閾値（以上で補正） |
| `--group_lr_auto_max_scale <float>` | `1.2` | scale上限 |
| `--group_lr_auto_max_change <float>` | `0.05` | 1epochあたりのscale変化率上限（上下両方向） |

## group別LR自動調整の仕組みと仕様

### 1) 基本方針

- 補正対象は「stepで実際に学習されたgroup」
- 補正は **boost-only**（`scale >= 1.0`）で、LRを1.0未満には下げません
- epoch内ではscale固定、epoch境界でのみ更新します

### 2) 更新タイミング

- 更新は各epoch末に行い、**次epoch** の学習stepから反映されます
- `warmup_epochs` の間は全group `scale=1.0` 固定です
  - 例: `warmup_epochs=3` の場合
    - epoch 1-3 は固定（`scale=1.0`）
    - epoch 4 で baseline 確定
    - epoch 5 で進捗比の判定
    - その判定結果が実stepへ反映されるのは epoch 6 から（epoch境界更新のため）

### 3) 更新に使う統計

- 指標は epochサマリの `ema_loss_end` から作る「進捗（改善率）」
- `count_epoch < min_count` のgroupは、そのepochの更新計算から除外
- `ema_loss_end` が非finiteまたは0以下のgroupも除外
- baseline は `baseline_loss_end[g]` として保持し、
  - warmup終了後、`count_epoch >= min_count` を満たした最初のepochで確定
  - `progress[g] = ema_loss_end[g] / baseline_loss_end[g]`
  - 値が大きいほど改善が遅く、ブースト対象になりやすい

### 4) 基準進捗（ref_progress）の決め方

- 有効group集合 `L = {progress[g]}`、その要素数を `N` とします
- `N >= 2`: `ref_progress = median(L)`
  - `N == 2` のときも、2値の平均（一般的なmedian定義）を使います
- `N <= 1`: 更新しない（全group `scale=1.0`）

### 5) scale計算

group `g` ごとに以下を計算します。

- `ratio = progress[g] / ref_progress`
- `ratio_ema` を epoch境界で更新します
  - 初回: `ratio_ema[g] = ratio[g]`
  - 以後: `ratio_ema[g] = beta * ratio_ema[g] + (1-beta) * ratio[g]`
  - `beta` は `--group_lr_auto_ratio_ema_beta`（既定 `0.85`）
  - warmup中 / baseline未確定 / `min_count` 未満のgroupは `ratio_ema` を更新しません（そのepochでは未定義扱い）
- 判定は固定閾値方式です
  - `ratio_ema[g] >= threshold` のgroupのみ補正対象
  - `ratio_ema[g] < threshold` は補正しません（`scale_candidate=1.0`）
  - `threshold` は `--group_lr_auto_boost_threshold`（既定 `1.10`）
- 補正対象の計算:
  - `scale_candidate = min(max_scale, ratio_ema[g] ** power)`

最後に、急変を抑えるため1epochあたりの変化率制限を適用します（上げ下げ両方向）。

- `lower = prev_scale * (1 - max_change)`
- `upper = prev_scale * (1 + max_change)`
- `scale_next = clamp(scale_candidate, lower, upper)`
- 最後に boost-only 制約と上限制約を再適用します
  - `scale_next = max(1.0, min(max_scale, scale_next))`

`min_count` 未満で更新対象外、または baseline 未確定のgroupは、そのepochでは `scale_candidate=1.0`（max_change制限後の値）として扱います。

### 6) 実学習への適用方法

- schedulerが算出したbase LRを壊さないため、以下の順序で適用します
  1. `optimizer.step()` 直前に、当該stepのgroupのscaleをparam_group LRに乗算
  2. `optimizer.step()` 実行
  3. LRを元の値に復元
  4. `lr_scheduler.step()` 実行
- これにより scheduler の進行ロジックと整合を保ちます

### 7) ログとの対応

- stepログ:
  - `group_scale_auto`: そのstep時点の自動補正値
  - `group_scale_applied`: 実際に適用した値（現仕様では同値）
- epochサマリ:
  - `group_scale_auto`, `group_scale_applied` を記録
  - 補正値の推移をepoch単位で追跡できます

## 出力ファイル

`output_dir` 配下に出力されます。`output_name` 未指定時は `last` が使われます。

- stepログ:
  - `group_loss_logs+<output_name>.csv`
- epochサマリ（`--group_loss_epoch_summary` 有効時のみ）:
  - `group_loss_epoch+<output_name>.csv`

## CSV列定義

### stepログCSV

ヘッダ:

`global_step,epoch,group,subset_index,loss,ema_loss_group,count_group,timestep,bucket_reso,group_scale_auto,group_scale_applied`

- `global_step`: optimizer更新単位のstep
- `epoch`: 1始まり
- `group`: subsetに設定したgroup（未指定は `__ungrouped__`）
- `subset_index`: 全dataset通しのsubset識別子
- `loss`: そのstepのloss
- `ema_loss_group`: そのgroupのEMA値
- `count_group`: そのgroupの有効step累計
- `timestep`: diffusion timestep
- `bucket_reso`: バケット解像度（`WxH`）
- `group_scale_auto`: そのstepのgroupに対する自動補正倍率
- `group_scale_applied`: そのstepで実際に適用した倍率（MVPでは `group_scale_auto` と同値）

### epochサマリCSV

ヘッダ:

`epoch,group,ema_loss_end,count_epoch,mean_loss_epoch,group_scale_auto,group_scale_applied,baseline_loss_end,progress,ref_progress,ratio,ratio_ema,selected_for_boost`

補足: epochサマリ行は「epoch終了時」に1回書き出されます。  
このため列には、当該epochの実績値と、epoch末に計算した次epoch向け判定値が混在します。

- `epoch`: 対象の **終了したepoch番号**
- `ema_loss_end`: **当該epoch終了時点** のEMA（実績値）
- `count_epoch`: **当該epoch内** の有効step数（実績値）
- `mean_loss_epoch`: **当該epoch内** のgroup平均loss（実績値）
- `group_scale_auto`: **当該epochのstepで実際に適用された** 自動補正倍率
- `group_scale_applied`: **当該epochのstepで実際に適用された** 倍率（現仕様では `group_scale_auto` と同値）
- `baseline_loss_end`: epoch末計算時点で保持しているbaseline（未確定時は空）
- `progress`: epoch末計算値 `ema_loss_end / baseline_loss_end`（未確定時は空）
- `ref_progress`: epoch末計算値（そのepochの有効groupから算出、未計算時は空）
- `ratio`: epoch末計算値 `progress / ref_progress`（未計算時は空）
- `ratio_ema`: epoch末計算値。`ratio` をEMA平滑化した値（未定義時は空）
- `selected_for_boost`: epoch末判定で **次epoch向けブースト候補**（`ratio_ema >= threshold`）に選ばれたら `1`、それ以外は `0`（`threshold` の既定値は `1.10`）

## CLI指定例

### 最小例（stepログのみ）

```bash
accelerate launch sdxl_train_network.py \
  --dataset_config /path/to/dataset.toml \
  --output_dir /path/to/out \
  --output_name sample_lora \
  --group_loss_log
```

### 全step記録 + 100stepごとにflush + epochサマリ

```bash
accelerate launch sdxl_train_network.py \
  --dataset_config /path/to/dataset.toml \
  --output_dir /path/to/out \
  --output_name sample_lora \
  --group_loss_log \
  --group_loss_ema_beta 0.98 \
  --group_loss_log_every_n_steps 100 \
  --group_loss_epoch_summary
```

### group別LR自動補正を有効化（boost-only）

```bash
accelerate launch sdxl_train_network.py \
  --dataset_config /path/to/dataset.toml \
  --output_dir /path/to/out \
  --output_name sample_lora \
  --group_loss_log \
  --group_loss_epoch_summary \
  --group_lr_auto \
  --group_lr_auto_warmup_epochs 3 \
  --group_lr_auto_min_count 20 \
  --group_lr_auto_ratio_ema_beta 0.85 \
  --group_lr_auto_power 0.5 \
  --group_lr_auto_boost_threshold 1.10 \
  --group_lr_auto_max_scale 1.2 \
  --group_lr_auto_max_change 0.05
```

## dataset toml 記載例

### 1) キャラごとにgroupを付与

```toml
[general]
enable_bucket = true

[[datasets]]
resolution = [1024, 1024]
batch_size = 1

  [[datasets.subsets]]
  image_dir = "D:\train_data\hyuzu"
  class_tokens = "hyuzu"
  num_repeats = 20
  group = "hyuzu"

  [[datasets.subsets]]
  image_dir = "D:\train_data\ieimi"
  class_tokens = "ieimi"
  num_repeats = 20
  group = "ieimi"
```

### 2) 解像度違いの `[[datasets]]` をまたいで同一groupに集約

```toml
[general]
enable_bucket = true

[[datasets]]
resolution = [720, 720]
batch_size = 1

  [[datasets.subsets]]
  image_dir = "D:\train_data\small\hyuzu"
  class_tokens = "hyuzu"
  num_repeats = 20
  group = "hyuzu"

[[datasets]]
resolution = [1024, 1024]
batch_size = 1

  [[datasets.subsets]]
  image_dir = "D:\train_data\big\hyuzu"
  class_tokens = "hyuzu"
  num_repeats = 10
  group = "hyuzu"
```

この例では、`group=hyuzu` のログが2つの `[[datasets]]` をまたいで合算されます。
