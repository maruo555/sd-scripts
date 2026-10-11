# SDXL LoRA 推奨学習設定

キャラクターの再現度を優先しつつ、過学習を抑えて柔軟に扱える LoRA を作るための設定

## 特徴

- VRAM 12GB または 16GB を想定
- batch size 1  (2以上にすると上手くいかない)
- fp16  
- rank 4～6  
- 画像数が少ない前提で dropout を高めにし、2万stepくらいじっくり学習する   
   例：
   5～10枚くらいの画像を用意し、各2～4つ程度のトリミング違いを用意して20枚くらいにし
   各 20～30 repeatにして 1 Epock 500stepくらい x 40 Epoc = 20000 stepくらい) 
- タグは独自タグのみで最小限に絞る  
- 本 fork の独自オプション群で過学習を抑えつつ柔軟性を高める  
- LoRA完成後、使用時に LoRA Block Weight で調整する前提  

## 全オプション一覧（学習コマンド）

| 設定項目 | 値 | 解説 | 補足説明 |
|---|---|---|---|
| **基本設定** |  |  |  |
| `--num_cpu_threads_per_process` | 8 | DataLoader など CPU スレッド数 | `accelerate launch` 側の指定 |
| 実行スクリプト | `sdxl_train_network.py` | SDXL LoRA 学習の本体スクリプト |  |
| `--pretrained_model_name_or_path` | `D:\models\base_model.safetensors` | 学習元のモデル |  |
| `--dataset_config` | `D:\datasets\example\dataset_config.toml` | dataset 設定ファイル | 下の TOML の例を参照 |
| `--prior_loss_weight` | 1.0 | prior loss の重み | prior 付き学習時の重み係数 |
| `--output_dir` | `..\lora_output` | 出力先ディレクトリ |  |
| `--output_name` | `loraABC` | 出力ファイル名のベース |  |
| `--learning_rate` | 3.5e-4 | UNet 側の基本学習率 | 3.5e-4か4e-4くらいが良い |
| `--max_train_epochs` | 40 | 最大エポック数 |  |
| `--optimizer_type` | AdamW8bit | 8bit AdamW |  |
| `--sdpa` | 有効 | SDPA を使う | 環境依存で有効化される --xformersが使えるならそちらでも可 |
| `--mixed_precision` | fp16 | 学習の混合精度 | fp16 前提 |
| `--save_precision` | fp16 | 保存精度 |  |
| `--seed` | 39 | 乱数シード | 何でもよいはず |
| `--save_model_as` | safetensors | 保存形式 |  |
| `--save_every_n_epochs` | 1 | エポックごとの保存間隔 |  |
| `--max_data_loader_n_workers` | 1 | DataLoader の worker 数 |  |
| **LoRA 基本設定** |  |  |  |
| `--network_module` | `networks.lora` | LoRA 実装モジュール |  |
| `--network_dim` | 4 | LoRA rank | 低 rank を基本に調整 4,5,6 あたりを使用 alphaは1(デフォルト) |
| `--network_args`（rank_dropout） | `rank_dropout=0.2` | LoRA の rank dropout | 過学習を防ぐ |
| **解像度・ノイズ・キャッシュ** |  |  |  |
| `--enable_bucket` | 有効 | バケット有効化 | TOML 側でも指定 |
| `--min_bucket_reso` | 384 | バケット最小解像度 |  |
| `--max_bucket_reso` | 1024 | バケット最大解像度 |  |
| `--noise_offset` | 0.15 | ノイズオフセット | 何となく設定 |
| `--adaptive_noise_scale` | 0.1 | 適応ノイズスケール | 何となく設定 |
| `--network_dropout` | 0.3 | LoRA dropout | 過学習を防ぐ |
| `--cache_latents` | 有効 | latent をキャッシュ | 速度・安定性向上 |
| **Text Encoder 関連** |  |  |  |
| `--text_encoder_lr` | 2e-4 | TE 共通の学習率 | 独自オプションで上書きしている |
| `--text_encoder_lr1`（独自） | 3e-4 | TE1 の学習率 | SDXL 2 系統 TE を分離 |
| `--text_encoder_lr2`（独自） | 2e-4 | TE2 の学習率 |  |
| `--downscale_freq_shift`（独自） | 有効 | 時間埋め込みの周波数ダウンスケールを 1.0 側に戻す | キャラ LoRA の identity 定着を狙う |
| `--te_mlp_fc_only`（独自） | 有効 | TE の学習対象を MLP (FC) 層のみに限定 | 語彙を固めつつ柔軟性を残す狙い |
| **安定化・平均化関連** |  |  |  |
| `--grad_norm_mode`（独自） | stable または stable_no_threshoff | skip_grad_norm の推奨プリセット | スパイクとなるstepの学習をskipして安定重視の挙動に寄せる stableの設定がいいと思っていたが、stable_no_threshoff でもいいかもしれない |
| `--rank_log`（独自） | 有効 | rank の変動や傾向を確認するためのログ出力 | 解析用のログを残すオプション 学習中の rank の挙動や偏りを後から見返したいときに役立つ |
| `--rank_log_mode`（独自） | per_module | rank_log の集計粒度 | `per_module` はモジュールごとに記録する 詳しく傾向を見たいとき向け |
| `--avg_cp`（独自） | 有効 | エポック平均（SWA/EMA 相当）を有効化 | LoRA 部分のみが対象 エポックを重みづけ平均しながら学習を進める(安定熟成狙い) |
| `--avg_cp_mode`（独自） | promote | avg_cp の動作モード | `live` は従来どおり平均重みをそのまま学習へ反映 `shadow` は raw と center を比較してログと best 保存だけ行う `promote` は比較結果が良いときだけ center を次 epoch に採用 推奨は `promote` |
| `--avg_window`（独自） | 4 | 平均に使う ckpt 数 | 窓サイズ |
| `--avg_begin`（独自） | 0.6 | 平均を開始する進行率 | 進行率 60% 以降 |
| `--avg_mode`（独自） | ema | 平均方法 | 直近ほど重くする EMA |
| `--avg_promote_pick`（独自） | fixed | promote 候補の選び方 | `promote` の `fixed` は `avg_mode` で指定した候補だけを採点して使う `best` は proxy bank 上で `ema` と `uniform` の両方を採点して良い方を選ぶ。`shadow` は常に両方を採点する |
| `--avg_shadow_bank_size`（独自） | 12 | proxy bank に保持するサンプル数 | 学習タグが多い場合は必要に応じて数を増やす |
| `--avg_save_last_candidates`（独自） | 無効 | 最終 raw / center を追加保存 | `shadow` / `promote` 専用 `<output_name>_raw.safetensors` と `<output_name>_center.safetensors` を保存する |
| `--avg_reset_stats`（独自） | `--no-avg_reset_stats` | 平均後に optimizer 統計をリセット | 推奨は `--no-avg_reset_stats` `promote` ではまず reset なしを試し、不安定なら戻す `live` も伸び鈍化が気になるなら reset なしを試す価値が高い。注: `AdamW8bit` は現行のリセット対象である `exp_avg` / `exp_avg_sq` / `exp_avg_max` を optimizer state に持たないため、実質何もリセットされない |
| `--fp16_safe_norms`（独自） | 有効 | fp16 + 小バッチで学習安定性を向上 | フェイク量子化とセットで運用 |
| **学習率スケジュール** |  |  |  |
| `--lr_scheduler` | constant_with_warmup | 学習率スケジューラ | 初期のみ 学習率をwarmup (最初に学習率0→指定値まで線形に上昇) |
| `--lr_warmup_steps` | 0.05 | 学習率warmup 比率 | 全体の 5% |

## dq_delta 関連オプション

量子化なしを基準にしつつ、画風・ポーズの柔軟性と再現性のバランスを試すための設定です。
以下は**固定mulで比較するときの共通設定**です。場所別配分と従来の自動調整は後の別表から選びます。

| 設定項目 | 値 | 解説 | 補足説明 |
|---|---|---|---|
| `--dq_delta_bits`（独自） | 8 | dq_delta の量子化ビット数 | 固定8bit。追加コストと画像の変化は環境・データ・設定による |
| `--dq_delta_granularity`（独自） | channel | dq_delta の粒度 | チャネル別に統計 |
| `--dq_delta_stat`（独自） | rms | dq_delta の統計基準 | range_mul の基準 |
| `--dq_delta_range_mul`（独自） | 比較する値（CLI既定3.0） | 全体の基本mul | 場所別指定がある部分だけ上書き。一律診断の5点は2.70 / 3.15 / 3.45 / 3.75 / 4.05 |
| `--dq_delta_mode`（独自） | stoch | dq_delta の丸め方式 | 確率的丸め |
| `--dq_delta_begin_after_lr_warmup`（独自） | 有効 | 学習率のwarmup完了後からdq_delta を開始 | 事前に学習率のwarmupをすることで量子化を安定させる |
| `--dq_delta_scope`（独自） | both | TE込みの意図を明示 | 現行の通常学習では`unet`指定でもstep処理後にTEが量子化される。場所別CLIはこの挙動を維持 |
| `--dq_delta_log`（独自） | 有効 | dq_delta の統計ログを出力 | クリップ率等の確認用 |
| `--dq_delta_log_scope`（独自） | both | TEとUNetをログ集計 | 旧`scope=unet`のまま学習する場合も、TEのログが必要なら明示する |

### 一律mulと、attn2・TE高mul型の比較

SDXLの`--network_module networks.lora`用です。診断ファイルには依存せず、学習コマンドへ直接書けます。
研究ではattn2・TE高mul型に好ましい質感や柔軟性が観察されたため比較候補に追加しました。
全データに有効、または診断スコアが良ければ高画質になる、という意味ではありません。

| 配分 | 追加するmul引数 | 読み方 |
|---|---|---|
| 一律の基準 | `--dq_delta_range_mul 3.75` | 全部位3.75の例。値は診断と画像評価に応じて選ぶ |
| attn2・TE高mul型の固定参照（旧A） | `--dq_delta_range_mul 2.70 --dq_delta_range_mul_attn2 3.75 --dq_delta_range_mul_te 3.75` | その他UNet2.70、attn2 / TE1 / TE2は3.75 |
| Body基準の追加候補 | 基本mulを2.70または3.15、attn2用mulとTE共通mulを診断のBody代表値にする | Bodyが求まらない・高低が逆転する場合は自動追加しない。同一配分は再利用する |

新しく追加した4引数はすべて既定`None`です。

| オプション | 対象・優先順位 |
|---|---|
| `--dq_delta_range_mul_attn2` | UNetのattn2 Q/K/V/Out。未指定なら基本mul |
| `--dq_delta_range_mul_te` | TE1とTE2の共通値。未指定なら基本mul |
| `--dq_delta_range_mul_te1` | TE1だけ。共通TE値より優先 |
| `--dq_delta_range_mul_te2` | TE2だけ。共通TE値より優先 |

例えば固定参照に`--dq_delta_range_mul_te2 3.45`を加えると、TE2だけ3.45です。
指定先のLoRAがない場合はエラーになります。`--dq_delta_scope unet`のままでもTE用mulは適用されます。
既存の学習率・dropout・平均化等は別の設定です。新しいmul引数がそれらを変更することはありません。

**場所別指定では固定bits・RMSのdelta量子化が必須です。** `--dq_delta_auto_range_mul`、
`--dq_delta_bits_sched`、`--dq_quantize_z`、`--dq_delta_policy_file`は併用せず、過去のコマンドから外してください。
`--resume`では保存時と同じ宣言・配分を維持します。
詳しいコマンド例・制約は[量子化ガイド](dq_delta_mechanism-ja.md#場所別の固定mulをcliで指定する)を参照してください。

標準診断はTE込み固定・dropout OFFが既定で、必要なら`--dq-profile-dropout-on`でONも追加確認できます。
これは**診断の比較方法**で、通常学習の`--network_dropout` / `rank_dropout`を外す指定ではありません。
新標準を最初から最後まで通す実SDXL GPU受入はまだ未実施です。
[検証済み範囲と次の受入手順](dq_standard_validation-ja.md)を参照してください。

### 自動mul調整・bits schedule（場所別固定mulとは別）

以下は従来の比較用設定例です。上の場所別mulの引数とは混ぜずに使います。
**最新仕様では`both`にしても、mulの自動判定はUNetの値を使います。**
通常は下記のように、UNet＋TEを記録しつつ、従来の`unet`指定と同じUNet統計で調整します。

```text
--dq_delta_scope both --dq_delta_log_scope both --dq_delta_auto_scope unet
```

未指定でも通常の`unet`／`both`学習はUNet統計を使います（TE専用時はTE）。
UNet＋TEの統合統計へ意図的に変更したい場合のみ`--dq_delta_auto_scope both`を明示します。
調整されるのはTEにも使う共通mulです。UNetだけのmulを調整する機能ではありません。
2026-10-11の変更では、旧`both`実装との互換性より、`unet`で運用していた自動判定を今後も保つことを優先しました。
場所別固定mulと自動調整の併用はエラーです。場所別に自動調整する機能はありません。
この対象選択の変更はCPU検証済み、GPU学習での受入は未実施です。

| オプション | 従来の設定例 | 意味 |
|---|---|---|
| `--dq_delta_auto_range_mul` | 有効 | clip_rateを見てmulを自動調整 |
| `--dq_delta_auto_scope` | `unet` | 調整の判定に使う統計の対象。量子化やログ対象とは独立 |
| `--dq_delta_auto_preset` | clip_rate_high | 目標clip帯。`clip_rate_low_auto`は誤差負荷に応じてlow帯からmid帯へ退避 |
| `--dq_delta_auto_init_range_mul_from_band` | 有効 | 目標帯から初期mulを設定（auto有効、RMS時） |
| `--dq_delta_auto_use_raw` | 有効 | EMAに加えてraw統計も判定に使う |
| `--dq_delta_bits_sched` | `0.0:8,0.9:10`（任意） | 終盤でbitsを変更。固定8bitとは別条件 |

### 量子化のヒント

- 量子化はしないのが一番無難
- 最初から量子化有で学習するとディティールが弱くなりやすいが、うまく行けば柔軟性が増す
- まず量子化無で学習し、`--network_weights` でその重みを読み込んで量子化ありで追加学習するのもよい
- `clip_rate_low` が合うデータもあるが、lowを維持するための量子化誤差が重くなる場合は `--dq_delta_auto_preset clip_rate_low_auto` でmid帯へ自動退避させる選択肢がある
- `--dq_delta_bits_sched "0.0:8,0.9:10"` で前半を 8bit、最後 10% だけ 10bit にする設定は、うまくいったこともあるが基本的には不要。ずっと 8bit のままで問題ない
- `--dq_quantize_z --dq_delta_mode det` は 10% 程度高速化する。学習は破綻なくできたが、内部で行われる丸め処理の場所が変わるため結果は通常設定と同等ではなくなる。量子化による自由度アップ性能が劣るようなので不採用

## 検証メモ

### TE の warmup を長くし、途中で TE を freeze する実験

通常設定:

```bat
--lr_scheduler constant_with_warmup --lr_warmup_steps 0.05
```

実験設定:

```bat
--lr_scheduler constant_with_warmup --lr_warmup_steps 0.05 --te1_lr_warmup_steps 0.1 --te2_lr_warmup_steps 0.1 --te1_freeze_at 0.85 --te2_freeze_at 0.85
```

- TE の warmup を長くすると TE の学習が薄まり、全体として学習が弱くなる
- 通常の U-Net と同期した warmup の方がよい
- TE を後半で freeze しても、特に良い効果はなかった

## データセット設定例（dataset_config.toml）
```
キャラクタータグ二人分 character_a,character_b
画像ごとの捨てタグ unique_01,unique_02... 画像1種類に1つ
1つのフォルダに同じ画像のトリミング違いの画像を1～4個ほど入れる
その他1girlなどのタグは付けない

[general]
enable_bucket = true
shuffle_caption = false

[[datasets]]
resolution = [720,720]
batch_size = 1

  [[datasets.subsets]]
  image_dir = 'D:\datasets\example\character_a\source_01\'
  class_tokens = 'character_a,unique_01'
  num_repeats = 20
  flip_aug = false

  [[datasets.subsets]]
  image_dir = 'D:\datasets\example\character_a\source_02\'
  class_tokens = 'character_a,unique_02'
  num_repeats = 20
  flip_aug = false

  [[datasets.subsets]]
  image_dir = 'D:\datasets\example\character_a\source_03\'
  class_tokens = 'character_a,unique_03'
  num_repeats = 20
  flip_aug = false

  [[datasets.subsets]]
  image_dir = 'D:\datasets\example\character_a\source_04\'
  class_tokens = 'character_a,unique_04'
  num_repeats = 20
  flip_aug = false

  [[datasets.subsets]]
  image_dir = 'D:\datasets\example\character_a\source_05\'
  class_tokens = 'character_a,unique_05'
  num_repeats = 20
  flip_aug = false

  [[datasets.subsets]]
  image_dir = 'D:\datasets\example\character_b\source_01\'
  class_tokens = 'character_b,unique_06'
  num_repeats = 20
  flip_aug = false

  [[datasets.subsets]]
  image_dir = 'D:\datasets\example\character_b\source_02\'
  class_tokens = 'character_b,unique_07'
  num_repeats = 20
  flip_aug = false

  [[datasets.subsets]]
  image_dir = 'D:\datasets\example\character_b\source_03\'
  class_tokens = 'character_b,unique_08'
  num_repeats = 20
  flip_aug = false

  [[datasets.subsets]]
  image_dir = 'D:\datasets\example\character_b\source_04\'
  class_tokens = 'character_b,unique_09'
  num_repeats = 20
  flip_aug = false

  [[datasets.subsets]]
  image_dir = 'D:\datasets\example\character_b\source_05\'
  class_tokens = 'character_b,unique_10'
  num_repeats = 20
  flip_aug = false

```
