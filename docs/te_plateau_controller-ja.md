# SDXL Text Encoder Plateau Controller

## 目的
- SDXL テキストエンコーダー (TE1/TE2) の LoRA 学習において、勾配ノルムが十分に落ち着いたタイミングで **学習率を 1 回だけ自動減衰** し、その後は停滞が続けば凍結して無駄な更新を抑える。
- GradScaler 利用時でも **未スケール勾配 (grad / scale)** を指標に採用し、`--skip_grad_norm` / `--grad_norm_log` と整合した値を扱う。
- TE1/TE2 を独立に監視し、片側のみ減衰・凍結させることができる。

## 動作の流れ
1. **大局的減衰判定**: 指数移動平均 (EMA) の勾配ノルム (`global_ema`) とそのピーク (`global_peak_recent`、`global_window` で保持) を追跡し、**`--te-plateau-global-drop`** で定義した閾値を **`--te-plateau-global-patience`** 連続で下回ると大局フラグをセットする。フラグは `cooldown` 中や勾配が 90% まで回復した時点で自動解除される。
2. **局所停滞判定**: `--te-plateau-local-window` 内の未スケール勾配から中央値 (`median_local`)、分位幅 (`spread_local = pct90 - pct10`)、線形回帰傾き (`trend_ratio`) を計算し、直近ピーク (`peak_window` で保持) に対する比率を用いて閾値を **`--te-plateau-local-patience`** 連続で満たすかを監視する。
3. **減衰 (1 回のみ)**: 大局フラグが有効な状態で局所停滞も成立したら、学習率を `--te-plateau-decay-mult` 倍に一括更新し状態は `Decayed` へ移行する。減衰処理はこれ 1 回以降は実行されない。
4. **凍結**: 減衰後も停滞が続き、`--te-plateau-freeze-patience` 連続で `freeze_threshold_local` と `freeze_threshold_global` を満たした場合、該当 TE のパラメータを `requires_grad=False`・学習率 0 に設定して `Frozen` 状態にする。
5. **終了**: `Frozen` 状態になった TE は解除されない。以後はメトリクスだけを記録し、追加操作は行わない。

各イベント後は `--te-plateau-cooldown` ステップのクールダウンを挟み、直後の再判定を防ぐ。

### ウィンドウの役割
- **local_window** … 直近 `N` ステップの勾配を蓄積し、中央値・分位幅・線形回帰傾きを算出するための母集団です。局所的に横ばいになっているかを測定します。
- **peak_window** … `local_window` とは別に、同区間で観測された値から**上位 5% を捨てたうえでの最大値**を保持します。`median_local / peak_recent` を計算することで、「直近ピークからどれだけ落ち込んだか」を追跡します。
- **global_window** … 指数移動平均 (EMA) を同様にトリム付きピークとして追跡し、より長期的な減衰を測るための窓です。ここでのピークに対する比率が `global_drop` を下回ると、大局的に十分落ちたと見なします。

## オプション一覧

### 共通設定
| オプション | 既定値 | 説明 |
|-------------|--------|------|
| `--te-plateau-enable` | `False` | プレート制御を有効化。SDXL テキストエンコーダーを学習対象に含む場合のみ意味を持つ。 |
| `--te-plateau-local-window` | `512` | 局所中央値・分位幅・傾きを計算する履歴窓。 |
| `--te-plateau-peak-window` | `4096` | 直近ピークを保持する窓。内部的には**上位 5% の値を除外した最大値**でピークを決め、スパイクの影響を抑える。 |
| `--te-plateau-global-window` | `8192` | グローバル EMA のピークを保持する窓。こちらも上位 5% を除外した最大値を使ってスパイク耐性を確保。 |
| `--te-plateau-local-patience` | `128` | 局所停滞判定を連続何ステップ満たせばよいか。 |
| `--te-plateau-global-patience` | `128` | 大局フラグを立てるのに必要な連続ステップ数。 |
| `--te-plateau-drop-threshold` | 既定なし | ローカル中央値/ピーク比 (`drop_ratio_local`) の共通閾値。未指定なら TE ごとの既定を使用。 |
| `--te-plateau-spread-limit` | 既定なし | 分位幅 (`spread_local`) の共通上限。 |
| `--te-plateau-trend-limit` | 既定なし | 正規化された線形回帰傾き (`trend_ratio`) の共通上限。 |
| `--te-plateau-global-drop` | 既定なし | グローバル EMA/ピーク比 (`drop_ratio_global`) の共通閾値。 |
| `--te-plateau-freeze-threshold-local` | 既定なし | 凍結判定時のローカル比の共通閾値。 |
| `--te-plateau-freeze-threshold-global` | 既定なし | 凍結判定時のグローバル比の共通閾値。 |
| `--te-plateau-freeze-patience` | `256` | 凍結へ移行するまでの連続ステップ数。 |
| `--te-plateau-decay-mult` | `0.5` | 減衰発火時に学習率へ掛ける係数 (単発)。 |
| `--te-plateau-ignore-steps` | `2048` | 判定を開始するまでのウォームアップ期間。 |
| `--te-plateau-cooldown` | `512` | 減衰・凍結後に判定を停止するステップ数。 |
| `--te-plateau-log-path` | 未指定 | 指定するとヘッダー付き CSV ログ (UTF-8) を出力。 |
| `--te-plateau-log-interval` | `500` | ログをファイルへフラッシュする間隔 (ステップ数)。 |

### TE 個別閾値
共通設定で指定した値は両 TE に適用されます。さらに `--te1-...` / `--te2-...` で同名のオプションを指定すると、その TE のみ上書きします。未指定の場合は下表の既定値が用いられます。

| オプション | TE1 既定値 | TE2 既定値 | 説明 |
|-------------|------------|------------|------|
| `--te1-plateau-drop-threshold` / `--te2-plateau-drop-threshold` | 0.34 | 0.34 | ローカル中央値/ピーク比の閾値。 |
| `--te1-plateau-spread-limit` / `--te2-plateau-spread-limit` | 0.22 | 0.25 | 分位幅の許容倍率。 |
| `--te1-plateau-trend-limit` / `--te2-plateau-trend-limit` | 4.0e-4 | 4.5e-4 | 正規化された線形回帰傾きの上限。 |
| `--te1-plateau-global-drop` / `--te2-plateau-global-drop` | 0.62 | 0.72 | グローバル EMA/ピーク比の閾値。 |
| `--te1-plateau-freeze-threshold-local` / `--te2-plateau-freeze-threshold-local` | 0.20 | 0.26 | 凍結判定時に要求するローカル比。 |
| `--te1-plateau-freeze-threshold-global` / `--te2-plateau-freeze-threshold-global` | 0.55 | 0.62 | 凍結判定時に要求するグローバル比。 |

各パラメータの目安（いずれも **「値を下回ったら」 条件成立**）:
- **drop_threshold** … `median_local / peak_recent` がこの値を下回るとピークから十分落ち込んだと判定。例: 0.34 ならピークの 34% 以下に沈んだときに条件が成立する。
- **spread_limit** … `pct90 - pct10` を `peak_recent` で割った値が上限未満なら局所変動が収束しているとみなす。値を小さくするほど「ほぼ一定に張り付いた」状態のみ停滞扱いになる。
- **trend_limit** … `trend_ratio = |傾き| / peak_recent` がこの値より小さければ横ばいと判断。小さいほど勾配の傾きがほぼゼロに近い状態を要求し、大きいと緩やかな減少でも停滞扱いにする。
- **global_drop** … `global_ema / global_peak_recent` が閾値を下回ったときに大局的減衰フラグをセット。値を小さくすると「より深い谷」でしか減衰を許可しない。
- **freeze_threshold_local / freeze_threshold_global** … どちらも閾値未満で凍結条件を満たす。低くするほど「もうほとんど動いていない」状態で凍結する、値を上げると早めに凍結が走る。

#### TE1 と TE2 で既定値が異なる理由
- **TE1 (ViT-L)** はパラメータ数が少なく収束が比較的早いため、`spread_limit` や `freeze` 閾値を厳しめに設定し「十分落ち着いたと判断してから」減衰・凍結するようにしています。
- **TE2 (BiG-G)** は変動幅が大きく勾配ボトムが浅い傾向があるため、`spread_limit` や `drop_threshold` をやや緩め、`global_drop` も 0.72 と高めに設定して「明確に谷ができたときだけ」減衰を許可する形にしています。

差異の読み取り方:
- TE2 をもっと粘らせたい場合は `--te2-plateau-global-drop` を 0.75 付近まで戻す、`--te2-plateau-drop-threshold` を 0.32〜0.33 に下げるなどして「さらに深い谷」を要求する方向に調整します。
- 逆に減衰が遅いと感じたら、共通値 (`--te-plateau-global-drop` など) で全体を緩めたうえで TE1/TE2 のどちらかだけ個別に上書きする、という流れで調整するのが扱いやすいです。

## ログ出力
`--te-plateau-log-path` を指定すると、以下の列構成で `--te-plateau-log-interval` ごとに CSV を追記します:

```
step,te_name,grad_norm,median_local,spread_local,trend_ratio,drop_ratio_local,drop_ratio_global,state,lr,event
```

イベント列には `decay` / `freeze` が記録されます。凍結後は指標のみが更新され、追加イベントは発生しません。

## 判定ロジック早見表

| 判定名 | 参照指標 / 窓 | 主な関連オプション | 条件成立のイメージ |
|--------|----------------|------------------|--------------------|
| 大局的減衰判定 | `global_ema`, **`global_peak_recent`（大局ピーク）** | `--te-plateau-global-drop`, `--te-plateau-global-patience` | 例: `global_drop=0.65` の場合、EMA が大局ピークの 65% 以下を 128 step 連続で維持するとフラグが立つ。 |
| 局所停滞判定（中央値） | `median_local`, **`peak_recent`（局所ピーク）** | `--te-plateau-drop-threshold`, `--te-plateau-local-patience` | 例: `drop_threshold=0.34` なら局所ピークの 34% 以下まで中央値が沈み、それを 128 step 保つと成立。 |
| 局所停滞判定（分位幅） | `spread_local = pct90 - pct10` | `--te-plateau-spread-limit` | 例: `spread_limit=0.22` のとき、幅が局所ピークの 22% 未満になれば「ほぼ一定」と判定。 |
| 局所停滞判定（傾き） | 線形回帰傾き (`trend_ratio`) | `--te-plateau-trend-limit` | 例: `trend_limit=4e-4` なら局所ピークに対する傾きが 0.0004 未満（ほぼ水平）で成立。 |
| 減衰イベント | 上記の大局フラグ＋局所 3 条件 | `--te-plateau-decay-mult` | 大局フラグが立ち、3 条件すべてが `local_patience` 連続で成立した瞬間に LR を `decay_mult` 倍に更新。 |
| 凍結判定 | `drop_ratio_local`, `drop_ratio_global` | `--te-plateau-freeze-threshold-local`, `--te-plateau-freeze-threshold-global`, `--te-plateau-freeze-patience` | 例: `freeze_threshold_local=0.2`, `freeze_threshold_global=0.55` なら局所ピークの 20% / 大局ピークの 55% 以下を 256 step 保つと凍結。 |
| クールダウン | イベント後のタイマー | `--te-plateau-cooldown` | 減衰や凍結直後は指定ステップ数だけ判定を休止し、再発火を防ぐ。 |

## 注意事項
- `--fused_optimizer_groups` や `--deepspeed` が有効な構成では現在サポートしていません。
- 凍結 (`Frozen`) 状態では勾配が計算されないため、`grad_norm` は最後に観測した値を保持します。再開は行わない設計のため、凍結後はそのまま停止状態として扱ってください。
- `--skip_grad_norm` / `--grad_norm_log` も未スケール勾配を利用するよう統一されているため、両機能のログで同じ値を確認できます。
- `--te-plateau-log-path` の CSV にはヘッダー行が付きます（`step, te_name, grad_norm, ... , event`）。Excel などに読み込む場合は UTF-8 を指定してください。

## おすすめプリセット

### 1. クイックスタート (既定値のまま使う)
```
--te-plateau-enable --te-plateau-log-path logs/te_plateau.csv
```
共通オプションは指定せず、TE1/TE2 の既定閾値をそのまま利用します。ログだけ取りたい場合に最小構成です。

### 2. 減衰をやや早めたい場合
```
--te-plateau-enable \
--te-plateau-global-drop 0.7 \
--te2-plateau-global-drop 0.68 \
--te-plateau-drop-threshold 0.38 \
--te-plateau-log-path logs/te_plateau.csv
```
共通 `global_drop` を 0.70 に緩めつつ、TE2 は個別に 0.68 に調整しています。`drop_threshold` も 0.38 に引き上げ、ピークから 38% 程度の落ち込みでも減衰を許容する設定です。

### 3. 凍結を厳しくしたい場合
```
--te-plateau-enable \
--te-plateau-freeze-threshold-local 0.18 \
--te-plateau-freeze-threshold-global 0.5 \
--te1-plateau-spread-limit 0.2 \
--te-plateau-log-path logs/te_plateau.csv
```
凍結条件を共通で引き締めつつ、TE1 のスプレッド閾値を 0.20 に下げています。よりフラットな状態になってから凍結したい場合に有効です。
