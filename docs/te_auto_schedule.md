# Text Encoder 自動スケジュール（EMA Plateau）

## 目的
- SDXL の Text Encoder (TE1/TE2) を別々に監視し、勾配ノルムの動きが高原（plateau）に入ったら自動的に学習率を下げ、必要に応じて凍結する。
- 平滑化された EMA 比（高速 EMA / 低速 EMA）と連続観測カウンタを用いて、高原状態を安定して検出する。
- 追加の forward/backward を発生させず、VRAM を消費しない純粋な統計的判定で制御する。

## アルゴリズム概要
1. 各ステップで Text Encoder の LoRA 勾配ノルム（L2）を取得し、`ema_fast`（α ≈ 0.2）と `ema_slow`（α ≈ 0.02）の指数移動平均を更新。
2. 比率 `ratio = ema_fast / ema_slow` が `--te-plateau-ratio` を下回った状態が `--te-plateau-patience` 回続けば plateau 検知と見なす（`--te-plateau-min-step` 以前は判定しない）。
3. plateau を検知したら `--te-plateau-decay-factor` をかけて TE の学習率を減衰。`--te-plateau-decay-limit` 回まで繰り返す。
4. `--te-auto-schedule freeze` の場合、減衰上限に達した後に `ratio < --te-plateau-freeze-ratio` が `--te-plateau-freeze-patience` 回継続すると凍結。凍結後は Unet のみ更新。
5. ログは CPU 側でバッファリングし、`--te-monitor-log-path` 指定時に 200 step ごとにまとめて書き込み（最終ステップで必ず flush）。

## オプション一覧
| オプション | 既定値 | 説明 |
|-------------|:------:|------|
| `--te-auto-schedule {off,monitor,freeze}` | `off` | 自動制御のレベル。`monitor` は減衰まで、`freeze` は最終凍結まで行う。 |
| `--te-ema-fast-alpha` | `0.2` | 高速 EMA の α。値を大きくすると直近の変化に敏感になる。 |
| `--te-ema-slow-alpha` | `0.02` | 低速 EMA の α。小さいほど長期傾向を重視。 |
| `--te-plateau-ratio` | `0.95` | `ema_fast / ema_slow` がこの値未満になったら plateau 候補。 |
| `--te-plateau-patience` | `80` | plateau 比を下回り続ける必要ステップ数。 |
| `--te-plateau-min-step` | `1000` | このステップまでは判定しない。初期ノイズ対策。 |
| `--te-plateau-decay-factor` | `0.5` | plateau 検知時に学習率へ掛ける係数。 |
| `--te-plateau-decay-limit` | `2` | 減衰可能な最大回数（`-1` で無制限）。 |
| `--te-plateau-freeze-ratio` | `0.9` | 凍結モードで利用。減衰上限到達後、この比率を下回り続けたら凍結候補。 |
| `--te-plateau-freeze-patience` | `160` | 凍結比を下回る必要ステップ数。 |
| `--te-monitor-log-path` | `None` | CSV/JSONL 出力先。指定時は 200 step ごとにまとめて追記。 |
| `--te-monitor-verbose` | `False` | 減衰や凍結イベントを標準出力へ表示。 |

### 個別オーバーライド
`--te1-plateau-ratio`, `--te2-plateau-decay-factor` のように `te1-` / `te2-` プレフィックス付きオプションで、TE1/TE2ごとの α やしきい値・パティエンスを個別指定できます。未指定時は全体設定を継承します。

## ログ出力内容
`--te-monitor-log-path` で CSV を保存した場合の例:

```
step,te_index,te_name,metric,ema_fast,ema_slow,ratio,lr,action,plateau_counter,decay_count,freeze_counter,plateau_ready
4090,0,TE1,0.0121,0.0109,0.0113,0.964,5.0e-05,decay,81,1,0,True
6279,1,TE2,0.0154,0.0143,0.0158,0.907,5.0e-05,decay,85,2,0,True
```

- `metric`: 勾配ノルム (L2)。
- `ema_fast` / `ema_slow`: それぞれの EMA 値。
- `ratio`: `ema_fast / ema_slow`。
- `action`: `monitor`, `decay`, `freeze`, `frozen` などの状態遷移。
- `plateau_counter`: 連続してしきい値を下回ったカウンタ。
- `decay_count`: 実行済みの減衰回数。
- `freeze_counter`: 凍結候補カウンタ（freeze モード時）。

## 推奨プリセット
1. **標準的な減衰のみ（monitor）**  
   `--te-auto-schedule monitor --te-plateau-ratio 0.95 --te-plateau-patience 80`

2. **凍結まで任せる（freeze）**  
   `--te-auto-schedule freeze --te-plateau-decay-limit 3 --te-plateau-freeze-ratio 0.9 --te-plateau-freeze-patience 160`

3. **TE2 を粘らせる個別設定**  
   `--te-auto-schedule freeze --te2-plateau-ratio 0.97 --te2-plateau-decay-factor 0.7`

ログで `ratio` と `plateau_counter` を確認しながら、検出タイミングが適切かどうか調整してください。EMA α を変えると plateau 判定の敏感さが変化します。 внешний学習率スケジューラと併用する場合は、`te-plateau-decay-limit` と最小 LR の整合に留意してください。
