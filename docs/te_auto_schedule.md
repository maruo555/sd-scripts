# Text Encoder 自動スケジュール（relative_warmup）

## 目的
- SDXL の Text Encoder (TE1/TE2) をそれぞれモニタリングし、学習率の自動減衰や凍結まで任せたい。
- AMP/GradScaler のスケールに影響されない "0〜1" の相対指標で判定し、閾値設定を直感的にする。
- VRAM 消費や forward/backward を増やさず、LoRA 学習ループに最小限のフックで組み込む。

## アルゴリズム概要
1. `--te-monitor-start-step` まではベースラインを集める準備期間。
2. `--te-warmup-steps` のあいだ勾配ノルム（LoRA パラメータの L2）を収集し、EMA (`--te-baseline-ema-beta`) で基準値を決定。
3. 監視フェーズでは `score = clamp(metric / baseline, 0〜1)` を計算し、しきい値を下回る状態が `patience` 回続いたら学習率を `--te-decay-factor` 倍する。
4. `--te-auto-schedule freeze` の場合は、十分に減衰したあと `--te-freeze-threshold` 未満が続けば自動凍結。凍結後は Unet のみ更新。
5. 解析ログは CPU 側でバッファリングし、`--te-monitor-log-path` 指定時は 200 step ごとにまとめてディスクへ追記。最終 step でもフラッシュされる。

TE1/TE2 は独立した状態を持ち、`--te1-*` / `--te2-*` で個別に閾値やウォームアップ長を上書きできます。

## オプション一覧
| オプション | 既定値 | 説明 |
|-------------|:------:|------|
| `--te-auto-schedule {off,monitor,freeze}` | `off` | 自動制御モード。`monitor` は学習率減衰まで、`freeze` は凍結まで行う。 |
| `--te-monitor-start-step` | `0` | 監視を開始するグローバルステップ。ここまではベースラインの下準備のみ。 |
| `--te-warmup-steps` | `300` | ベースライン収集に使うステップ数。終了時に `score=1` とみなす基準が確定。 |
| `--te-decay-threshold` | `0.4` | `score` がこの値を下回り続けたら学習率を減衰。 |
| `--te-decay-patience` | `100` | 減衰条件を満たす必要ステップ数。0 を指定すると即判定。 |
| `--te-decay-factor` | `0.5` | 減衰時に掛ける倍率。 |
| `--te-decay-max` | `1` | 減衰できる最大回数。`-1` で無制限。上限到達後は次の判定で凍結候補。 |
| `--te-freeze-threshold` | `0.2` | freeze モード時、`score` がこの値を継続して下回れば凍結。 |
| `--te-freeze-patience` | `100` | 凍結判定に必要な連続ステップ数。0 で即判定。 |
| `--te-min-baseline` | `1e-6` | ベースラインの下限。基準が極端に小さいケースで凍結を遅延させる。 |
| `--te-baseline-ema-beta` | `0.9` | ベースライン算出に使う EMA の係数。1 に近いほど滑らか。 |
| `--te-monitor-log-interval` | `0` | `score`/`baseline` などのログを `n` ステップごとに記録。0 で定期ログ無効。 |
| `--te-monitor-log-path` | `None` | CSV/JSONL の保存先。指定時は 200 step ごとにまとめて追記し、学習終了時にフラッシュ。 |
| `--te-monitor-verbose` | `False` | 判定イベント（减衰/凍結/ベースライン確定）を標準出力へも表示。 |

### 個別オーバーライド
`--te1-warmup-steps`, `--te1-decay-threshold`, … のように `te1-` / `te2-` プレフィックス付きで、TE1/TE2 それぞれの閾値・ウォームアップ長・減衰倍率などを上書きできます。未指定時は全体設定を継承します。

## ログ出力内容
`--te-monitor-log-path` に CSV を指定した例:

```
step,te_index,te_name,metric,baseline,score,lr,action,warmup_remaining,decay_counter,freeze_counter,decay_applied,baseline_ready
200,0,TE1,1.12e-03,2.45e-03,0.46,5.0e-05,decay,0,0,0,1,True
400,1,TE2,5.87e-04,2.11e-03,0.28,5.0e-05,monitor,0,12,12,0,True
```

- `metric`: LoRA パラメータ勾配の L2 ノルム（unscale 後）。
- `baseline`: ウォームアップで確定した基準値。
- `score`: `metric / max(baseline, te_min_baseline)` を 0〜1 にクリップした指標。
- `action`: `warmup`, `baseline_finalized`, `monitor`, `decay`, `freeze`, `frozen` などの状態遷移。

## 推奨プリセット
1. **緩やかな自動減衰 (monitor モード)**  
   `--te-auto-schedule monitor --te-warmup-steps 150 --te-decay-threshold 0.5 --te-decay-patience 80`

2. **凍結まで任せる (freeze モード)**  
   `--te-auto-schedule freeze --te-decay-threshold 0.45 --te-freeze-threshold 0.15 --te-decay-max 3`

3. **TE2 を長めに粘らせる個別設定**  
   `--te-auto-schedule freeze --te2-warmup-steps 200 --te2-decay-threshold 0.55 --te2-decay-factor 0.7`

ログを観察しつつ `threshold` と `patience` を調整することで、収束スピードと表現力のバランスを取りやすくなります。学習率を外部スケジューラで操作している場合は、最小 LR (`optimizer` 側) と `te-decay-max` の整合に注意してください。
