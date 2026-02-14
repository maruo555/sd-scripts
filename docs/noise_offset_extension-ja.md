# noise_offset ランダム強度拡張（`train_network`系）

このドキュメントは、`--noise_offset_random_strength` の振る舞いを拡張する追加オプションの仕様をまとめたものです。
対象は `train_network.py` ベースの学習系（`sdxl_train_network.py` 含む）です。

## 1. 既存オプション（拡張前から存在）

以下は既存のオプションです。

- `--noise_offset=0.15`
  - Noise offset の基本値を指定します。
- `--adaptive_noise_scale=0.1`
  - `noise_offset + abs(mean(latents, dim=(2,3))) * adaptive_noise_scale` でチャネルごとに補正します。
- `--noise_offset_random_strength`
  - noise offset 強度をランダム化します。
  - 従来挙動は `0 ~ noise_offset` の一様乱数です。

## 2. 新規オプション（本拡張）

- `--noise_offset_random_min_ratio`（default: `0.0`）
  - `--noise_offset_random_strength` 有効時の最小比率（0〜1）。
- `--noise_offset_random_max_ratio`（default: `1.0`）
  - `--noise_offset_random_strength` 有効時の最大比率（0〜1）。
- `--noise_offset_random_min_ratio_sched`（default: `None`）
  - `min_ratio` を進行率で変えるスケジュール。
  - 書式: `"p1:r1,p2:r2,..."`（例: `"0.0:0.8,0.5:0.6,1.0:0.75"`）
  - `p` は総stepに対する進行率、`r` は `min_ratio`。
  - 区間は線形補間、進行率が範囲外なら端点値を使います。
- `--noise_offset_random_max_ratio_sched`（default: `None`）
  - `max_ratio` を進行率で変えるスケジュール。
  - 書式: `"p1:r1,p2:r2,..."`（例: `"0.0:1.0,0.7:0.8"`）
  - `p` は総stepに対する進行率、`r` は `max_ratio`。
  - 区間は線形補間、進行率が範囲外なら端点値を使います。

## 3. 計算ルール

`--noise_offset_random_strength` が有効なとき、実際の `noise_offset` は次で決まります。

1. `min_ratio = noise_offset_random_min_ratio_sched(progress)`（sched未指定時は `noise_offset_random_min_ratio`）
2. `max_ratio = noise_offset_random_max_ratio_sched(progress)`（sched未指定時は `noise_offset_random_max_ratio`）
3. `ratio ~ Uniform(min_ratio, max_ratio)`
4. `effective_noise_offset = noise_offset * ratio`

補足:

- `min_ratio` / `max_ratio` / `sched` の ratio 値は 0〜1 です。
- `min_ratio <= max_ratio` が必須です（全進行率でチェックされます）。
- 固定値とスケジュールを同時指定した場合、指定された側はスケジュール値が優先されます。

## 4. 使用例

### 4.1 固定レンジでのランダム化

```bash
--noise_offset=0.15 \
--noise_offset_random_strength \
--noise_offset_random_min_ratio 0.6 \
--noise_offset_random_max_ratio 1.0
```

この場合、`noise_offset` は `0.09 ~ 0.15` の範囲でランダム化されます。

### 4.2 後半で揺らしを弱める（進行率スケジュール）

```bash
--noise_offset=0.15 \
--noise_offset_random_strength \
--noise_offset_random_min_ratio_sched "0.0:0.8,0.5:0.6,1.0:0.75" \
--noise_offset_random_max_ratio_sched "0.0:1.0,0.7:0.8"
```

- 学習序盤は `min=max=0.8` で固定に近い挙動にできます。
- 学習中盤は `max-min` を広げて揺らし幅を確保できます。
- 学習後半は `max-min` を狭めて安定化を狙えます。

### 4.3 序盤固定→中盤で揺らす→後半で揺らしを弱める

```bash
--noise_offset=0.15 \
--noise_offset_random_strength \
--noise_offset_random_min_ratio_sched "0.0:0.85,0.3:0.70,0.6:0.74,1.0:0.78" \
--noise_offset_random_max_ratio_sched "0.0:0.85,0.3:1.00,0.6:0.86,1.0:0.82"
```

- 序盤（`progress=0.0`付近）は `min=max=0.85` なのでほぼ固定 `noise_offset` です。
- 中盤（`progress=0.3`付近）は `0.70 ~ 1.00` まで広がり、ランダムな揺さぶりを強めます。
- 後半（`progress=1.0`付近）は `0.78 ~ 0.82` まで狭まり、揺らしを弱めて安定化させます。

## 5. 後方互換

以下は従来と同等です。

```bash
--noise_offset 0.15 --noise_offset_random_strength
```

デフォルト `min_ratio=0.0` / `max_ratio=1.0` のため、従来どおり `0 ~ noise_offset` の範囲でランダム化されます。
