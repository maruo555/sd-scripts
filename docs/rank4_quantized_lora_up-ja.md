# rank 4 Quantized LoRA-Up（C0）設計・利用・検証ガイド

この文書は、dq_delta の rank 4 専用高速経路「Quantized LoRA-Up（C0）」と、同時に導入した scope semantics version 2 をまとめたものです。

- dq_delta 全体の仕組み: [dq_delta_mechanism-ja.md](dq_delta_mechanism-ja.md)
- ログと auto の仕様: [dq_delta_autotune_spec-ja.md](dq_delta_autotune_spec-ja.md)
- 既存 Triton A/B と導入方法: [triton_windows_setup.md](triton_windows_setup.md)

## C0 の位置づけ

通常の delta 量子化は、概念上、次の順で処理します。

```text
z = LoRA-Down(x)
delta = LoRA-Up(z)
delta_q = fake_quant(delta)
```

C0 は rank 4 の `LoRA-Up` と channel RMS scale、stochastic fake quant を専用 Triton pipeline にまとめます。適用できるときの経路優先順位は次の通りです。

```text
C0 rank-4 Quantized LoRA-Up
  └─ fallback → 既存 Triton A/B
                   └─ fallback → PyTorch
```

C0 はデフォルトで OFF です。C0 が使えない shape や設定でも学習を止めず、従来経路へ戻します。rank 5 など rank 4 以外は通常経路を使います。

## CLI

追加オプションは次の通りです。

| オプション | 既定 | 説明 |
| --- | --- | --- |
| `--dq_delta_triton_fused_up_mode {off,c0}` | `off` | `c0` で rank 4 Quantized LoRA-Up を要求します。`--dq_delta_use_triton` が必要です。 |
| `--dq_delta_triton_fused_up_scope {unet,te,both}` | `unet` | C0 を試す scope。実際の対象は `--dq_delta_scope` との共通部分です。 |
| `--dq_delta_triton_fused_up_diagnostics` | OFF | shape、適用回数、fallback 理由などの詳細診断を収集します。ベンチマーク時は OFF を推奨します。 |

UNet の rank 4 学習で C0 を試す例:

```text
--network_dim 4
--dq_delta_bits 8
--dq_delta_granularity channel
--dq_delta_stat rms
--dq_delta_mode stoch
--dq_delta_scope unet
--dq_delta_use_triton
--dq_delta_triton_stats
--dq_delta_triton_fused_up_mode c0
--dq_delta_triton_fused_up_scope unet
```

詳細な適用状況を確認する最初の短縮 run では、さらに次を指定します。

```text
--dq_delta_triton_fused_up_diagnostics
```

Text Encoder も C0 の候補にする場合は、明示的に次のようにします。

```text
--dq_delta_scope both
--dq_delta_triton_fused_up_scope both
```

TE もコード上は同じ eligibility 判定を通ります。ただし既定を UNet 専用にしているのは、最初に検証する範囲を狭く保つためです。TE を含む実学習の品質・速度は未検証なので、短縮 run と A/C 比較を先に行ってください。

## C0 の適用条件

C0 は次の条件をすべて満たす forward だけに適用されます。

```text
--dq_delta_triton_fused_up_mode c0
--dq_delta_use_triton
dq_delta の対象が delta（--dq_quantize_z ではない）
bits=8
granularity=channel
stat=rms
mode=stoch
LoRA rank=4
LoRA-Up が bias なし Linear
activation が contiguous CUDA 3D tensor（NLC）、dtype=float16、最終次元=4
LoRA-Up weight が同じ CUDA device 上の contiguous float32
rows=N*L が 1..2048
autograd が有効な学習 forward（`torch.no_grad()` ではない）
gradient checkpointing なし
C0 が扱える basic stats、または stats 不要の step
検証済み GPU capability と channel/launch dispatch
```

初期 dispatch で有効な CUDA capability は RTX 5080 の `(12, 0)` です。正式な performance dispatch は、activation=`float16`、LoRA-Up weight=`float32` の次の組み合わせです。

| rows bucket | channel | stats mode |
| --- | --- | --- |
| `1..128` | `320, 640, 768, 1280, 2560, 3072, 5120, 10240` | `none`, `basic` |
| `129..512` | `320, 640, 768, 1280, 2560, 3072, 5120, 10240` | `none`, `basic` |
| `513..2048` | `320, 640, 768, 1280, 2560, 3072, 5120, 10240` | `none`, `basic` |

この table は、各 bucket の代表・境界 shape について、本番と同じ PyTorch 乱数生成を含む forward+backward が既存 A/B に対して `1.05x` 以上になることを採用条件として登録しています。未知の GPU capability、未登録 channel、未対応 shape は通常経路へ fallback します。起動・診断情報には PyTorch/CUDA/GPU capability と Triton version を残し、別環境の結果を区別します。

この dispatch を推論や forward-only の採用判定へ流用してはいけません。学習用の forward+backward で採用した table であり、basic stats の `rows=513, channel=10240` では forward-only が `0.865x` だった測定例もあります。そのため、学習中でもサンプル生成などの `torch.no_grad()` forward は C0 を使わず通常経路へ戻します。forward-only 経路を追加する場合は、別の性能 table と採用基準で検証してください。

gradient checkpointing 中は backward 再計算と統計収集の扱いを分離して検証する必要があるため、C0 は現在無効です。既存 Triton A/B または PyTorch 経路へ戻ります。

## 数値仕様と学習結果への影響

C0 は従来と同じ dq_delta の意味を保つよう設計していますが、forward の全要素が PyTorch と bit-for-bit 一致することを要件にはしていません。Triton と PyTorch の演算順、丸め境界、reduction 順序が異なるためです。

数値上の固定事項:

- channel RMS の `eps` は `1e-8`
- scale は FP32
- 初期 C0 は LoRA-Up weight が FP32 の場合だけ対象
- stochastic rounding の乱数は PyTorch で FP32 tensor として1回だけ生成
- C0 が失敗した場合も同じ乱数 tensor を既存 Triton A/B または PyTorch fallback で再利用
- custom backward は通常の LoRA-Up と同じ勾配式を使い、native PyTorch oracle と照合

このため、C0 の意図は「量子化設定を変えて学習結果を変えること」ではなく「同じ設定をより速く計算すること」です。ただし forward の微小な数値差が長時間学習で増幅される可能性はゼロではありません。synthetic correctness、短縮実学習、同一条件の A/C 交互 run、生成物評価の順に確認してください。

## stats と診断

basic stats を融合する場合、次の5値を packed FP32 tensor として返します。

```text
numel
clip_count
sumsq
xq_sumsq
xxq_sum
```

packed stats は `mark_non_differentiable` され、学習グラフの勾配対象になりません。

`full`、`per_module`、`near_zero_rate`、`ZeroRate`、`AbsMax`、`ScaleMin/Mean/Max` など C0 が持たない詳細統計が必要な step は通常の stats 経路へ fallback します。forward 全体を止めるのではなく、必要な詳細度を優先します。

`--dq_delta_triton_fused_up_diagnostics` は、scope、batch、length、channel、rank、dtype、stats mode、contiguous 状態、成功/fallback 理由を集計します。Triton import、compile、launch、runtime failure と eligibility 不一致は理由別に確認できます。診断収集自体にも Python 側のコストがあるため、性能の正式測定では OFF にします。

## 起動エラーと runtime fallback

設定矛盾は起動時エラーにします。

- `mode=c0` なのに `--dq_delta_use_triton` がない
- `--dq_delta_auto_scope` が apply scope の部分集合ではない
- `mode=c0` で fused-up scope と apply scope の共通部分が空
- scope や mode の値が不正

環境・tensor・kernel に依存する問題は warning と fallback にします。

- Triton を import できない
- 未検証 GPU capability または未登録 dispatch
- rank、dtype、shape、weight、stats、gradient checkpointing などの eligibility 不一致
- Triton compile、launch、runtime failure

Windows の初回 Triton compile では、Python 開発ヘッダ / import library と Visual Studio 2022 C++ toolchain / Windows SDK が必要になる場合があります。組み込み Python に `Python.h` や対応する `python310.lib` などがない場合、warning を出して通常経路へ fallback します。cache なしの初回起動要件は [triton_windows_setup.md](triton_windows_setup.md#インストールと検証環境) を参照してください。

失敗した kernel 構成は failure cache に記録し、同じ構成を毎回再試行しません。C0 を要求したのに学習終了まで成功が0回だった場合は、全rankの成功件数を集約して warning を出します。ベンチマークでは C0 成功が0件なら測定成功とはせず、非0の終了コードで失敗させます。

## scope semantics version 2

### 修正前の不具合

旧実装では、起動時に `--dq_delta_scope unet` を適用して TE を無効化しても、step 更新の `set_delta_quant_enabled(True)` が module の有効状態を直接上書きしていました。そのため dq_delta 開始後は TE が再有効化され、実際には `both` に近い適用になる場合がありました。

version 2 では次を分離し、実効状態を論理積で決めます。

```text
effective_enabled = runtime_enabled AND scope_allowed
```

step による ON/OFF は `runtime_enabled` だけを更新し、scope 制約を上書きしません。`networks/lora_lbw.py` に入るのはこの scope 修正だけで、C0 は実装しません。また、`lora_lbw.py` は C0 性能 baseline の対象外です。

### requested と resolved

起動時ログと保存 metadata には scope の指定値（requested）と実際の値（resolved）を分けて記録します。metadata には `ss_dq_scope_semantics_version=2` も保存します。`ss_dq_delta_scope_application` は適用方式を記録し、組み込みのversion 2 APIは `native`、旧networkへの再適用fallbackは `legacy`、適用不能は `unsupported`、dq_delta未設定は `not_configured` です。

- apply: `--dq_delta_scope`
- log: `--dq_delta_log_scope`。未指定時は apply を継承し、指定時も apply との共通部分へ制限
- auto: `--dq_delta_auto_scope`。未指定時は apply を継承し、apply の部分集合でなければ起動エラー
- C0: `--dq_delta_triton_fused_up_scope`。実際の対象は apply との共通部分

log scope が apply scope より広い場合は resolved scope を共通部分へ縮小し、warning を出します。共通部分が空なら dq_delta log を無効化します。

### resume と旧実動作の近似再現

dq_delta 有効時に `--resume` を使うと、version 2 より前の run から scope の実効動作が変わる可能性があるため、起動時に警告します。

旧実装で「指定は UNet だったが、step 開始後は TE も量子化されていた」状態をできるだけ近く再現する設定は次です。

```text
--dq_delta_scope both
--dq_delta_log_scope unet
--dq_delta_auto_scope unet
--dq_delta_triton_fused_up_mode off
```

これは旧バグの実効 scope を明示的な正常設定へ置き換えるものです。コード版、乱数列、optimizer state など他の差まで完全再現する保証ではありません。

## RTX 5080 synthetic 検証結果

RTX 5080 / CUDA capability `(12, 0)` で、次を確認しました。

- correctness の relative L2: およそ `1e-6`〜`7e-5`
- `grad_z` / `grad_weight`: native PyTorch oracle と bitwise 一致
- 本番相当の PyTorch 乱数生成を含む forward+backward の正式 grid:
  - stats なし: 既存 A/B baseline に対する最小 speedup `1.139x`
  - basic stats あり: 最小 speedup `1.088x`
- basic stats の最小値を示した条件の再測定は `1.131x`。短い CUDA Events 測定の揺らぎを考慮し、dispatch の採用下限は個々の一回の値ではなく `1.05x` としています。

正式 grid は、3つの rows bucket、8つの channel、`none` / `basic` の両 stats mode の代表・境界 shape を対象にしています。上記は synthetic な C0 対象処理の倍率であり、学習全体の倍率ではありません。当初候補の「学習全体で 3〜8%」はまだ未検証です。データ読み込み、UNet 本体、optimizer、対象外 module、fallback 率を含むため、実際の効果は C0 coverage に依存します。

正式評価では、同じ設定の A（C0 OFF）と C（C0 ON）を交互に実行して温度・キャッシュ・バックグラウンド負荷の偏りを減らします。step time、C0 coverage、fallback 理由、loss、DQ stats、生成物を合わせて比較してください。

## 検証コマンド

リポジトリ root と、CUDA/Triton を導入した同じ Python 環境で実行します。

correctness・fallback・gradient:

```bash
python tools/check_triton_lora.py --quick
python tools/check_triton_lora.py
```

CUDA Events による C0 と既存 A/B の比較:

```bash
python tools/benchmark_triton_lora.py --quick
python tools/benchmark_triton_lora.py --warmup 50 --iterations 1000 --repeats 7
python tools/benchmark_triton_lora.py --basic-stats --warmup 50 --iterations 1000 --repeats 7
```

ベンチマークの既定は production RNG です。baseline と C0 の両方で、実運用と同じく各 forward の PyTorch `torch.rand` 生成・確保を測定に含めます。`--basic-stats` は既存 Triton B+stats と C0 の packed basic stats を比較します。

`--fixed-rand` は、事前生成した乱数 tensor を再利用して kernel 算術だけを調べる内部向けモードです。production RNG を含む正式な性能判定には使いません。

```bash
python tools/benchmark_triton_lora.py --fixed-rand --case 468,1280
```

任意 shape:

```bash
python tools/benchmark_triton_lora.py --case 468,1280 --warmup 50 --iterations 1000 --repeats 7
```

VRAM は baseline と C0 を同じ process で順番に測らず、別 process で測定します。

```bash
python tools/benchmark_triton_lora.py --case 468,1280 --memory-mode baseline
python tools/benchmark_triton_lora.py --case 468,1280 --memory-mode c0
```

VRAM の正式指標は `torch.cuda.max_memory_allocated()` に基づく peak allocated です。allocator の履歴を含みやすい peak reserved は参考値として併記します。

RTX 5080 で production RNG を含め、baseline / C0 を別 process で測った結果は次の通りでした。`call delta = peak allocated - allocated before` です。

| shape | route | allocated before | peak allocated | call delta |
| --- | --- | ---: | ---: | ---: |
| `rows=468, channel=1280` | baseline | `20,658,176` | `26,663,936` | `6,005,760 bytes` |
|  | C0 | `12,138,496` | `18,599,936` | `6,461,440 bytes` |
| `rows=468, channel=10240` | baseline | `45,961,216` | `94,007,296` | `48,046,080 bytes` |
|  | C0 | `37,441,536` | `85,364,736` | `47,923,200 bytes` |

絶対 peak は `channel=1280` で `-30.2%`、`channel=10240` で `-9.2%` でした。一方、call delta はそれぞれ `+7.6%` と `-0.26%` で、概ね同等ですが 1280 では C0 が少し増えています。route ごとに persistent allocation（allocated before）が異なるため、絶対 peak の差を C0 kernel 単体の削減率とは解釈できません。学習全体の VRAM 効果は、実学習 process で改めて確認してください。

## 採用判断

C0 の採用条件は kernel 単体の倍率だけではありません。

1. correctness / fallback / gradient 回帰がすべて成功する
2. C0 成功件数が0ではなく、代表 shape の coverage が十分ある
3. A/C 交互の短縮実学習で step time が改善する
4. loss、DQ stats、生成物に許容できない差がない
5. VRAM peak allocated に実運用上の悪化がない

未知 GPU や未検証設定を dispatch table に追加するときも、同じ順序で検証してから有効化します。
