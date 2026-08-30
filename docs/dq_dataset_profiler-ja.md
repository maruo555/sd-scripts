# SDXL DQ Dataset Profiler 利用ガイド

## 1. この診断機能は何を調べるものか

SDXL DQ Dataset Profilerは、LoRA学習でdelta量子化を使ったときに、
datasetと`range_mul`の組み合わせが学習勾配へ与える数値的な変化を調べる機能です。

主に次の問いに答えます。

- このdatasetでは、量子化により通常範囲の勾配がどの程度変わるか。
- 一部の画像・timestepだけで大きな変形が発生していないか。
- 試したmulのうち、他候補より明確に強い変形を起こす候補はあるか。
- datasetごとに、mulへの反応曲線がどの程度違うか。

一方、現在の診断だけでは次を決めません。

- 最終生成画質が最も良いmul
- 量子化がno-quantより有益か
- 画風やキャラクター再現が良くなるか
- 40 epoch後の最良checkpoint

この区別は重要です。本機能が測るのは**数値的なSafety/Fidelity**であり、
最終画質のUtilityではありません。

## 2. 通常学習から分離している理由

公開入口は`python -m dq_profile`です。このorchestratorが、内部stage専用の
`sdxl_dq_dataset_profile.py`を必要な順序で起動します。診断入口は次を強制します。

- 診断専用にコピーしたtrainerとLoRA実装を使用する。
- 通常のモデル保存先、resume、tracker、sample生成へ書き込まない。
- 診断出力ディレクトリ以外へ成果物を書かない。
- 各Accelerate stageを`num_processes=1`、`num_machines=1`で起動し、ユーザー環境の分散設定を継承しない。
- DataLoaderをworker 0に固定し、分岐には固定済みreplay batchを使う。
- 同一snapshot、同一画像、同一noise、同一timestep、同一dropout条件で候補を比較する。
- 量子化乱数を候補名に依存させず、mul間でcommon random numbersを使う。

通常学習経路が診断コードをimportしないことは、研究中に守ってきた重要な隔離条件です。

## 3. 現在の実用診断で行う処理

通常利用の`canonical-v1`は、40 epochを最後まで学習する処理ではありません。
量子化開始直前の共通状態を複数回作り、その状態から再現性検査とLocal Body／Tail計測を
行う多段protocolです。各GPU stageは独立processとして起動し、stage間で暗黙のmutable stateを
共有しません。

### 3.1 Preflightと実行契約

最初にGPUを使わず、次を確認します。

- model、dataset TOML、各`image_dir`が存在する。
- 学習loaderと同じ拡張子・大文字小文字規則および非再帰探索で、各`image_dir`直下に画像があり、dataset全体で8画像以上、独立した`image_dir`が4 group以上ある。
- source-group prefixとworkerが返す画像keyを一致させるため、`image_dir`へ`.`／`..`のpath componentを含めず、どの階層にもsymlink、junctionなどのreparse pointを含めない。
- すべての有効な`image_dir` groupをprobeへ最低1件ずつ含められることを確認する。group数が検証済みprobe上限を超える設定は、部分的なconfidenceを出さず開始前に拒否する。
- `cache_latents`と両立しない`color_aug=true`または`random_crop=true`が、subset／dataset／`[general]`のfallback後に有効でないことを確認する。
- DreamBooth loaderに必須の`resolution`が、datasetまたは`[general]`のfallback後に定義されていることを確認する。
- TOMLの`[general]`／dataset／subset fallbackを解決し、batch・bucket設定（`bucket_no_upscale=false`を含む）が`canonical-v1`と一致することを確認する。
- CLIが`canonical-v1`と互換である。
- 通常checkpoint、dataset、repositoryと診断出力先が重ならない。
- Git HEAD、ソースhash、preset、model内容のSHA-256、dataset、source inventoryからprotocol fingerprintを作る。
- 各GPU workerの起動直前にmodel内容とsource inventoryを再度hash照合し、長い多段runの途中でmodel、画像、caption、cache sidecarが変化した場合は混在させず停止する。
- repositoryに追跡済みの未コミット変更がある場合は、HEADとの差分全体もbinary diffとしてhash化する。未追跡ファイルは対象外と明記する。
- 実画像数と`min(実画像数, 32)`であるprobe budgetを記録する。

`--dq-profile-preflight`ではここまで実行し、GPU stageを起動しません。
通常のpreflight／dry-runを含む全runで`execution_plan.json`も作り、GPU process数、
warmup境界、Prefix update数、Local probe数、固定grid、参考時間の算出条件を記録します。

### 3.2 量子化開始境界とsnapshot検算

通常学習コードと同じ規則で`dq_delta_begin_step`を求め、量子化開始直前までno-quantで
warmupします。`canonical-v1`では40 epoch相当の総stepと5% LR warmupから境界が決まります。

`standard`と`strict`は、同じ初期状態からSnapshot AとSnapshot Bを別processで作り、LoRA重み、
optimizer、scheduler、GradScaler、Guardian、RNG、replay位置などのfingerprintを比較します。
`quick`はSnapshot Aを1回だけ作り、後続のPrefix processが同じ境界を再現できたかを比較します。
どのmodeでも境界fingerprintが一致しなければmul比較を開始しません。Quickは独立snapshotを
1回減らすぶん速い一方、同じstageを2回作る検算深度はStandardより低くなります。

以下で時間例に使う小規模datasetは、通常学習が8,400 stepだったため、境界は
`8,400 × 0.05 = 420 step`でした。

### 3.3 Prefix parity gate

同じsnapshotから、no-quantとanchor候補`mul=3.15`について次を実行します。

| execution mode | short A | short B | long | 比較checkpoint | 合計branch update |
|---|---:|---:|---:|---|---:|
| `quick` | 8 | 8 | 16 | `0, 1, 4, 8` | 64 |
| `standard` | 8 | 8 | 16 | `0, 1, 4, 8` | 64 |
| `strict` | 64 | 64 | 128 | `0, 1, 32, 64` | 512 |

各modeでA対B、およびA対long runの同じ先頭prefixを比較します。sample、noise、timestep、
rank／network dropout、量子化乱数、Loss、LR、skip、勾配、LoRA重み、optimizer、scheduler、
GradScaler、Guardian、replay cursorを検査します。このstageは通常学習に近い経路を検査するため
dropout有効です。`quick`と`standard`は短いsmoke QA、`strict`は環境変更・リリース前・
再現性調査用のreference QAです。Quickは独立snapshot検算を1回減らします。同じ`PASS`でも深度が異なるため、`execution_mode`と
`qa_depth`をJSONとレポートへ別々に記録します。prefix gateまたはsource contractが失敗した場合、
Local計測へ進みません。

### 3.4 Local Body／Tail scan

ここが製品レポートの診断本体です。optimizer更新を行わず、同じ画像、noise、timestepで
no-quantと各固定mulの勾配を比較します。

- 画像数: Standard／Strictは8～32、Quickは8～16。各上限を超えてもprobe budgetは増えない。
- timestep: 4帯。
- no-quant: 3 noise replicas。
- 各mul: 2 noise replicas × 2 stochastic quant repeats。
- stateless量子化乱数を使い、共通候補間でcommon random numbersを保つ。
- dropoutを無効にした`structural_dropout_off` regimeで測る。
- module単位の勾配を集約し、Body、Tail、hard-safety、source別の不確実性を作る。

比較用branchの先頭replay windowは固定したままです。この固定windowにrepeat数の少ない
`image_dir` groupが含まれなかった場合だけ、DataLoaderを最大2 epoch分追加走査します。
不足groupを初めて含んだbatchだけをprobe用に保持し、全source groupを揃えてから画像を
round-robin選択します。追加batchはbranchの128-step prefixへ混ぜないため、候補間比較の
再現契約は変わりません。極端にrepeatが偏るdatasetでは、このcoverage走査ぶんだけ
Local計測開始前の時間が増える場合があります。

このLocal結果だけを通常レポートのSafety/Fidelityと候補削減に使用します。dropout有効の
128-step Trajectoryは研究専用の別channelであり、現在の製品入口では実行しません。

画像数を`I`、そのstageで測るmul数を`M`とすると、概念的なLocal probe数は次です。

```text
no_quant probes = I × 4 timestep bins × 3 replicas
candidate probes = I × 4 timestep bins × M × 2 noise replicas × 2 quant repeats
total = I × 4 × (3 + 4M)
```

各probeにはforward、backward、activation/gradient hook、module集計が含まれます。そのため、
通常学習の単純な1 stepと完全に同じ費用ではありません。

### 3.5 Quick、Standard、Strictの候補探索

`quick`と`standard`は`2.70, 3.15, 3.45, 3.75, 4.05`を1 processで一度だけ測ります。
端点でも改善傾向が続く場合は`edge_unresolved`と表示しますが、範囲外を追跡しません。
この場合、単一代表を出さず、Fidelity retained候補を1点へ自動縮約しません。

両者の違いはLocal画像上限です。Standardは最大32画像、Quickは最大16画像です。Quickでも
4 timestep帯、no-quant 3 replicas、candidate 2 noise × 2 quant repeatsを維持するため、Body／Tailの
定義は変えません。ただしsamplingが薄いため、レポートのconfidence上限をMediumとし、
`reduced_descriptive`と明示します。独立source groupが16を超えるdatasetは、全groupを最低1件ずつ
含められないためQuickを開始前に拒否します。その場合はStandardを使用してください。

`strict`はcore grid `2.70, 3.15, 3.45`から開始します。候補集合が測定端に残る場合だけ、
最大2段まで外側を追加します。下端側は`2.25`、なお未解決なら`1.80`、上端側は`3.75`、
なお未解決なら`4.05`です。両端が残る場合は両方向を同じroundで追加します。
edge追加時は以前のmulも含む拡張grid全体を別processで再測定し、共通mulの全probe行を
exact parityで検査します。Strictの再測定は校正能力を高めますが、主要な時間増加要因です。

### 3.6 CPU解析とレポート

最後にsource groupを等重みとするbootstrapを2,000回行い、Body、Tail、95%区間、
Fidelity retained set、robust dominance、source LOOなどを作ります。`report.html`、
`technical_report.html`、JSON、CSVへ保存します。

この実測例では各CPU解析は5～7秒程度で、HTML生成を含めても全時間への影響は
小さいものでした。

### 3.7 13画像・8,400 step datasetの参考実測例

同じPC・GPUにおけるStrict実測runを基準にしています。対応する通常40 epoch学習は
約1時間23分、Strict診断は約1時間28分02秒でした。

| stage | 目的 | 実測時間 | 全体比 |
|---|---|---:|---:|
| Snapshot A | 量子化開始境界を作る1回目 | 約4分08秒 | 約5% |
| Snapshot B | 境界再現性を確認する2回目 | 約4分06秒 | 約5% |
| Prefix gate | 64A／64B／128のprefix再現性 | 約21分20秒 | 約24% |
| Core Local | 3 mul、780 probe相当 | 約15分58秒 | 約18% |
| Edge 1 | 4 mul、988 probe相当 | 約19分20秒 | 約22% |
| Edge 2 | 5 mul、1,196 probe相当 | 約22分42秒 | 約26% |
| CPU解析・parity・レポート | bootstrapと成果物生成 | 約28秒 | 1%未満 |

2回edge延長したため、Snapshot A/B、Prefix、Core、Edge 1、Edge 2の6 GPU processが
それぞれmodel準備と420-step境界作成を行いました。40 epochを連続学習してはいませんが、
512 prefix branch stepsと、合計2,964 Local probe相当を追加計測するため、通常学習と近い時間に
なりました。

### 3.8 datasetによる所要時間の違い

どのdatasetでも同じ時間になるわけではありません。主に次で変わります。

| 要因 | 時間への影響 |
|---|---|
| 通常学習相当の総step数 | 5% warmup境界が変わる。画像repeatやdataset設定が多いほど、各GPU stageの境界作成が長くなる |
| 実画像数 | Local部分は8～32画像の範囲でほぼ比例する。32画像を超える分は直接増えない |
| bucket解像度 | 高解像度bucketが多いほど各forward/backwardが重くなる |
| edge延長回数 | 0～2回。現在は拡張grid全体を再測定するため、もっとも大きな可変要因 |
| GPU、precision、backend | 同じprotocolでも1 probe当たりの時間が変わる |
| source group数 | bootstrapのCPU時間と区間の安定性に影響するが、通常はGPU時間より小さい |

この実測例と同じGPU・似たbucket構成・似たwarmup step数なら、次が目安です。

| mode | 実行内容 | 実測例を基準にした概算 |
|---|---|---:|
| Quick | Snapshot 1回、8A／8B／16、最大16画像、固定5点、edgeなし | 約60～75分（37画像・warmup 1,480 step級での事前見積もり） |
| Standard | 8A／8B／16、固定5点を1回、edgeなし | 約37分 |
| Strict（edgeなし） | 64A／64B／128、core 3点 | 約46分 |
| Strict（edge 1回） | 上記＋拡張grid再測定1回 | 約65分 |
| Strict（edge 2回） | 上記＋拡張grid再測定2回 | 約88分 |

これは保証値ではありません。実行中は`status.json`の`current_stage`と`run.log`の
`RUN`／`DONE`時刻で進行を確認してください。

### 3.9 軽量化の境界

`standard`は、最大32画像、4 timestep帯、no-quant 3 replicas、candidate 2 noise × 2 quant repeatsを
Strictと同じまま保ちます。短縮したのは長いPrefix検算とedge再測定です。そのためStandardは、
同じLocal物差しを短いQAで使う日常modeです。

`quick`は4 timestep帯とreplica数を保ったまま、Local画像上限を16へ下げ、独立snapshotを
2回から1回へ減らします。Body／Tailの定義やhard-safetyは変えませんが、sourceと画像のsamplingが
薄くなるため、Standardと同じ証拠量とは扱いません。Quickは新しいdatasetの傾向を早く見る用途、
Standardは通常の正式診断、Strictは環境・実装のreference検算に使い分けてください。

## 4. Mul affinity curveの読み方

### Body

Bodyは、通常範囲におけるcandidate勾配とno-quant勾配の変形量です。
小さいほど、そのmulの勾配がno-quantに近いことを表します。

### Tail

Tailは、最も厳しいtimestep帯における変形量です。
Bodyが小さくてもTailだけ大きい場合は、通常は穏やかでも一部条件で強く変わる
「tail-sensitive」なdatasetである可能性があります。

### 距離1.0

グラフの赤い`1.0`は画質の合格・不合格線ではありません。
使用する相対勾配距離は、勾配cosineとnorm ratioから次で計算します。

```text
d = sqrt(1 + norm_ratio^2 - 2 * norm_ratio * gradient_cosine)
```

概念的には次のように読みます。

- `d = 0`: candidateとno-quantの勾配が一致する。
- `0 < d < 1`: 差分normが基準勾配normより小さい。
- `d = 1`: 差分normが基準勾配normと同程度。
- `d > 1`: 差分normが基準勾配normより大きい。

`1`未満でも、値が小さいほど常に画質が良いとは限りません。適度な量子化摂動が
正則化として役立つ可能性があるためです。本グラフはno-quantへの数値的近さを示します。

### 点の上下にある半透明の棒

上下の棒は、独立source groupを等重みで再標本化したbootstrapの**95%区間**です。
点は保存された実測のBodyまたはTail、棒の下端と上端はbootstrap分布の
2.5%点と97.5%点です。

棒が長いときは、主に次を意味します。

- どのsourceを含めるかで値が変わりやすい。
- 独立source数が少ない。
- 特定sourceまたはtimestepがTailを押し上げている。
- 候補間の細かな順位を断定しにくい。

したがって「長いほど量子化結果が必ず悪い」という意味ではありません。
点推定が低くても棒が長い場合は、**平均的には穏やかだが結論の確信度は低い**と読みます。
複数候補の棒が重なっていても、それだけで同等とは判定せず、対応のあるbootstrapによる
勝率・Pareto dominance・source LOOも併用します。

### edge unresolved

測定gridの端点が候補集合に残った場合、真の最小点がgrid外にある可能性があります。
この場合は`edge_unresolved=true`とし、最良mulを宣言しません。

## 5. レポートの候補集合

### Hard-safety pass

NaN、Inf、極端なgradient explosion、optimizer stateの非finiteがなかった候補です。
これは最低限の安全条件であり、画質保証ではありません。

### Fidelity retained

source-cluster bootstrapで、他候補にBodyとTailの両方で高確率にPareto劣位と
判定されなかった候補集合です。現在はbetaの候補削減機能です。

### Body代表／Tail代表

- Body代表: Body点推定が最も小さい候補
- Tail代表: Tail点推定が最も小さい候補

両者が異なる場合はtrade-offです。無理に単一代表へまとめません。

### 単一代表

BodyとTailが同じ候補を支持し、候補削減規則と矛盾しない場合だけ表示します。
これは最終画質のbest mulではありません。

## 6. 実用CLI

### 6.1 唯一の通常入口: `python -m dq_profile`

診断は、学習に使うPython環境を有効にしてrepository rootから直接起動します。
`accelerate launch`では包まないでください。必要なaccelerate processはprotocol orchestratorが
各stageで起動します。

最小構成ではmodelとdataset TOMLだけが必須です。

```bat
cd /d D:\work\sd-scripts

python -m dq_profile ^
  --dq-profile-mode=standard ^
  --pretrained_model_name_or_path="D:\models\sdxl_base.safetensors" ^
  --dataset_config="D:\datasets\example\dataset.toml"
```

最初に短時間で傾向を確認したい場合は、modeだけ`quick`へ変えます。

```bat
python -m dq_profile ^
  --dq-profile-mode=quick ^
  --pretrained_model_name_or_path="D:\models\sdxl_base.safetensors" ^
  --dataset_config="D:\datasets\example\dataset.toml"
```

Quickは最大16画像の縮小samplingです。正式な通常診断は既定のStandardを使用してください。

診断名、出力先、完了後のレポート表示まで指定する推奨例です。

```bat
cd /d D:\work\sd-scripts

python -m dq_profile ^
  --dq-profile-name="example_dataset" ^
  --dq-profile-output-dir="D:\outputs\dq_diagnostics" ^
  --dq-profile-preset="canonical-v1" ^
  --dq-profile-mode=standard ^
  --dq-profile-open-report ^
  --pretrained_model_name_or_path="D:\models\sdxl_base.safetensors" ^
  --dataset_config="D:\datasets\example\dataset.toml" ^
  --output_name="example_dataset_r4"
```

現在の長い通常学習コマンドを再利用する場合は、先頭の
`accelerate launch ... sdxl_train_network.py`を`python -m dq_profile`へ置き換えます。
ただし、過去commandに`AdamW8bit`、`native_accum`、異なるdimなどが含まれると
`canonical-v1`との衝突で停止します。最小構成を使い、presetへ固定値の指定を任せる方法が
もっとも安全です。

### 6.2 診断入口のCLI一覧

| オプション | 必須 | 既定値 | 用途 |
|---|:---:|---|---|
| `--pretrained_model_name_or_path` | 必須 | なし | SDXL base modelのファイルまたはディレクトリ |
| `--dataset_config` | 必須 | なし | kohya形式dataset TOML。実効`resolution`と各subsetの`image_dir`が必要 |
| `--output_name` | 任意 | dataset TOMLのstem | 診断名のfallback。パス区切りを含まない名前 |
| `--dq-profile-name` | 任意 | `output_name` | datasetごとの親フォルダ名 |
| `--dq-profile-output-dir` | 任意 | repositoryの`..\lora_output\dq_dataset_profiler` | 診断runを格納する基底ディレクトリ |
| `--dq-profile-preset` | 任意 | `canonical-v1` | versioned互換性・計測契約。現在の対応presetは1つ |
| `--dq-profile-mode` | 任意 | `standard` | `quick`: 最大16画像の短時間傾向確認、`standard`: 日常用の正式診断、`strict`: 長いreference QA＋bounded edge再測定 |
| `--dq-profile-preflight` | 任意 | false | パス、source、CLI契約、fingerprintまで作りGPUを起動しない |
| `--dq-profile-dry-run` | 任意 | false | `execution_plan.json`と解決済みCore commandを書き、GPUを起動しない |
| `--dq-profile-open-report` | 任意 | false | Windowsで正常完了した場合に`report.html`を開く |

`resolution`はdataset sectionまたは`[general]`で指定してください（例: `resolution = 1024`）。`image_dir`はドライブ名またはUNCから始まる絶対パスで指定し、4つ以上の独立したsource groupを用意してください。`~`は学習loaderが展開しないため使用できません。子孫フォルダだけにある画像は通常のDreamBooth学習loaderから見えないため診断でも数えません。子フォルダを個別subsetとしてTOMLへ列挙するか、画像を`image_dir`直下へ配置してください。`num_repeats`は1以上が必要です。画像inventoryがworkerごとに変わり得る`cache_info=true`は現在のdiagnostic contractでは拒否します。

事前検査だけ行う例です。検査結果も新しいrunディレクトリへ保存します。

```bat
python -m dq_profile ^
  --dq-profile-preflight ^
  --pretrained_model_name_or_path="D:\models\sdxl_base.safetensors" ^
  --dataset_config="D:\datasets\example\dataset.toml"
```

command planまで確認したい場合は`--dq-profile-dry-run`を使用します。

```bat
python -m dq_profile ^
  --dq-profile-dry-run ^
  --pretrained_model_name_or_path="D:\models\sdxl_base.safetensors" ^
  --dataset_config="D:\datasets\example\dataset.toml"
```

`sdxl_dq_dataset_profile.py`と、その`--dq_profile_*`オプションは内部stage・研究用です。
通常利用で直接呼ぶと、Snapshot A/B、prefix gate、edge extension、成果物の昇格を手動管理する
必要があるため、公開CLIとして使用しません。

### 6.3 `canonical-v1`が固定する学習設定

次の値は、省略すればpresetが自動挿入します。同じ値を明示した場合は
`matched_preset`、異なる値を明示した場合はGPU開始前に`rejected`となります。
TOMLの`[general]`またはdataset sectionで`batch_size`、`enable_bucket`、`bucket_no_upscale`、bucket範囲を
上書きした場合も、fallback解決後の実効値をこの表と比較します。

| 分類 | 学習オプション | 固定値 |
|---|---|---|
| 基本 | `prior_loss_weight` | `1.0` |
| 基本 | `max_train_epochs` | `40`。全epochを診断学習するためではなく、通常step数とwarmup境界の算出にも使う |
| 基本 | `seed` | `39` |
| optimizer | `optimizer_type` | `AdamW8bitFast` |
| optimizer | `learning_rate` | `3.5e-4` |
| precision | `mixed_precision` | `fp16` |
| precision | `fp16_safe_norms_mode` | `strict`。`--fp16_safe_norms`もstrict aliasとして許可 |
| attention | `sdpa` | enabled |
| batch | `train_batch_size` | `1` |
| batch | `gradient_accumulation_steps` | `1` |
| DataLoader | `max_data_loader_n_workers` | `0`へ強制 |
| LoRA | `network_module` | 入力契約は`networks.lora`、実行時は隔離した`dq_profile.copied_lora`へ差し替え |
| LoRA | `network_dim` | `4` |
| LoRA | `network_args` | `rank_dropout=0.2`だけを許可 |
| LoRA | `network_dropout` | `0.3` |
| bucket | `enable_bucket` | enabled |
| bucket | `bucket_no_upscale` | disabled。画像由来bucketへ切り替わるため`true`は拒否 |
| bucket | `min_bucket_reso` | `384` |
| bucket | `max_bucket_reso` | `1024` |
| bucket | `bucket_reso_steps` | `64` |
| noise | `noise_offset` | `0.15` |
| noise | `adaptive_noise_scale` | `0.1` |
| latent | `cache_latents` | enabled |
| latent互換 | dataset `color_aug` | fallback後にdisabledであることを要求 |
| latent互換 | dataset `random_crop` | fallback後にdisabledであることを要求 |
| Text Encoder | `text_encoder_lr` | `2e-4` |
| Text Encoder | `text_encoder_lr1` | `3e-4` |
| Text Encoder | `text_encoder_lr2` | `2e-4` |
| SDXL | `downscale_freq_shift` | enabled |
| SDXL | `te_mlp_fc_only` | enabled |
| Guardian | `grad_norm_mode` | `stable_no_threshoff` |
| averaging | `avg_cp` | enabled |
| averaging | `avg_cp_mode` | `promote` |
| averaging | `avg_window` | `4` |
| averaging | `avg_begin` | `0.6` |
| averaging | `avg_mode` | `ema` |
| averaging | `avg_shadow_bank_size` | `12` |
| averaging | `avg_reset_stats` | false (`--no-avg_reset_stats`) |
| averaging | `avg_save_final_raw` | enabled |
| scheduler | `lr_scheduler` | `constant_with_warmup` |
| scheduler | `lr_warmup_steps` | `0.05` |
| rank log | `rank_log` | enabled |
| rank log | `rank_log_mode` | `per_module` |
| DQ | `dq_delta_bits` | `8` |
| DQ | `dq_delta_granularity` | `channel` |
| DQ | `dq_delta_stat` | `rms` |
| DQ | `dq_delta_mode` | `stoch` |
| DQ | `dq_delta_begin_after_lr_warmup` | enabled |
| DQ | `dq_delta_scope` | `unet` |
| DQ | `dq_delta_log` | enabled |
| DQ | `dq_delta_log_detail` | `basic` |
| DQ backend | `dq_delta_use_triton` | enabled |
| DQ backend | `dq_delta_triton_stats` | enabled |

`--fp16_safe_norms_mode=native_accum`、別dim、別optimizerなどを調べること自体は可能ですが、
現在の実測と同じ物差しではなくなります。既存presetの値を暗黙に変えず、別のversioned presetを
追加し、snapshot／prefix／Local parityを検証してから使用します。

### 6.4 Local測定契約

| 項目 | 固定値・動作 |
|---|---|
| 製品scope | Local Body／Tail Safety/Fidelity。最終画質Utilityではない |
| probe画像数 | Standard／Strictは`min(dataset実画像数, 32)`、Quickは`min(dataset実画像数, 16)`。最低8画像 |
| timestep bins | `4` |
| no-quant replicas | noise 3回 |
| candidate replicas | noise 2回 × stochastic quant 2回 |
| Local dropout | off (`structural_dropout_off`) |
| Prefix dropout | on。通常学習に近いprefix再現性検査 |
| update branch | 製品Localでは0。128-step Trajectoryは研究専用で実行しない |
| Guardian ablation | `common_only` |
| CountSketch | 幅512、独立seed 2個 |
| Accelerate process | `num_processes=1`、`num_machines=1`を各stageへ明示 |
| CPU threads/process | `8` |
| bootstrap | source単位、2,000回、固定seed |

QuickでもBody／Tailの数式、4 timestep帯、replica数、bootstrap、hard-safetyは変えません。
`sampling_depth=reduced_16_image`、`confidence_ceiling=reduced_descriptive`を成果物へ保存し、
証拠量が少ないことをStandard／Strictと区別します。

### 6.5 execution modeごとのQA・候補探索契約

| 項目 | `quick` | `standard` | `strict` |
|---|---|---|---|
| 用途 | 新規datasetの短時間傾向確認 | 日常の正式dataset診断 | コード／CUDA／PyTorch／bitsandbytes変更後、リリース前、再現性調査 |
| 独立snapshot | 1回。Prefix processとの境界一致を検査 | A/Bの2回 | A/Bの2回 |
| Prefix | 8A／8B／16@8 | 8A／8B／16@8 | 64A／64B／128@64 |
| state checkpoints | `0, 1, 4, 8` | `0, 1, 4, 8` | `0, 1, 32, 64` |
| Prefix branch updates | 64 | 64 | 512 |
| Local画像上限 | 16 | 32 | 32 |
| 最初のgrid | `2.70, 3.15, 3.45, 3.75, 4.05` | `2.70, 3.15, 3.45, 3.75, 4.05` | `2.70, 3.15, 3.45` |
| edge extension | なし。端点傾向は未解決として表示 | なし。端点傾向は未解決として表示 | 最大2 round、拡張gridを再測定してparity検査 |
| GPU process数 | 3 | 4 | 4～6 |
| QA表示 | `Quick smoke` | `Standard smoke` | `Strict reference` |
| confidence上限 | Medium | 通常 | 通常 |

Quickでは全source groupをprobeへ最低1件ずつ含める契約を維持します。独立groupが16を超える場合は、
部分的な結果を黙って出さずpreflightで拒否します。Standardへ切り替えてください。

`--dq_profile_level=standard`は低レベルprotocol内部の別概念です。公開CLIの
`--dq-profile-mode=standard`と混同しないよう、成果物には`execution_mode`、`qa_depth`、
`internal_profile_level`を別フィールドで保存します。

### 6.6 明示しても診断値へ置き換えるオプション

次は過去の長い学習commandを受け取りやすくするためエラーにしませんが、診断にはそのまま
使用しません。`resolved_args.json`へ`overridden_with_reason`として値と理由を保存します。

| オプション群 | 診断時の扱い |
|---|---|
| `output_dir` | 通常checkpoint出力には書かず、診断run directoryだけを使う |
| `save_precision`, `save_model_as` | 通常checkpointを保存しないため不使用 |
| `save_every_n_epochs`, `save_every_n_steps` | epoch／step checkpointを保存しないため不使用 |
| `training_comment` | versioned diagnostic provenance commentへ置換 |
| `max_data_loader_n_workers` | deterministic replayのため0へ強制 |
| `dq_delta_range_mul` | fixed diagnostic mul gridへ置換 |
| `dq_delta_auto_range_mul`と全`dq_delta_auto_*` | fixed scanではauto rangeを無効化 |
| `dq_delta_log_every`, `dq_delta_log_scope`, `dq_delta_log_mode` | protocolが記録頻度・範囲を管理 |
| `dq_delta_log_error_parts` | Local protocol独自の誤差分解を使用 |

### 6.7 拒否するオプション

| オプション | 拒否理由 |
|---|---|
| `resume`, `resume_from_huggingface` | 全候補をfresh common snapshotから開始できなくなる |
| `network_weights` | 既存LoRA重みがfresh-snapshot比較を壊す |
| `max_train_steps` | `canonical-v1`は40 epoch相当からwarmup境界を算出する |
| `config_file` | config展開は未対応。必要なtraining optionを直接渡す |
| `full_fp16` | canonical fp16契約外 |
| `fp8_base` | 未検証 |
| `dq_delta_bits_sched` | 固定8-bit契約外 |
| `dq_delta_step` | step-based quantizationは契約外 |
| `dq_quantize_z` | z量子化は契約外 |
| `optimizer_args` | custom optimizer設定は未検証 |
| `network_alpha` | custom alphaは未検証 |
| 未知のオプション | typoや値欠落を黙って無視しない |
| parserが認識しても上記の許可表にないオプション | presetで検証されていないため拒否 |

診断入口は各明示指定を次の4種類に分類し、`resolved_args.json`へ保存します。

- `consumed`: model、dataset、output名など診断要求に使用する。
- `matched_preset`: `canonical-v1`と一致するため許可する。
- `overridden_with_reason`: 理由を記録して診断値へ置換する。
- `rejected`: GPU開始前にエラーにする。

例えば`--optimizer_type=AdamW8bit`を明示すると、要求される`AdamW8bitFast`との衝突として
停止します。`--fp16_safe_norms_mode=native_accum`も、現在は同じく停止します。

## 7. 推奨する初回マージ範囲

安定機能として含める範囲:

- isolated profiler entry
- stateless quant RNGとsealed replay
- snapshot、manifest、status、hard-safety
- Local Body／Tail／Tail Amplification
- Mul affinity curveと単一dataset HTML
- 匿名化した比較例
- unit、schema、isolation、golden report test

experimentalと明記する範囲:

- Fidelity retainedによるLocal候補削減
- 単一代表

研究専用として通常レポートから外す範囲:

- 128-step Trajectory
- 画質Utilityとの対応

初回マージでは最終画質の自動推薦を行いません。

## 8. 出力の扱い

既定では次の構造でGit管理外へ保存します。

```text
<project-root>\lora_output\dq_dataset_profiler\
  <profile_name>\
    <YYYYMMDD_HHMMSS>_<protocol fingerprint>\
      report.html
      technical_report.html
      summary.json
      status.json
      ...
```

各実行は新しいrunディレクトリを排他的に作成し、既存runを上書き・再利用しません。
`.git`、venv、dataset画像ディレクトリ、通常checkpoint出力そのものは出力先として拒否します。
最低限、次を保管します。

- `report.html`: 通常利用向けの自己完結Local-onlyレポート
- `technical_report.html`: 解析詳細を残す技術レポート
- `practical_report.json`と`report_contract.json`: 表示モデルと意味契約
- `summary.json`
- `resolved_args.json`
- `protocol_fingerprint.json`
- `execution_plan.json`: GPU process数、warmup、Prefix、Local probe数、非保証の参考時間
- `dataset_config_snapshot.toml`
- `source_manifest.json`と`candidate_definitions.json`
- `status.json`
- 候補・timestep・bootstrapのCSV
- 実行ログ

### 第三者へ共有する前のプライバシーチェック

生のrun directoryは、そのまま公開しないでください。少なくとも次を削除または一般化します。

| 情報 | 含まれ得る成果物 | 公開用の扱い |
|---|---|---|
| dataset名、作品名、人物名、固有タグ | profile名、report、TOML、CSV | `Dataset A`など意味を持たないIDへ置換 |
| model／dataset／outputの絶対パス | `resolved_args.json`、manifest、log、TOML | `<model-path>`、`<dataset-path>`などへ置換 |
| caption、画像ファイル名、画像キー | TOML、probe manifest、raw CSV、log | 削除または連番IDへ置換 |
| ファイルhash、source inventory | source manifest、source map | 公開例から削除。hashもdatasetを照合できる識別情報として扱う |
| host名、ユーザー名、環境固有情報 | manifest、log | 比較に不要なら削除 |
| API key、token、password | 本来CLIへ渡さない | 文字列を伏せ、漏えい時は無効化・再発行する |

リポジトリへ収録する例は、生のrun directoryではなく、公開項目を限定した匿名化成果物を
新しく作ります。本リポジトリの実測比較例は、名称、パス、caption、画像識別子、hashを
含まない形にしてあります。

## 9. 実測比較例

[dataset差の実測例](examples/dq_dataset_profiler_anonymized_example.html)には、
保存済みv2.4実測の点推定と95%区間を丸め、dataset名、作品名、人物名、パス、caption、
画像識別子、source hashを除いた抜粋を収録しています。

例では次のような違いを確認できます。

- 全mulでBody/Tailが小さいdataset
- 低mulのTailだけが大きいdataset
- 試した全候補で変形が比較的大きいdataset
- mul増加に沿って穏やかになるdataset
- 同じ画像でもタグ設計だけで曲線が変わるpaired dataset

これらはdataset固有の数値的反応が観測できることを示しますが、画質の優劣を示すものではありません。

## 10. Trajectory検証の結果と残作業

128-step Trajectory検証では、Localで棄却されたcontrolがTrajectoryでは最小となり、
Local Tailとの順位相関が`-0.5`になる反転を確認しました。共有probe parityは`pass_exact`、
5 repeatのleave-one-outも安定していたため、単なる実行不良ではなく両channelが別の性質を
測っていると判断します。

このため製品レポートはLocal-onlyを維持し、Trajectoryは研究用の説明channelに限定します。
詳細は[Trajectory channel 検証判断](dq_dataset_profiler_trajectory_decision-ja.md)を参照してください。

現在残っている主な検証:

- 独立profile seedで、Local候補集合とedge傾向のrun-level再現性を確認する。
- 将来、固定済みblind評価によるUtility Bridgeで数値指標と最終画質を結ぶ。

Utility Bridgeを終える前でも、Local-onlyのSafety/Fidelity説明診断としては利用・マージ可能です。
