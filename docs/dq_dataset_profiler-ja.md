# SDXL DQ Dataset Profiler 利用ガイド

画像・フォルダ・キャラタグ別の追加診断、warmup前後比較、52画像化については、[データセット診断ガイド](dq_dataset_diagnostics-ja.md)を参照してください。

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
- 有効な`image_dir` groupの全inventoryをsource contractへ保存する。group数がprobe上限を超える場合は、TOMLの全source順序を均等に覆う決定的な部分集合をprobe対象とし、probe数／全group数とcoverageをレポートへ明示する。
- `cache_latents`と両立しない`color_aug=true`または`random_crop=true`が、subset／dataset／`[general]`のfallback後に有効でないことを確認する。
- DreamBooth loaderに必須の`resolution`が、datasetまたは`[general]`のfallback後に定義されていることを確認する。
- TOMLの`[general]`／dataset／subset fallbackを解決し、batch・bucket設定（`bucket_no_upscale=false`を含む）が`canonical-v1`と一致することを確認する。
- CLIが`canonical-v1`と互換である。
- 通常checkpoint、dataset、repositoryと診断出力先が重ならない。
- Git HEAD、ソースhash、preset、model内容のSHA-256、dataset、source inventoryからprotocol fingerprintを作る。
- 各GPU workerの起動直前にmodel内容とsource inventoryを再度hash照合し、長い多段runの途中でmodel、画像、caption、cache sidecarが変化した場合は混在させず停止する。
- repositoryに追跡済みの未コミット変更がある場合は、HEADとの差分全体もbinary diffとしてhash化する。未追跡ファイルは対象外と明記する。
- 実画像数とmode別probe budgetに加え、全source group数、probe対象group数、決定的な選択規則を記録する。

`--dq-profile-preflight`ではここまで実行し、GPU stageを起動しません。
通常のpreflight／dry-runを含む全runで`execution_plan.json`も作り、GPU process数、
warmup境界、Prefix update数、Local probe数、固定grid、参考時間の算出条件を記録します。

### 3.2 量子化開始境界とsnapshot検算

通常学習コードと同じ規則で`dq_delta_begin_step`を求め、量子化開始直前までno-quantで
warmupします。`canonical-v1`では40 epoch相当の総stepと5% LR warmupから境界が決まります。

`strict`は、同じ初期状態からSnapshot AとSnapshot Bを別processで作り、LoRA重み、optimizer、
scheduler、GradScaler、Guardian、RNG、replay位置などのfingerprintを比較します。`standard`は
Snapshot Aを1回だけ作り、後続のPrefix processが同じ境界を再現できたかを比較します。
どちらも境界fingerprintが一致しなければmul比較を開始しません。Standardは専用のSnapshot Bを
省くぶん速い一方、同じsnapshot-only stageを2回作る検算深度はStrictより低くなります。

以下で時間例に使う小規模datasetは、通常学習が8,400 stepだったため、境界は
`8,400 × 0.05 = 420 step`でした。

### 3.3 Prefix parity gate

同じsnapshotから、no-quantとanchor候補`mul=3.15`について次を実行します。

| execution mode | short A | short B | long | 比較checkpoint | 合計branch update |
|---|---:|---:|---:|---|---:|
| `standard` | 8 | 8 | 16 | `0, 1, 4, 8` | 64 |
| `strict` | 64 | 64 | 128 | `0, 1, 32, 64` | 512 |

各modeでA対B、およびA対long runの同じ先頭prefixを比較します。sample、noise、timestep、
rank／network dropout、量子化乱数、Loss、LR、skip、勾配、LoRA重み、optimizer、scheduler、
GradScaler、Guardian、replay cursorを検査します。このstageは通常学習に近い経路を検査するため
dropout有効です。`standard`は短いsmoke QA、`strict`は環境変更・リリース前・
再現性調査用のreference QAです。Standardは独立snapshot検算を1回減らします。同じ`PASS`でも深度が異なるため、`execution_mode`と
`qa_depth`をJSONとレポートへ別々に記録します。prefix gateまたはsource contractが失敗した場合、
Local計測へ進みません。

### 3.4 Local Body／Tail scan

ここが製品レポートの診断本体です。optimizer更新を行わず、同じ画像、noise、timestepで
no-quantと各固定mulの勾配を比較します。

- 画像数: Standard／Strictとも8～32。上限を超えてもprobe budgetは増えない。
- timestep: 4帯。
- no-quant: 3 noise replicas。
- 各mul: 2 noise replicas × 2 stochastic quant repeats。
- stateless量子化乱数を使い、共通候補間でcommon random numbersを保つ。
- dropoutを無効にした`structural_dropout_off` regimeで測る。
- module単位の勾配を集約し、Body、Tail、hard-safety、source別の不確実性を作る。

比較用branchの先頭replay windowは固定したままです。この固定windowにprobe対象の
`image_dir` groupが含まれなかった場合だけ、DataLoaderを最大2 epoch分追加走査します。
不足groupを初めて含んだbatchだけをprobe用に保持し、対象groupを揃えてから画像を
round-robin選択します。group数が画像上限を超える場合、source inventory自体は省略せず、
TOMLの先頭だけへ偏らない決定的な等間隔選択で対象groupを絞ります。追加batchはbranchの
prefixへ混ぜないため、候補間比較の再現契約は変わりません。極端にrepeatが偏るdatasetでは、
このcoverage走査ぶんだけLocal計測開始前の時間が増える場合があります。

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

### 3.5 StandardとStrictの候補探索

`standard`は`2.70, 3.15, 3.45, 3.75, 4.05`を1 processで一度だけ測ります。
端点でも改善傾向が続く場合は`edge_unresolved`と表示しますが、範囲外を追跡しません。
この場合、単一代表を出さず、Fidelity retained候補を1点へ自動縮約しません。

Standard／Strictとも最大52画像、4 timestep帯、no-quant 3 replicas、candidate 2 noise ×
2 quant repeatsを使うため、Local Body／Tailの物差しは共通です。独立source groupが画像上限を
超えるdatasetも実行できますが、最大52群だけを決定的にprobeします。全groupはsource contractに
残り、レポートには`probe / total`を表示します。未probe群がある結果は完全coverageと同一視しません。

`strict`はcore grid `2.70, 3.15, 3.45`から開始します。候補集合が測定端に残る場合だけ、
最大2段まで外側を追加します。下端側は`2.25`、なお未解決なら`1.80`、上端側は`3.75`、
なお未解決なら`4.05`です。両端が残る場合は両方向を同じroundで追加します。
edge追加時は以前のmulも含む拡張grid全体を別processで再測定し、共通mulの全probe行を
exact parityで検査します。Strictの再測定は校正能力を高めますが、主要な時間増加要因です。

### 3.6 CPU解析とレポート

最後にsource groupを等重みとするbootstrapを2,000回行い、Body、Tail、95%区間、
Fidelity retained set、robust dominance、source LOOなどを作ります。`report.html`、
`beginner_report.html`、`technical_report.html`、JSON、CSVへ保存します。

`beginner_report.html`は最上部のMul affinity curveから読み始められる概要版です。
Body／Tail／ヒゲ、候補の役割、Body × Tailマップ、5軸の性格カルテ、
source／timestep偏りを短い説明付きで表示します。性格カルテの参照位置は、
匿名化した固定Standard参照設定内での相対位置であり、良否の閾値や画質推薦には使いません。

同じGPU測定済みCSVから、候補選択へ加点しない説明専用channelも作ります。

- **Source localization**: candidateごとにsource等重みのq85／q90／q95を基準とし、
  Tailの超過負担がどのsourceへ集中するか、上位source比率、実効source数、thresholdを
  変えたときの安定性を記録します。最大負担sourceと、source LOOでTailが最も下がるsourceは
  別々に表示します。高い集中率でも絶対Tailが小さい場合は、それだけで警告にしません。
- **No-quant baseline profile**: candidate／quant repeat間で重複保存された同一no-quant参照を
  probe単位にまとめ、勾配normのq05／median／q95／RMS、source別energy比、実効source数、
  timestep別信号規模を記録します。収束、最終画質、rank、LR、epoch数は予測しません。
- **Dataset character vector**: 絶対的な受容帯、mul応答、Tail増幅、source集中、no-quant信号を
  独立した5 channelとして並べます。多数決や平均による単一スコアには変換しません。
- **Image coverage**: probe画像数／dataset実画像数を表示します。52画像を超えるdatasetでは
  未probe画像が残るため、説明値をdataset全体の完全観測とは扱いません。

これらは`selector_input=false`、`not_quality_or_utility=true`として保存します。
Fidelity retained set、Hard Safety、代表候補の決定規則は変えません。同じmodel、network、
optimizer、precision契約のrun同士で比較するときのdataset体質記述に使用します。

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
| 実画像数 | Local部分はprobe上限までほぼ比例する。Standard／Strictとも最大52画像 |
| bucket解像度 | 高解像度bucketが多いほど各forward/backwardが重くなる |
| edge延長回数 | 0～2回。現在は拡張grid全体を再測定するため、もっとも大きな可変要因 |
| GPU、precision、backend | 同じprotocolでも1 probe当たりの時間が変わる |
| source group数 | bootstrapのCPU時間と区間の安定性に影響するが、通常はGPU時間より小さい |

preflight後に作られる`execution_plan.json`の`reference_time_estimate.minutes`には、
そのrunの画像数、warmup境界、mode、候補数を反映した参考時間を保存します。
`minimum`が通常経路、`maximum_if_all_edge_rounds_run`がStrictで全edge延長を使った場合の
上限側の目安です。これは単一環境の実測を基にした保証のない概算であり、bucket構成やGPU環境で
変わります。実行中は`status.json`の`current_stage`と`run.log`の`RUN`／`DONE`時刻で
実時間と進行を確認してください。

対話consoleでは、warmupを含む`tqdm`の進捗を通常学習と同じ1行上で更新します。
`run.log`は後から検索しやすいよう、各進捗更新を独立した行として保存します。

### 3.9 軽量化の境界

`standard`は、最大52画像、4 timestep帯、no-quant 3 replicas、candidate 2 noise × 2 quant repeatsを
Strictと同じまま保ちます。短縮するのは専用Snapshot B、長いPrefix検算、edge再測定です。
Snapshot Aと後続Prefix processの境界parityは維持するため、同じLocal物差しを短いQAで使う
日常modeです。

`strict`は独立Snapshot A/B、長いPrefix、bounded edge再測定を使うreference modeです。コード、
CUDA、PyTorch、bitsandbytesの変更後、リリース前、またはStandardの結果が疑わしい場合に使います。

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

### 表の記号

「試したmulと役割」の記号は、すべて同じ意味の合格票ではありません。

- 緑の`✓`: Hard-safetyを通過した候補
- 青の`✓`: Fidelity retained setに残った候補
- 橙の`注意`: Hard-safetyは通過したが、同じdataset内の他候補よりBody・Tailの摂動が強い候補
- 紫の`★`: Body代表、Tail代表、または単一代表
- `●`: その挙動分類に該当
- 灰色の`—`: 非該当

特に橙の`注意`は「学習結果や画質が悪い」という判定ではありません。no-quantからの勾配変形が
候補内で相対的に強いため、穏やかな候補とは別枠で比較するとよい、という注意表示です。
`✓`、`注意`、`★`はそれぞれ安全性・相対的な摂動・数値上の代表という別の役割を示します。

### Hard-safety pass

NaN、Inf、極端なgradient explosion、optimizer stateの非finiteがなかった候補です。
これは最低限の安全条件であり、画質保証ではありません。
1件でも非finiteなgradient probeが出たmulは、その候補全体を数値比較から外して
`Hard unsafe`として残します。他の有限なmulはbootstrapとHTML生成を継続するため、
1候補の異常だけで診断全体を失敗させません。原因と非finite件数はsummary／候補カードへ保存します。

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

通常は既定の`standard`を使用します。コード、CUDA、PyTorch、bitsandbytesの変更後や、
Standardの結果が疑わしい場合だけ`--dq-profile-mode=strict`へ切り替えます。

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
| `--dq-profile-mode` | 任意 | `standard` | `standard`: 最大52画像・snapshot 1回の日常診断、`strict`: 独立snapshot A/B・長いreference QA＋bounded edge再測定 |
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
| probe画像数 | Standard／Strictとも`min(dataset実画像数, 32)`。最低8画像 |
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

### 6.5 execution modeごとのQA・候補探索契約

| 項目 | `standard` | `strict` |
|---|---|---|
| 用途 | 日常の正式dataset診断 | コード／CUDA／PyTorch／bitsandbytes変更後、リリース前、再現性調査 |
| 独立snapshot | 1回。Prefix processとの境界一致を検査 | A/Bの2回 |
| Prefix | 8A／8B／16@8 | 64A／64B／128@64 |
| state checkpoints | `0, 1, 4, 8` | `0, 1, 32, 64` |
| Prefix branch updates | 64 | 512 |
| Local画像上限 | 32 | 32 |
| 最初のgrid | `2.70, 3.15, 3.45, 3.75, 4.05` | `2.70, 3.15, 3.45` |
| edge extension | なし。端点傾向は未解決として表示 | 最大2 round、拡張gridを再測定してparity検査 |
| GPU process数 | 3 | 4～6 |
| QA表示 | `Standard smoke` | `Strict reference` |
| confidence上限 | 通常 | 通常 |

source groupがmodeの画像上限を超える場合もpreflightでは拒否しません。全inventoryをsource contractへ
保持したまま、Standard／Strictは最大52群をTOML全域から決定的に選びます。
`report.html`とsummaryには`source_group_count_probed`／`source_group_count_total`／coverage規則を
残し、未probe群がある場合はLocal confidenceを過大評価しません。

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

## 7. 出力の扱い

既定では次の構造でGit管理外へ保存します。

```text
<project-root>\lora_output\dq_dataset_profiler\
  <profile_name>\
    <YYYYMMDD_HHMMSS>_<protocol fingerprint>\
      report.html
      beginner_report.html
      technical_report.html
      summary.json
      status.json
      ...
```

各実行は新しいrunディレクトリを排他的に作成し、既存runを上書き・再利用しません。
`.git`、venv、dataset画像ディレクトリ、通常checkpoint出力そのものは出力先として拒否します。
最低限、次を保管します。

- `report.html`: 通常利用向けの自己完結Local-onlyレポート
- `beginner_report.html`: 結論から段階的に読める自己完結の概要レポート
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
- `source_localization.json/.csv`と`source_localization_detail.csv`: Tail負担のsource集中
- `no_quant_baseline_profile.json`、`no_quant_source_load.csv`、
  `no_quant_timestep_profile.csv`: no-quant短期勾配の規模と偏り
- `dataset_character_vector.json`: 合成点を作らないdataset体質の5 channel
- 実行ログ

## 8. 実測比較例

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
