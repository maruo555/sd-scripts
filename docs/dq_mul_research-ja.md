# 固定mul配分の研究用診断

`sdxl_dq_mul_research.py` は、事前に承認したローカルの実験契約を読み、
共通のwarmup境界で量子化対象と固定mul配分を比較する研究用入口です。
この研究用入口から、`python -m dq_profile` のpresetは変更しません。
標準診断側は下記の `canonical-v2`、旧条件の再現は `canonical-v1` で区別します。
この入口は短期学習・本学習・画像生成を実行しません。

## 研究結果を通常診断へ取り入れる仕様

2026-10-10に、旧称「元A／候補A」を**attn2・TE高mul型［固定基準］**と命名しました。
配分は、その他UNet=2.70、attn2=3.75、TE1=TE2=3.75です。
以下の最初のPolicy例はこの固定基準に対応します。

複数データセットの完走学習と画像評価で、一律mulとは違う質感やポーズの自由度を持つ例が
見られたため、通常診断へ比較候補として取り入れます。診断の低スコアが良い画像を保証した
わけではなく、TE単独・attn2単独の寄与も確定していません。
一律診断から得たBody代表をattn2・TEの高側mulにし、その他UNetを2.70／3.15とした
2配分を追加し、固定基準も参照用に残す設計です。同一配分は重複計測しません。

[標準診断の採用仕様](dq_dataset_profiler-ja.md#spatial-diagnostic-spec)に、
実験の背景、名称、TE込み固定、dropout OFFを基本とするON追加確認、部位別指標、
レポート構成を記載しています。自動追加と統合レポートは `canonical-v2` に実装しました。
新標準全体のSDXL GPU受入は未実施で、過去の研究実験とは区別しています。
旧条件IDや実験記録・学習済みファイル名は改名しません。

## 実装範囲

- `library/dq_mul_policy.py`: 軽量な固定policy resolver。
- `dq_profile/mul_research.py`: 既存Local passを用いる対応付き比較。
- `dq_profile/research_budget.py`: 承認・入力・コード・累積資源の検査。
- `sdxl_dq_mul_research.py`: 単一GPUジョブの直列起動と結果の受け渡し。

通常trainerは研究runtimeをimportしません。通常学習でscope指定後の
共通setterがTEも有効化する従来の挙動を維持します。研究側のTE OFFは、
TE学習を止めずに量子化だけを外す明示的な介入です。

## Policy

```json
{
  "base_mul": 2.70,
  "components": {"unet": 2.70, "te1": 3.75, "te2": 3.75},
  "te_quantized": true,
  "group_overrides": [
    {"group_id": "unet.attn2", "range_mul": 3.75}
  ]
}
```

優先順位は `module_overrides > 役割＋部位 > 役割 > 大分類 > components > base_mul` です。
UNet群は `unet.attn1`、`unet.attn2`、`unet.ff`、
`unet.other_projection` の排他的な4大分類です。attentionには
`.q`、`.k`、`.v`、`.out` を追加でき、さらに `.input`、`.middle`、`.output`
を追加できます。例: `unet.attn2.q.output`。
`module_overrides` は実際のLoRAモジュール名からmulへの辞書です。
同じ具体性の重複は拒否し、JSONの記載順では優先順位を変えません。
未知の成分・群・モジュール、
重複指定、対象のないoverride、非有限または0以下のmulは拒否します。
OFFをmul=0で表現しません。

実際のモジュール名へ展開した設定を保存し、passの共通setterの後で適用します。
同じ測定cell内で展開済み割当が一致する別名候補は、量子化repeatごとに
一度だけF/Bを実行します。state・画像・noise・timestep・dropoutが異なるcellへは
このcacheを持ち越しません。別名と再利用元を記録し、F/B予算には実行数を使います。
snapshotはモジュールごとのON/OFFとmulも復元します。この研究runtimeでは
学習resumeを行いません。

## 通常学習への明示的な適用

2026-10-10から、通常の`sdxl_train_network.py`でmulをCLIへ直接指定できます。
診断の実行や診断結果ファイルの読み込みは必要ありません。基本値は従来の
`--dq_delta_range_mul`とし、指定した部位だけ上書きします。

| 指定 | 適用先・省略時の扱い |
|---|---|
| `--dq_delta_range_mul` | 基本値。部位別の指定がなければ全体に適用（既定3.0） |
| `--dq_delta_range_mul_attn2` | 全Down/Mid/Upのattn2 Q/K/V/Out。省略時は基本値 |
| `--dq_delta_range_mul_te` | TE1・TE2共通。省略時は基本値 |
| `--dq_delta_range_mul_te1` | TE1個別。TE共通値より優先 |
| `--dq_delta_range_mul_te2` | TE2個別。TE共通値より優先 |

attn2・TE高mul型［固定基準］にする場合、既存の学習コマンドの量子化設定を次のように
指定します。その他のoptimizer・dropout・warmup・平均化の引数はそのままです。

```text
--dq_delta_bits 8 --dq_delta_stat rms --dq_delta_scope both --dq_delta_range_mul 2.70 --dq_delta_range_mul_attn2 3.75 --dq_delta_range_mul_te 3.75
```

Body基準ならattn2とTE共通値の`3.75`を診断に表示された具体的なBody代表mulへ置き換え、
基本値を`2.70`または`3.15`にします。学習コマンドが診断結果から自動選択することはありません。
TE1・TE2で分ける場合は、例えば`--dq_delta_range_mul_te 3.75 --dq_delta_range_mul_te2 3.45`
ならTE1=3.75、TE2=3.45になります。

部位別の直接指定はUNetとTEを量子化対象とし、従来のscope指定に優先します。
**`--dq_delta_scope unet`のままでも`--dq_delta_range_mul_te`は無視されず、
量子化開始後のTE1・TE2へ適用されます。** 個別TE指定がある場合はそちらが優先します。
意図を明示するため`--dq_delta_scope both`を推奨します。固定bitsかつRMSのdelta量子化を
必須とし、自動mul調整・bits schedule・z量子化との併用は拒否します。
各mulは有限の正数とし、0で量子化OFFを表現しません。存在しないTEやattn2への指定は
実ネットワークへの解決時にエラーにします。部位別指定をすべて省略した通常学習の挙動は変更しません。

直接指定も既存の固定policy resolverへ変換するため、warmup中の量子化OFF、再設定後の配分保持、
保存メタデータとresumeの一致検査は共通です。再開時も同じ宣言と配分の指定を渡してください。
設定を省略・変更して途中から別配分へ切り替える再開は拒否します。
展開後のmulが同じでも、宣言が異なるJSONからCLIへの途中切替は拒否します。
例えばJSONに`components.unet=base_mul`という冗長な指定がある場合、CLIの省略指定と
実際のmulは一致しても宣言は一致しません。その学習は元のJSON指定で再開し、CLIへの
移行は新規学習で行ってください。診断結果をCLIで使う通常の新規学習には影響しません。

### 研究用JSONとの互換性

既存の`--dq_delta_policy_file=policy.json`は過去の研究設定の再利用用に残します。
通常の直接指定にJSONは不要で、部位別の直接指定とこのファイル指定は併用できません。
JSON側は量子化対象もpolicyが決め、従来のscope指定に優先します。
以下はattn2全体ではなく、Upのattn2 Qだけを変える研究用の例です。

```json
{
  "base_mul": 2.70,
  "components": {"te1": 3.75, "te2": 3.75},
  "group_overrides": [
    {"group_id": "unet.attn2.q.output", "range_mul": 3.75}
  ]
}
```

配分は実ネットワークに対して一度解決し、存在しない指定があれば学習前に
停止します。warmupのOFF/ONと共通setterによる再設定の後も配分を維持します。
TE OFFは量子化だけを無効にし、TEのtrainable状態を変えません。
checkpointにはpolicy宣言、解決済み配分のSHA-256、policy版を保存します。
これらは `--no_metadata` でも残します。policyファイルのパスはこのメタデータに
含めません。avg promoteで重みを読み戻しても配分は維持されますが、既存の
promoteモードのresume制約は変更しません。

通常trainerには埋め込み実行用の任意observer hookがあります。CLIはobserverを
設定しません。研究runnerはこのhookで学習batch、shadowのforward、backward後、
更新後を記録・監視できます。通常trainerは研究用budgetやruntimeをimportしません。

## ローカル実験契約

実パス・dataset名・入力hash・試験コード・出力は、Git管理外の作業フォルダに
置きます。承認記録を作る操作自体をユーザーの承認の代わりにしてはいけません。
このCLIはdatasetの自動探索や代替選択をしません。

入力は `approval.json`、`proposal.private.json`、
`input_manifest.private.json`、`baseline_contract.private.json`、
`fixture_results.json` です。承認記録は提案hash、実際の承認への参照、
許可stage、時間・F/B・容量・候補数等の上限を持ちます。

```powershell
# 承認済みの入力とCPU fixtureの記録を検証し、コード・source mapを固定する
python sdxl_dq_mul_research.py --workspace .tmp/research_workspace --prepare

# 同じ契約で診断を起動、または完了済み成果物を検証して継続する
python sdxl_dq_mul_research.py --workspace .tmp/research_workspace
```

`--prepare` は実データのF/Bを実行しません。入力の読み取り・hash確認と
source map作成を行います。一度ジョブが始まると契約の上書きを拒否します。
コード修正が必要になった場合は旧契約と消費量を保持し、影響を検証してから
新しい実行記録を作ります。古いprefixのsource hashを書き換えて流用しません。

runnerはlockで二重起動を防ぎます。ジョブ時間には読込・warmup・集計・保存も
含みます。F/Bは開始前に累積計上し、失敗・再試行・候補名の変更でリセットしません。
上限不足、入力・コードの変更、比較の不一致、非有限値は停止理由です。
GPUメモリ不足を理由にbatch、解像度、dtype、rank、TE対象を自動変更しません。

完了jobは成果物hashを検証して再利用します。未完了stageを完了cacheとして
扱いません。強制終了でlockやactive jobが残った場合は、プロセスが終了したことと
未計上時間を確認してから復旧してください。無条件のlock削除は実装していません。

## 比較と保存

初期版の研究スケジュールは、mul 2.70 / 3.15 / 3.75のscope比較、UNet/TEの
高低交差、TEを高mulで固定したUNet4群の両方向介入、2ペア、独立noiseでの
確認と別regimeのdropout確認です。実行前の契約にこの範囲を含めます。

既存のprefix検査を通過した同一source契約を必須とし、別プロセスで作った
warmup snapshotの全成分も照合します。候補間ではstate、画像、noise、timestep、
共通UNet量子化乱数、dropoutを対応させ、TE条件ごとにembeddingを再計算します。
pilotでは既存UNet-only Localとの一致と候補順を反転した一致を検査します。

各stageには `result.json`、`per_image.jsonl`、`gradient_tail.jsonl`、
`recipient_energy.jsonl`、`interactions.jsonl` を保存します。
モジュールinventory、展開済みpolicy、snapshot、probe、時間・メモリ記録と
最終 `research_summary.json` も研究フォルダ内に保存します。

Body/Tailは既存v2.4のsource均等重みとsource bootstrapを再利用します。
P50は中央値、Bodyは95%点、Tailはtimestep帯別95%点の最大です。
群別内訳は加算可能な差分エネルギーと参照ノルムを併記します。
介入した群と変化を観測した群は同じとは限りません。

この数値は同一stateでの勾配変形を表し、画像品質の順位ではありません。
独立noiseで同じ画像を再測定しても、未知画像での確認とは呼びません。
学習後の品質、保存LoRAでの推論、resume、avgの検証は後段の実行範囲です。

初期版のprobe乱数addressは画像の選択順を含みます。そのため、異なる画像部分集合の
dropout OFF/ONの差をdropout単独の因果効果として扱えません。
継続用の `stable-image-v1` addressは画像キー・timestep帯・noise番号で決まり、
部分集合内の順序を含みません。モデルのdropout seedと量子化repeatも分離します。
旧版のaddressは元の観測の再現に残し、新しい結果にはaddress版を明記します。

更新を伴わないLocal passの `native_would_skip=false` はguardianの安全確認結果を
意味しません。`native_guardian_checked` で実際に検査したかを区別します。
更新passにはclip前normとAMPによるskipも記録し、`optimizer_step_performed` は
guardian等のskipだけでなくAMPのskipも反映します。
更新検査ではclip後の実勾配ノルム・hashと、同じ量子化repeatの対照候補に対する
実パラメータ更新差も記録します。旧外れ値は元のmodel/quantization addressのまま
実更新へ通し、更新前の勾配が旧記録と一致することを別途検査します。

## 通常診断でのTE・場所別mul

通常の `python -m dq_profile` は、Local診断の各mulをUNetとTEの双方に
適用することを既定にしています。`--dq-profile-te-quantized` で明示もできます。
従来のUNetのみの量子化は `--dq-profile-no-te-quantized` で選びます。
TEなしは量子化だけを無効にする指定で、TEの学習を止める指定ではありません。
通常のstandardモードの一律grid、2.70 / 3.15 / 3.45 / 3.75 / 4.05は維持します。

TEのON/OFFは `resolved_args.json` と `protocol_fingerprint.json` に記録します。
以前のTEなし診断とは条件が異なるため、数値を同一条件の反復として混ぜません。
通常学習の挙動は変更しません。既存のsnapshot・Prefix検算も維持し、
TE込みのLocal診断を、通常学習のdropout・更新・平均化まで再現したものとは扱いません。
内部stage用の直接診断CLIは互換性のため省略時の挙動を維持し、
LocalでTE込みにする場合は `--dq_profile_te_quantized` を指定します。

場所別の比較は `--dq-profile-policy-grid-file=grid.json` を使います。
直接診断CLIでは `--dq_profile_policy_grid_file` です。ファイルはgridの数値を
キー、上記のpolicy宣言を値とするJSONオブジェクトで、全grid点の指定が必要です。
この場合、ラベルの数値は比較条件の識別子であり、変更対象は各policyが決めます。
明示的なgridでは範囲外への自動延長を止め、途中でファイルが変わると停止します。
grid内のTE設定は上記ON/OFFと一致させます。不一致はGPU起動前にエラーにします。
これらの指定は現在 `v24-acceptance-local` のみに対応します。
`fixed_policy_assignments.json` に解決済み配分と入力hashを保存します。
場所別gridは研究用の明示指定であり、画像評価を済ませた推奨プリセットではありません。

## 段階的な継続研究と完走学習

`sdxl_dq_mul_continue.py` は別の承認済みworkspaceを使います。`--prepare` で
CPU検証記録・入力・コード・過去証拠・レビュー済みデスクトッププロセス一覧を
固定し、その後の起動でprefix、段階的診断、独立初期化した本学習を直列実行します。
実験開始後の契約変更は拒否し、失敗jobは原因のレビューなしに再試行しません。
本学習前の検証実装を修正する場合のみ、停止・旧契約/コード/消費量の保存・修正理由と
CPU再検証を揃えて `--prepare-revision` で新しい版を固定できます。旧成果物を
上書きせず、時間とF/Bはリセットせず、同じjobの承認済み試行数を越えません。

継続版のgridは2.70 / 3.15 / 3.45 / 3.75 / 4.05です。一律UNet＋TEとattn2を
確認し、attn2の役割、部位、最大4個別モジュールへ順に絞ります。探索の順位は
source除外時のP50改善の一貫性、次にP50差で決めます。個別モジュールの選定には
介入に対する勾配差分エネルギーを使いますが、因果寄与とは解釈しません。
attn2＋FFと、単独介入が支持された場合のみ1個別ペアを確認します。

追加の本学習候補は、独立noiseでP50差とそのsource bootstrap区間が負、全sourceの
leave-one-outで改善、dropout ONでもP50が改善してsource除外の悪化が最大1件、
既定対照と異なる配分、という条件を満たすものから選びます。最良改善の10%以内なら
変更モジュールの少ない方を優先します。適格候補がなければ6本目は作りません。
これらの規則は画像品質の予測精度を保証しません。

本学習は各条件40epoch、13,600予定stepで、有限値・実際のmul・guardian/AMPによる
更新見送りを監視します。更新見送りを補うために予定stepを延長しません。
追加監視の既定値は、非有限勾配を検出した時点で停止する `strict` です。
通常のguardian/AMPによる見送りを維持する研究では、承認と実行契約の両方で
`training_gradient_handling=verified_native_skip` を明示できます。この場合も
通常学習のしきい値・clip・optimizer・AMPの処理は変えません。非有限勾配を
検出したstepでは、既存処理が更新を見送り、パラメータとoptimizer状態が有限かつ
前後でbyte単位に一致することを検証します。AMPによる見送りではloss scaleの
低下も確認します。非有限loss・状態異常・当該stepでの更新・非有限勾配の3回連続
発生は停止条件です。通常の有限なGradNormスパイクの見送りはこの連続回数に
含めません。各イベントの明細と検証結果を研究フォルダへ記録します。

同じseed・引数でも、GPU演算を含む全学習の数値軌跡が完全一致するとは限りません。
設定と入力の一致、診断probeの再現性、全学習の数値的再現性を区別して確認します。
再実行のlossや見送り回数に差があれば記録し、条件間の差をmulだけの効果と断定しません。
一つの見送りstepの前後をbyte単位で検証することは、別実行同士の完全再現の証明とは
異なります。通常レシピを維持する比較では、再現のために演算設定を黙って変更しません。

各epoch、最終、final_rawの保存値・メタデータ・hashを確認して一覧化します。
`final_raw` もそれ以前のpromoteの影響を含み得るため、平均化履歴を併記します。
画像生成は行いません。学習用のF/B枠を診断開始前から確保し、latentsの準備と
avg shadowのforwardのみの処理も保守的に計上します。

明示的な場所別mulで保存した学習stateには、policy宣言と展開済み配分のhashも保存します。
`--resume` は保存時と同じ宣言・配分でのみ許可します。policyファイルを移動しても
内容と配分が同じなら再開できます。同じパスで内容を変えた場合、policy指定を外した
場合、またはpolicy記録のない古いstateへ明示policyを追加した場合は再開を拒否します。
設定を変える場合は別の学習runとして扱ってください。policyを指定しない従来のstateの
再開動作は維持します。推論用LoRAのA/Bテンソル形式は変更しません。


明示policyでresumeする場合は、NumPy乱数stateの復元に必要な型だけを、読み込みの間に
限定して許可します。古いAccelerateと、`torch.load` が既定で `weights_only=True` の
PyTorchを組み合わせたときの互換性対応です。任意のpickle読み込みを有効にはしません。
policy未指定の従来resumeと、新規学習の経路は変更しません。
