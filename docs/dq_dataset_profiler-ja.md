# SDXL DQ Dataset Profiler 利用ガイド

画像・フォルダ・キャラタグ別の追加診断、warmup前後比較、52画像化については、[データセット診断ガイド](dq_dataset_diagnostics-ja.md)を参照してください。

TE込みの標準化、場所別mul、dropoutの追加確認、レポート拡張については、
[2026-10-10採用の標準仕様](#spatial-diagnostic-spec)を参照してください。
標準入口は `canonical-v2` です。旧 `canonical-v1` は再現用に残しています。

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

### Local診断のTE量子化

`python -m dq_profile`の新標準 `canonical-v2` は、**UNet＋TE1＋TE2の量子化を固定**します。
一律5点と場所別配分を測り、dropout OFFを基本にします。`--dq-profile-dropout-on`を付けると
同じ条件集合でON測定も追加します。通常学習のdropout設定を変える指定ではありません。

従来のUNetのみの計測は `--dq-profile-preset=canonical-v1 --dq-profile-no-te-quantized` で
再現できます。標準v2でTEを外す指定はエラーとし、互換モードへの明示的な切り替えを求めます。
旧presetの既定値やコピーした学習経路は書き換えていません。

<a id="spatial-diagnostic-spec"></a>
## 標準診断：attn2・TE高mul型

2026-10-10採用・実装。Body代表からの追加配分、dropout ON確認、部位別集計、統合HTMLを
`canonical-v2` に組み込みました。小型LoRAのCPU forward/backward、通常学習CLIの回帰、
独立Body解析、オフラインHTMLの表示を検証対象としています。
**新プロトコル全体でのSDXL GPU実行・prefix/boundary受入は未実施**です。
過去の研究実験の受入結果を、この新しい標準経路のGPU受入へ読み替えません。
再実行できるCPUテスト、検証範囲、次のGPU受入項目は
[標準診断の検証記録](dq_standard_validation-ja.md)を参照してください。

### 公開CLIと出力

```powershell
python -m dq_profile --pretrained_model_name_or_path="D:\models\sdxl_base.safetensors" --dataset_config="D:\datasets\example\dataset.toml"
```

- 既定：`--dq-profile-preset=canonical-v2`。一律5点＋重複を除いた追加配分、TE込み、dropout OFF。
- `--dq-profile-dropout-on`：OFFの後、同じ候補をONでも追加計測。ONだけでBodyを選び直さない。
- `--dq-profile-uniform-only`：一律5点に絞る。TE込みと部位別記録は維持する。
- `--dq-profile-data-diagnostics=warmup` がv2の既定。学習前の対応付きforwardも行う。
  `local` は初期forwardを省き、`off` は画像inventory／raw MSE記録を省く。欠測は補わない。
- `--dq-profile-mode=strict`：snapshot／prefixの検算を深くする。v2ではStrictも一律5点を使い、範囲外への自動拡張はしない。
- `--dq-profile-dry-run`：GPUを使わず、コマンド・計画・見込みprobe数を保存する。
- 旧TEなし：`--dq-profile-preset=canonical-v1 --dq-profile-no-te-quantized`。

`report.html`／`beginner_report.html` は統合概要、`dataset_report.html` は画像・グループ別表示です。
旧形式の詳細表示は `uniform_report.html` と `technical_report.html` に保持します。
`ai_summary.json` と `observations.json` は匿名IDの共有用数値、`data.js` は画像表示用のローカル情報を
含みます。画像・caption・実パスがある `data.js` を匿名の共有用JSONと混同しないでください。
HTMLは同梱のPlotlyを読み、外部CDNへ接続しません。実行前に依存パッケージを更新してください。

### 名称と追加の背景

場所別mulの正式名称を **attn2・TE高mul型** とします。
UNetのクロスアテンションであるattn2と、TE1・TE2のLoRAに高側mul `H`、
attn2以外のUNet LoRAに低側mul `L` を適用する配分です。
attn2のQ/K/V/OutはDown/Mid/Upを通じて対象とし、FFは「その他UNet」に含めます。
「高mul」はその他UNetとの大小関係を表し、品質や安定性の保証ではありません。

| 表示名 | その他UNet | attn2 | TE1 / TE2 | 旧称 |
|---|---:|---:|---:|---|
| attn2・TE高mul型［固定基準］ | 2.70 | 3.75 | 3.75 / 3.75 | 元A、候補A |
| attn2・TE高mul型［Body基準・低側2.70］ | 2.70 | B | B / B | Body基準Aの低側2.70 |
| attn2・TE高mul型［Body基準・低側3.15］ | 3.15 | B | B / B | Body基準Aの低側3.15 |

`B`は、一律5点・dropout OFF・全体集計で求めたBody代表mulです。
この`B`は追加候補を作る規則であり、TEやattn2単独の最適mulや、最良画質を推定した値ではありません。
名称が長い図では「固定基準」「Body基準 L=2.70」等に短縮し、配分表・ツールチップに
`その他UNet / attn2 / TE1 / TE2`を明記します。旧ログの`attn2_A`等のID、学習済み
重みのファイル名、元の観測記録は変更しません。新名称は表示名と旧称の対応として保存します。

一律の高mulでは形が保たれる一方で表現が硬くなり、一律の低mulでは柔軟性が出る一方で
顔・衣装などの再現が弱くなる、という画像評価を出発点に場所別配分を調べました。
2つのデータセットで診断と40 epochの学習を行い、固定基準の配分に、
一律配分とは違う柔らかな質感やポーズの自由度が得られる例がありました。
最初のデータセットでは別seedでも特徴のある質感が観察され、別のデータセットでも
画像評価上の有望候補になりました。一方で顔の特徴が薄くなる例もありました。

高側を4.05へ上げた配分が常に改善したわけではなく、衣装・髪型の再現が弱まる例も
ありました。TEだけ、またはattn2だけを高mulにする比較にも、一貫した利点は確認できて
いません。**attn2単独の効果や、TEとの相互作用の因果関係はまだ確定していません。**
同一設定でも学習結果は変動し、診断の数値順位と画像の好みも一致するとは限りません。

したがって、追加の目的は固定基準を「最良設定」と推薦することではなく、
**一律mulだけでは見落とす配分を、同じ診断条件で比較できるようにすること**です。
固定基準を実験の参照点として残し、Body基準の2配分でデータセットへの反応を確認します。
公開ドキュメントには実データ名・画像・個人環境のパスを含めません。

### 変更の経緯と、2026-10-10に採用した仕様

当初の通常診断は、**UNetだけを量子化し、一律mulの5点をdropout OFFで比べる**構成でした。
その後の研究で、比較対象の通常学習ではTE側LoRAも量子化されていることを確認し、
診断との条件差を減らすため、先にLocal計測をTE込みが既定となるよう拡張しました。
この研究段階では、TEを外すオプションと、場所別配分を明示するpolicy gridも用意しました。

続く実学習・画像評価でattn2・TE高mul型が有望候補になったため、2026-10-10に、
**TE込みを標準条件として固定し、一律5点へ少数の場所別配分を追加する仕様**を採用しました。
以下は当初からの変更です。実装と検証範囲はこの節の冒頭に記載しています。

| 項目 | 当初の通常診断 | 2026-10-10に採用した標準仕様 |
|---|---|---|
| TEの量子化 | UNetのみを量子化し、TEは量子化しない | **UNet＋TE1＋TE2を標準プロトコルの固定条件**にする |
| 一律mul | 2.70 / 3.15 / 3.45 / 3.75 / 4.05 | 5点を維持する |
| 場所別mul | 通常診断には含めず、一律mulを比較 | **Body基準2条件＋固定基準1条件**を通常の追加比較にする |
| Local診断のdropout | OFF | **OFFを基本とし、CLIオプションでON測定を追加**する |
| 部位別の表示 | 主に全体の集計 | 全体・TE・TE1・TE2・UNet・attn2・その他UNetを切り替える |
| 冒頭のグラフ | 一律mulの曲線 | 同条件で測った場所別配分の点を重ねる |
| 画像別レポート | 数値mulを選択 | 配分を持つ「診断条件」を選択し、量子化の表示を連動させる |

TEを標準で固定する理由は、比較する通常学習でTE側LoRAも量子化していることと、
TEの量子化が全体の勾配比較に影響するためです。ここでの量子化対象はLoRAのdeltaであり、
ベースTEの重みを整数型へ置換する設定ではありません。

UNetのみの測定は削除せず、**過去結果の再現・要因分離をする研究／互換経路**として分離します。
`--dq-profile-no-te-quantized` は `canonical-v1` でのみ使用できます。
標準v2と互換v1はpreset・fingerprintで区別します。
TE ON/OFFを普段の標準診断の選択項目にはせず、OFFをONと同じ測定の反復として扱いません。
互換経路ではTE込みBody基準の追加配分を自動作成しません。UNet内だけの配分研究は別指定です。

通常学習の`--dq_delta_scope`と診断専用のTE指定は別です。
現行`canonical-v1`には互換性のため`dq_delta_scope=unet`が残っています。
通常学習用のコマンドを`both`へ移す方針を理由に、この既存presetを黙って変更しません。
新しい `canonical-v2` は `both` を明示します。診断へ旧 `unet` を渡した場合は、
上書き理由を記録してv2の `both` に統一します。通常学習自体の旧scope挙動は変更しません。

### 条件の作り方と測定の揃え方

1. 共通のwarmup状態で、TE込み・dropout OFFの一律5点を測定する。
2. 既存の有効性・Hard Safety・Body代表選出規則を適用し、一律5点から`B`を決める。
   追加配分で`B`を選び直す循環にはしない。部位別表示への切り替えでも`B`を変えない。
3. `L=2.70, H=B`と`L=3.15, H=B`、固定基準`L=2.70, H=3.75`を測定する。
4. ON追加確認が指定された場合は、OFFで確定した同じ候補集合をdropout ONでも測定する。

同じ配分は1回の測定へまとめます。`L=B`なら一律条件を再利用し、`L>B`ならそのBody基準
候補を省きます。`B=3.75`なら低側2.70と固定基準が重複します。
したがって固有条件数は**最大8条件**です。代表が決められない場合は高側を推測せず、
Body基準の自動追加を見送り、その理由を記録します。固定基準を実測しただけで
Body推奨や安全判定を通過したことにはしません。

候補間でsnapshot、画像入力、source group、noise、timestep、測定反復、重み付けを共通に
します。TEの条件ごとにembeddingを再計算します。更新を伴わない同じLocal計測として
一律と場所別配分を測り、別実験の曲線へ無条件に点を重ねません。
重複・派生画像の所属は同じsource-map版に固定し、グループ数を結果に合わせて変えません。
実行前の計画は6〜8条件の範囲、実測後の契約は重複除去後の実数を記録します。
画像数をI、追加固有条件数をEとすると、OFF追加は `I×4×2×(1+2E)` F/Bです。
先頭の1は追加配分用に同じno-quantを再測定する費用です。ON追加は
`I×4×2×(1+2(5+E))` F/B。warmup前評価は別に `I×4×3` forwardのみを使います。
一律のみ指定時はE=0でOFFの追加参照も省きます。HTMLの表示切り替えだけでは計測は増えません。

### 診断から実学習へ

実学習への受け渡しは、診断結果ファイルへの依存を作らず、**具体的なmulのCLI直接指定**とします。
レポートには各条件の全配分と、`--dq_delta_range_mul`、`--dq_delta_range_mul_attn2`、
`--dq_delta_range_mul_te`等の対応する引数を表示します。Body代表を使う場合も`B`という記号を
渡さず、診断で確定した数値を記載します。利用者は同じ引数を手入力しても学習できます。
直接指定の学習側実装と上書き順序は[通常学習への適用](dq_mul_research-ja.md#通常学習への明示的な適用)を参照してください。

### dropout OFFを基本にする意味

OFFは、量子化による変化をdropoutのマスク変動と分けて比較するための観測条件です。
新しい配分・新しいデータセットや、dropoutを使う長時間学習の前には、OFFの結果が穏やかでも
ONの追加確認を推奨します。通常学習のdropoutを無効にする指定ではなく、warmup・prefixのdropout設定も変更しません。
既存研究ではOFF/ONの差が一様とは限らず、OFFだけで実学習の挙動を代表できるとは断定しません。

ON追加確認では、候補とno-quantで同じマスクを使い、候補間でも入力・マスクを揃えます。
OFF/ONは同じ画像集合とnoise/timestepを使い、異なる画像部分集合の差をdropoutだけの効果と
呼びません。マスクの反復と量子化丸めの反復を区別して保存します。
両regimeを混ぜて1本のBody曲線にせず、同じ目盛で横並びにします。
ONで順位が変わる・差が広がる場合はそのまま示し、都合のよいregimeだけで推奨を選びません。
公開CLIは `--dq-profile-dropout-on` です。マスクは画像・時刻帯・noiseごとに変わり、
量子化丸めの2反復では同じマスクを使います。ONの結果は別regimeとして保存します。

### レポートの構成

初期の表示案には、研究用の12配分を縦に並べる独立した比較図がありました。
正式案では候補を一律5点と高mul型へ絞り、その比較を冒頭の曲線・追加点と数値表へ
統合したため、この独立図は基本画面に置きません。元の研究記録は保持し、
TEだけを変えた対照など、標準候補に含めない配分は研究用の比較として扱います。

基本表示は、次の4項目を維持します。

- **冒頭のmul曲線**：一律5点のP50・Body・Tailに、高mul型の実測点を重ねる。
  横軸はその他UNetのmul。一律は線と丸、Body基準は菱形、固定基準は星で区別し、
  高側mulは配分表示へ明記する。場所別の点を一律の線へ結んだり、未測定域を補間したりしない。
- **元の勾配方向への成分**：元勾配との平行成分のP50/P05を表示する。
  1は同じ平行成分、0は平行成分なし、負は逆方向、1超は増幅。
  1でも直交方向の変化があり得るため、勾配全体が同じという意味ではない。特徴の保持率や画質点とは呼ばない。
- **Warmupでの誤差減少（量子化OFF）**：共通warmup前後のモデル全体の誤差を表示する。
  未記録なら欠測とし、別runの初期値で埋めない。
- **量子化による変化**：warmup後の同じ重み・同じ入力で量子化OFF/ONの誤差差を表示する。
  追加学習後の改善や、学習の強さを直接予測する値とはしない。

基本の部位選択は「全体」「TE全体」「TE1」「TE2」「UNet全体」「attn2」「attn2以外のUNet」。
部位別の値は、全体で量子化した結果としてその部位に現れた勾配反応です。
TEの勾配変化をTE自身のdelta量子化誤差と同一視せず、原因の特定には介入条件の比較が必要です。
部位を変えるのは勾配指標であり、モデル全体の予測誤差をTE用MSE等に分解しません。
FF、attn1、Down/Mid/Up、Q/K/V/Outなどの追加軸は研究・詳細拡張に留め、基本画面には増やしません。

部位`S`ごとに`||g_ref,S||²`、`||g_quant,S||²`、`<g_ref,S, g_quant,S>`を保存し、
相対勾配差と平行成分を求めます。部位の参照ノルムが小さすぎる場合は未算出にします。
部位別の内積がない旧記録から平行成分は復元しません。全体の値を部位別として転用しません。
P50／Body／Tailには全体と同じsource均等重みの規則を用いますが、部位の分母が異なるため
部位別P50やBodyを足して全体へ戻すことはできません。

「もう少し詳しく」には、P10–P90・P25–P75の分布、source bootstrapの参考95%区間、
TE1・TE2・attn2・その他UNetの重複しない4区分の内訳を置きます。
内訳は全体の参照勾配を共通分母にした平均二乗差の総量と100%構成比を対で表示します。
これは**勾配差が現れた場所**であり、誤差を発生させた原因の割合ではありません。
分位幅、bootstrap区間、別seedでの再学習のばらつきも区別します。

AI用データには条件ID、旧称、全配分、実際の量子化対象、regime、測定同一性、
source-map版、候補の再利用元、部位別統計、欠測理由を保存します。必要に応じて
匿名IDの観測行も書き出します。画面に出さない数値を捨てず、配布用の集計と
実パス・captionを含むローカル記録を分けます。
共有JSONの`schema_version=spatial-shared-v2`では、`loss_contract`に使用する誤差の種類・定義・
raw MSEの記録率と欠測理由を保存します。観測行は`raw_mse`と`objective_loss`を別々に保持します。
全行にraw MSEが揃う場合だけMSEで比較し、一部でも欠ければ全条件をobjective lossに統一します。
互換用の`reference_loss`／`quantized_loss`には選択した種類を入れ、行の`loss_kind`で明示します。

ローカルの画像詳細にはcaption・subset・repeat・提示履歴と保存済みサムネイルを残します。
同じ画像に複数の学習設定がある場合は全設定を表示し、その中からwarmup反応を推測で選びません。
これらの個人データは共有JSONに含めません。
最終HTMLの全書き出し後に`analysis_manifest.json`と`standard_report_manifest.json`の
出力パス・SHA-256を更新します。解析stageの元manifestと観測入力の記録は保持します。

データセット間で絶対値を比べるときは、量子化対象・probe regime・モデル・warmup・
重み付け等の測定条件も併記します。同じ目盛にしただけでは同じ比較条件にはなりません。
この新仕様でも、診断だけで絵の硬さや顔・衣装の再現を判定せず、最終判断は画像評価で行います。

## 3. 実用診断の共通基盤

以下は共通のsnapshot／Prefix／一律Local基盤の説明です。v2の追加配分と新HTMLは上記仕様に従います。
過去の計測時間例にはv2の追加配分・ON確認・初期forwardの費用を含みません。

通常利用の`canonical-v2`と互換用`canonical-v1`は、40 epochを最後まで学習する処理ではありません。
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
- TOMLの`[general]`／dataset／subset fallbackを解決し、batch・bucket設定（`bucket_no_upscale=false`を含む）が選択したpresetと一致することを確認する。
- CLIが選択したpresetと互換である。
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
warmupします。両presetとも40 epoch相当の総stepと5% LR warmupから境界が決まります。

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

- 画像数: Standard／Strictとも8～52。上限を超えてもprobe budgetは増えない。
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

新v2は両modeとも固定5点＋同じ追加配分です。以下の端点拡張は旧v1の説明です。

`standard`は`2.70, 3.15, 3.45, 3.75, 4.05`を1 processで一度だけ測ります。
端点でも改善傾向が続く場合は`edge_unresolved`と表示しますが、範囲外を追跡しません。
この場合、単一代表を出さず、Fidelity retained候補を1点へ自動縮約しません。

Standard／Strictとも最大52画像、4 timestep帯、no-quant 3 replicas、candidate 2 noise ×
2 quant repeatsを使うため、Local Body／Tailの物差しは共通です。独立source groupが画像上限を
超えるdatasetも実行できますが、最大52群だけを決定的にprobeします。全groupはsource contractに
残り、レポートには`probe / total`を表示します。未probe群がある結果は完全coverageと同一視しません。

旧 `canonical-v1` の `strict` はcore grid `2.70, 3.15, 3.45`から開始します。候補集合が測定端に残る場合だけ、
最大2段まで外側を追加します。下端側は`2.25`、なお未解決なら`1.80`、上端側は`3.75`、
なお未解決なら`4.05`です。両端が残る場合は両方向を同じroundで追加します。
edge追加時は以前のmulも含む拡張grid全体を別processで再測定し、共通mulの全probe行を
exact parityで検査します。Strictの再測定は校正能力を高めますが、主要な時間増加要因です。

### 3.6 CPU解析とレポート

最後にsource groupを等重みとするbootstrapを2,000回行い、Body、Tail、95%区間、
Fidelity retained set、robust dominance、source LOOなどを作ります。`report.html`、
`beginner_report.html`、`technical_report.html`、JSON、CSVへ保存します。

旧v1の`beginner_report.html`は最上部のMul affinity curveから読み始められる概要版です。
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
  --dq-profile-preset="canonical-v2" ^
  --dq-profile-mode=standard ^
  --dq-profile-open-report ^
  --pretrained_model_name_or_path="D:\models\sdxl_base.safetensors" ^
  --dataset_config="D:\datasets\example\dataset.toml" ^
  --output_name="example_dataset_r4"
```

現在の長い通常学習コマンドを再利用する場合は、先頭の
`accelerate launch ... sdxl_train_network.py`を`python -m dq_profile`へ置き換えます。
ただし、過去commandに`AdamW8bit`、`native_accum`、異なるdimなどが含まれると
選択したpresetとの衝突で停止します。最小構成を使い、presetへ固定値の指定を任せる方法が
もっとも安全です。

### 6.2 診断入口のCLI一覧

| オプション | 必須 | 既定値 | 用途 |
|---|:---:|---|---|
| `--pretrained_model_name_or_path` | 必須 | なし | SDXL base modelのファイルまたはディレクトリ |
| `--dataset_config` | 必須 | なし | kohya形式dataset TOML。実効`resolution`と各subsetの`image_dir`が必要 |
| `--output_name` | 任意 | dataset TOMLのstem | 診断名のfallback。パス区切りを含まない名前 |
| `--dq-profile-name` | 任意 | `output_name` | datasetごとの親フォルダ名 |
| `--dq-profile-output-dir` | 任意 | repositoryの`..\lora_output\dq_dataset_profiler` | 診断runを格納する基底ディレクトリ |
| `--dq-profile-preset` | 任意 | `canonical-v2` | TE込み・追加配分の新標準。`canonical-v1` は再現用 |
| `--dq-profile-dropout-on` | 任意 | false | 同じ配分のdropout ON計測を追加 |
| `--dq-profile-uniform-only` | 任意 | false | 追加配分を省き、一律5点のみ測定 |
| `--dq-profile-data-diagnostics` | 任意 | v2は`warmup`、v1は`off` | 初期評価・画像別raw MSEの記録 |
| `--dq-profile-no-te-quantized` | 任意 | false | v1でのみ許可。旧UNetのみの測定 |
| `--dq-profile-mode` | 任意 | `standard` | `standard`: 最大52画像・snapshot 1回の日常診断、`strict`: 独立snapshot A/B・長いreference QA（v1のみbounded edge再測定） |
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

### 6.3 presetが固定する学習設定

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
| DQ | `dq_delta_scope` | v2は`both`、互換v1は`unet`。v2へ旧unetを入力した場合は理由を記録してbothへ統一 |
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
| probe画像数 | Standard／Strictとも`min(dataset実画像数, 52)`。最低8画像 |
| timestep bins | `4` |
| no-quant replicas | noise 3回 |
| candidate replicas | noise 2回 × stochastic quant 2回 |
| Local dropout | 基本はoff (`structural_dropout_off`)。v2はCLI指定で対応付きON確認を追加 |
| Prefix dropout | on。通常学習に近いprefix再現性検査 |
| update branch | 製品Localでは0。128-step Trajectoryは研究専用で実行しない |
| Guardian ablation | `common_only` |
| CountSketch | 幅512、独立seed 2個 |
| Accelerate process | `num_processes=1`、`num_machines=1`を各stageへ明示 |
| CPU threads/process | `8` |
| bootstrap | source単位、2,000回、固定seed |

### 6.5 execution modeごとのQA・候補探索契約

以下のgrid・edge extension・GPU process数は旧`canonical-v1`の候補探索です。
新`canonical-v2`では両modeとも一律5点を使い、edge extensionは行いません。
その後にBody基準の追加配分を同じLocal workerで計測します。snapshot／prefixのQA深度は下表のままです。

| 項目 | `standard` | `strict` |
|---|---|---|
| 用途 | 日常の正式dataset診断 | コード／CUDA／PyTorch／bitsandbytes変更後、リリース前、再現性調査 |
| 独立snapshot | 1回。Prefix processとの境界一致を検査 | A/Bの2回 |
| Prefix | 8A／8B／16@8 | 64A／64B／128@64 |
| state checkpoints | `0, 1, 4, 8` | `0, 1, 32, 64` |
| Prefix branch updates | 64 | 512 |
| Local画像上限 | 52 | 52 |
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
- `matched_preset`: 選択presetと一致するため許可する。
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
