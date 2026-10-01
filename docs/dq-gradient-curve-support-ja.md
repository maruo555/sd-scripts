# Body/TailのP50と元方向成分の補助グラフ

`beginner_report.html` と通常の `report.html` の共通グラフに、勾配距離dのP50と、直下の元方向成分P50/P05を表示します。通常レポートの自動スケール拡大図にも反映します。表示操作で候補選択や推薦Mulは変更しません。

## 集計契約

- dは保存済み `relative_gradient_distance`。既存Bodyと同じ `source_group` 等重み・source内保存観測等重みで、全binをまとめてP50を計算します。source未記録時の `image_key` フォールバックも既存Bodyと共通です。
- aは**各sample行**の `gradient_norm_ratio * gradient_cosine`。ratio未保存時だけ、同じ行の `grad_norm_candidate / grad_norm_noquant` を利用できます。aのP50/P05も同じsource重みです。repeat・noise・画像の事前平均はありません。
- `_cluster_quantile` → `_cluster_values` → `_weighted_quantile` をそのまま利用します。昇順の累積重みが分位点×総重みに初めて達する観測値で、補間しません。
- 既存の浮動小数累積計算も維持します。**CDFが分位点にちょうど一致する境界では、観測数の増加による丸めの差で隣の観測値になる場合があります**。例：同じsourceの重み0.1を10回足しても厳密な1になりません。整数重みへの置換やepsilonによる補正は、既存Bodyとの計算互換性と指定された実測照合値を変えるため導入していません。sourceの総重みは数学上1で、通常の境界外のfixtureでは分布を保った観測数増加・学習repeat設定・行順による不変を検証しています。この厳密境界の制約自体も回帰テストに記録しています。
- P05は5パーセンタイルです。最小値・最悪bin・CI下限ではありません。追加指標にCIはありません。
- source数は測定画像グループ数です。元絵数とは認証しません。上下のP50が同じ観測を指す保証もありません。

`gradient_curve_support.json` と、実用レポートmodelの `gradient_curve_support` namespaceに、値・状態・理由・件数・定義version `1.0.0`・`selector_input=false` を保存します。既存selection、summary、rawログとそのhash対象には追加しません。productionのpromote対象にも新JSONを含めます。

## 有効性と旧ログ

対象は採用された同一runの `gradient_tail.csv` または `raw_gradient_tail.csv` の `record_type=sample` 行です。候補名とMulは既存model/scoreの対応を利用し、候補名から推測しません。

- 測定keyは既存契約と同じ `(image_key, timestep_bin, noise_replica, quant_repeat)`。HardSafety合格候補のkey集合を使い、欠けた候補を共通部分に縮めて表示しません。画像とsourceの矛盾、候補/Mul不一致、候補間の基準norm不一致も検出します。
- 完全重複は1観測。矛盾する重複は不整合。保存されたrun/snapshot/edge round等が混在する場合も未算出にします。
- 余分なCSV列があるsample行は候補の追加指標を未算出にし、`malformed_sample_columns`を記録します。観測を黙って除外したり、追加集計だけの理由で既存診断全体を失敗させたりしません。
- 明示されたtopology不一致、非finite、無効norm、保存された勾配無効理由を検査します。no-quant normが `1e-12` 以下の場合は、既存dataset diagnosticsの近ゼロ基準に沿って未算出にします。分母をepsilonへ置き換えません。
- 量子化normが明示的に0で基準norm等が有効なら、cosineが未定義でもa=0です。
- dとaは別々に検証します。無効行を落として再集計せず、その指標を候補全体でnullにします。線は欠測点で途切れ、未測定Mulの補間もしません。
- 必要なスカラーと測定keyがそろう旧CSVは、`data_diagnostics` のsidecarなしで利用できます。topology列がない旧CSVは未記録件数を保存し、整合を認証したとは扱いません。
- 保存済みBodyと同じ入力から求めたP95も照合します。母集団が合わなければ追加指標は未算出です。集計済みBody/Tailしか残っていないログから追加値を推定しません。

## GPUなしで再生成する

リポジトリのルートで、通常のPython環境（NumPyが必要）から実行します。

```powershell
.\venv\Scripts\python.exe -m tools.rebuild_dq_gradient_report `
  "D:\path\to\saved-run" `
  --output-dir "D:\path\to\rebuilt-report"
```

入力には保存済みの単一datasetの `practical_report.json` が必要です。既存の候補状態・説明・Body/Tail・CIをそのまま読み取り、判定・bootstrapをやり直しません。GPU、torch、訓練runtimeは不要です。通常診断の終了時には既存の `tools.analyze_dq_v24_local` から自動生成されます。

入力CSVは同じフォルダの `raw_gradient_tail.csv` / `gradient_tail.csv` を使います。見つからない場合、既存analysis manifestが記録した**正確なパスとhash**が利用可能ならそのファイルを使います。他runやedge roundを探索・結合しません。

CSVを別の場所へ移した場合は、明示的に指定できます。

```powershell
.\venv\Scripts\python.exe -m tools.rebuild_dq_gradient_report `
  "D:\path\to\saved-analysis" `
  --gradient-csv "D:\path\to\same-run\gradient_tail.csv" `
  --output-dir "D:\path\to\rebuilt-report"
```

保存済みsummary/manifestにCSV hashがあれば一致を要求します。不一致・曖昧な複数CSV・CSV欠落では、理由付きの未算出グラフを生成します。hashが残っていない旧ログでは、同じフォルダのCSVまたは明示されたCSVとして出典を記録し、測定key・source・P95の検証を行います。hashによるrun同一性の検証はできません。

出力は追加JSON、追加namespaceを持つ `practical_report.json`、`beginner_report.html`、`report.html` です。元の技術レポートがあれば同じ内容をコピーします。rawログと既存判定ファイルは書き換えません。再生成先を入力と別にすれば、入力model/HTMLも維持できます。

元の `analysis_manifest.json` がある場合は、出力先へ保存し、再生成したHTMLのパス・hashを更新します。入力ログと選択判定の検証情報は維持します。同じフォルダへの再生成でも、レポートのhashが古いまま残らないようにします。

## 表示と操作

上段は既存の固定Y軸・CI・1.0基準・preset・edge・HardSafety表示を維持します。追加P50は緑の破線と菱形です。軸上限超過は三角で示し、詳細表と数値表示で実値を確認できます。

下段は0〜1.1、範囲外のP50/P05がある場合は下段だけ拡張します。負値・1超を丸めません。0と1の目盛と「元方向への成分が同じ大きさ」を表示します。

上下の同じMulへマウスを置くと縦線・点を連動強調します。クリック／タップ／Enter／Spaceで固定・解除、Escで解除できます。Tab・左右キーでも数値を確認できます。全候補の実値・対象件数・未算出理由はグラフ下の詳細にも表示します。

## 検証

今回の関連CPUテスト結果は **108 passed**（diffusersの既存FutureWarningが2件）です。概要・通常レポートの双方でオフラインブラウザ検証も通過しました。通常の診断完了経路では、追加指標の算出／未算出を切り替えても `local_selection.json`、`local_acceptance.csv`、`summary.json`、`source_bootstrap.csv` がバイト単位で不変でした。

```powershell
.\venv\Scripts\python.exe -m pytest `
  tests/test_dq_profile_gradient_support.py `
  tests/test_dq_profile_v24_practical_report.py `
  tests/test_dq_profile_v24_beginner_report.py `
  tests/test_dq_profile_v24_acceptance.py `
  tests/test_dq_profile_v24_descriptive.py `
  tests/test_dq_profile_production.py -q
```

ブラウザ検証にはPlaywrightとEdge（または環境変数 `DQ_BROWSER_EXECUTABLE` で指定するChromium系ブラウザ）を使えます。外部通信を遮断し、初期表示・X位置・ホバー・固定解除・タッチ・キーボード・モバイル表示・JavaScriptエラーを検査します。

```powershell
node tools/validate_dq_gradient_report.cjs `
  "D:\path\to\rebuilt-report\beginner_report.html" `
  "D:\path\to\rebuilt-report\stacked-curves.png"
```

実測ログとの照合はローカルで実施済みです。公開用の文書にはデータセット名、run ID、元ファイルのhash、個別の実測値を含めず、再現可能なCPU fixtureを検証基準にします。

処理時間・メモリ・出力サイズは入力の件数によって変わります。追加forward/backward・optimizer更新は0回で、追加値は保存済みスカラーのCPU集計だけで生成します。

## 変更ファイル

| ファイル | 変更内容 |
|---|---|
| `dq_profile/v24_gradient_support.py` | CSVスカラー検証・追加分位点・表示用namespace |
| `dq_profile/v24_gradient_curve.py` | 下段グラフ・上下連動・値と理由の詳細表示 |
| `dq_profile/v24_practical_report.py` | 既存グラフへd P50追加、上下グラフの共通描画 |
| `dq_profile/v24_beginner_report.py` | 初期表示への反映、Body/Tail説明の整理 |
| `tools/analyze_dq_v24_local.py` | 通常診断完了時の追加集計とJSON出力 |
| `dq_profile/production_runner.py` | 新JSONのpromote |
| `tools/rebuild_dq_gradient_report.py` | 保存済み判定を使うCPU再生成CLI |
| `tools/validate_dq_gradient_report.cjs` | オフラインブラウザ検証・画面保存 |
| `tests/test_dq_profile_gradient_support.py` | 集計・異常入力・不変性・CLI/production fixture |
| `tests/test_dq_profile_v24_practical_report.py` | 数値確認先の説明変更に対応 |
| `tests/test_dq_profile_v24_beginner_report.py` | P95/P50凡例と新しい説明を検証 |
| `docs/dq-gradient-curve-support-ja.md` | 本文書 |
