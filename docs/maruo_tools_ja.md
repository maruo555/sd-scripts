# SDXL LoRA ツールの操作ガイド

学習を終えたあと、数値で経過を調べるところから、生成画像を見比べて研究台帳に判断を残すところまでを扱います。各オプションの仕様は[独自CLIリファレンス](maruo_cli_ja.md)、ログを記録するタイミングや重みの保存順は[学習機能の解説](maruo_training_features_ja.md#run-output)にあります。

|やりたいこと|入力|成果物|
|---|---|---|
|[1回の学習を診断する](#report)|runの索引・設定・学習ログ・保存重み|run内の `diagnostic_report/report.html` と `report.json`|
|[重みのブロック別分布を見る](#density)|LoRAの重み1本|run内の `density/<重み名>.html` と `.json`|
|[複数runの学習曲線を比較する](#viewer)|各runの診断 `report.json`|ブラウザでの比較、任意で保存する比較HTML|
|[強度やLBWを指定して画像を作る](#generate)|SDXLベースモデル、LoRA、プロンプト|指定先のPNG|
|[条件をそろえて画像を比較・採点する](#compare)|モデル、比較条件、プロンプト、seed|比較フォルダ、3種類のHTML、採点CSV|
|[評価・履歴を研究台帳に残す](#ledger)|同じ出力ルートのrun、画像比較フォルダ、人が入力する所見|出力ルート直下の `ledger/` と評価時点のexport|
|[タグの分割・候補を調べる](#tokenizer)|タグ、または候補に含めたい語|TE1・TE2別のコンソール出力|

以下はWindows PowerShellの例です。リポジトリ直下で実行します。出力先と学習名を手元のものに置き換えます。

```powershell
$python = '.\venv\Scripts\python.exe'
$outputRoot = '..\lora_output'
$baseName = 'example_r4'
```

<a id="report"></a>

## 1回の学習を診断する

普段は学習名と出力ルートを渡せば、日時入りのフォルダ名を入力せずに済みます。

```powershell
& $python tools\make_lora_diagnostic_report.py --input_dir $outputRoot --base_name $baseName --lora_epoch_trend
```

該当するrunが1件ならそのまま進みます。同名のrunが複数なら新しい順に表示されるので、開始日時とUUIDを見て番号を選びます。非対話で複数見つかった場合は、表示された `--run_dir` 付きのコマンドで指定し直せます。完了時に表示された `diagnostic_report/report.html` を開きます。

- 学習中の勾配・loss、DQのclip率と倍率、rank、group別loss、promoteの判断など、記録した項目を確認できます。使わなかった機能や取得できなかった値は欠測として読みます。rank energyが1本でも欠けた部位は合計が欠測になり、全体を分母にする構成比も空欄になります。実測の0は集計に含みます。たとえばDQなしの構成では、DQのグラフは出ません。
- `--lora_epoch_trend` を付けると、保存したepoch重みも解析して分布の推移を表示します。重み解析を省いてログを先に見たい場合は、選んだrunに `--skip_lora_analysis` を指定します。
- 学習コマンドに `--report_after_training` を加えると、最終保存とログの確定後にそのrunへ `--lora_epoch_trend` 相当の診断を作ります。診断生成に失敗した場合、保存済みの重みとログから手動で再生成できます。

以降の例で同じrunを使うときは、一度だけフォルダを指定します。

```powershell
$runDir = Read-Host '対象runのフォルダのパス'
& $python tools\make_lora_diagnostic_report.py --run_dir $runDir --skip_lora_analysis
Invoke-Item (Join-Path $runDir 'diagnostic_report/report.html')
```

再生成すると、そのrunのHTMLとJSONを新しい生成IDで更新します。epoch推移つきに戻す場合は `--skip_lora_analysis` を外し、`--lora_epoch_trend` を指定して再実行します。単独のdensity用HTMLを手元に残すときは次の操作を使います。

<a id="density"></a>

## LoRAのブロック別分布・densityを調べる

1本の重みを指定して、UNet・TE1・TE2のどのブロックで重みが大きく、どこに値が集中しているかを見ます。HTMLには分布・RMSや特異値に基づく指標などがあり、画像上の有用性は生成比較で判断します。

```powershell
$runIndex = Get-Content -LiteralPath (Join-Path $runDir 'run.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$final = $runIndex.weights | Where-Object { $_.role -eq 'final' -and $_.status -eq 'saved' }
$weight = Join-Path $runDir $final.path
& $python tools\analyze_lora_density.py $weight --bins 128
```

runに登録された標準finalを選ぶと、索引上のepoch重みも参照して推移を作ります。`final_raw` や個別のepochファイルを指定すると、その1本を解析します。結果は `$runDir\density\<重みの拡張子を除いた名前>.html` と同名のJSONです。同じ重みで再実行すると同じ保存先が更新されます。run索引のない旧重みも直接指定でき、その場合は従来のファイル名から系列を探し、作業ディレクトリの `analyze_lora_result/` に保存します。

<a id="viewer"></a>

## 複数runの診断曲線を見比べる

1. [LoRAレポート比較](../tools/lora_comparison.html)をローカルのブラウザで開きます。PowerShellからは `Invoke-Item .\tools\lora_comparison.html` で開けます。
2. **JSONを追加**で各runの `diagnostic_report/report.json` を選びます。同じ名前のファイルでもrunのUUIDで識別します。比較表示には最大4 runを選べます。
3. グラフと表示方法を選びます。たとえば「倍率の判定前後」を「重ねる」で見て、「制御scopeのclip率と帯」でその理由を確認できます。「基準との差」は同じ系列名を突き合わせ、連続値は線形補間、離散値（自動倍率・low_autoの連続数など）は直前の観測値を使います。欠測区間は空きます。
4. 選択と表示を残すには **比較をHTMLに保存** を押します。保存したHTMLはデータを含むので、元のJSONがない環境でも開けます。同一runの診断を再生成したときは、新しいJSONを追加して差し替えます。

<a id="generate"></a>

## 部位別強度・LBWを指定して画像を作る

ここでは、すでに保存した重みを生成器に読み込みます。学習を続けながら途中の状態でサンプル画像を作る操作は[本家のrun control](run_control.md)にあります。

まず1枚で指定を試す例です。`$weight` には上で選んだ標準final、`$model` には学習に対応するSDXLベースモデルを指定します。

```powershell
$model = Read-Host 'SDXLベースモデルのパス'
& $python sdxl_gen_img.py --ckpt $model --network_module networks.lora_lbw --network_weights $weight --network_mul 0.8 0.25 0.5 --network_lbw XLMLT1 --prompt 'paper lantern, still life' --seed 39 --sampler euler_a --steps 20 --H 1024 --W 960 --bf16 --sdpa --outdir ..\generated
```

LoRA 1本の場合、`--network_mul` の1値はTE1・TE2・UNet共通、2値はTE共通・UNet、3値はTE1・TE2・UNetの順です。例ではTE1が0.8、TE2が0.25、UNetが0.5です。LBWはこの部位別強度にさらに掛かる12個の係数です。先頭は両TEに共通で、続く4個はUNetの入力ブロック4・5・7・8、次の1個は中央、最後の6個は出力ブロック0〜5に対応します。対応しない層の係数は1です。`ALL` は12個すべてが1のプリセットです。

プロンプト別に変えるときは `--from_file prompts.txt` を `--prompt` の代わりに指定します。例えば1行を `paper lantern, still life --d 39 --am 0.8,0.25,0.5` とすると、その行だけseedと強度を指定できます。強度を部位別にしたいときは動的にLoRAを適用し、重みの事前マージと事前計算は外します。生成されたPNGのメタデータでTE1・TE2・UNetの各強度と動的スロットのLBWを確認できます。

<a id="compare"></a>

## GUI・CUIで同じ条件の画像を比較し、blind評価する

### GUIで比較条件を用意する

```powershell
& $python sdxl_lora_report_gui.py
```

GUIを初めて使うときは、このリポジトリの仮想環境に `& $python -m pip install -r requirements-gui.txt` でGUI依存を追加できます。

1. SDXLモデル、プロンプトファイル、seed、出力ルートを指定します。プロンプトは例えば `paper lantern, still life` と `paper lantern, outdoors` を各1行とし、固定seedを2個用意します。
2. **Add files** でLoRAを追加し、1本ずつの条件は **Make single conditions** で作ります。条件内のLoRA行を選び、**Edit strengths** で部位別強度を設定します。LBWは同じ行の **LBW** 列で編集します。条件5つ・プロンプト2行・seed 2つなら計20枚です。
3. **Run report** で生成します。計画だけを確かめる場合は **Dry run** にチェックを付けます。比較ごとに「日時・実行名・試行IDの先頭」を含む新しいフォルダが出力ルートにできます。生成済みの比較は **Open report** で開けます。
4. まとめて実行するときは **Add to queue** → **Run queue** を使います。**Stop after current** は現在の比較が終わってから停止し、失敗・中止・部分完了の項目は **Retry selected** で待機に戻し、**Run queue** で新しい試行として実行できます。

比較結果は **Output root** の下に保存され、**Open report** で開けます。キューの保存先は `.tmp/queue/queue_state.json` です。GUIはこのファイルに記録した出力先と実行IDを使って、再起動後もキュー項目と比較結果を照合します。

### 同じ設定をCUIで再利用する

GUIは Model、Output root、Generation、Common args を `.tmp/lora_report_gui_last.json` に記憶します。Prompt file、LoRA条件、seed は比較ごとに指定します。比較をそのままCUIで再実行するときは、**Run report** が作成した比較フォルダの `config.json` を使います。

```powershell
$reportDir = Read-Host "GUIで作成した比較フォルダのパス"
Copy-Item (Join-Path $reportDir "config.json") .\study.json
& $python sdxl_lora_report_cui.py --config .\study.json --dry-run
& $python sdxl_lora_report_cui.py --config .\study.json
```

最初のCUI実行は設定とジョブを検査して試行フォルダとmetadataを作ります。画像生成は2回目に行います。各試行のコンソールに保存先が出ます。比較を別の場所へ渡すときは `metadata.json`、`config.json`、`images/` と各HTMLを含む比較フォルダごと保管します。

### 画像を見て判断を残す

|開くファイル|使い方|
|---|---|
|`report.html`|条件名を表示して、プロンプト・seedをそろえた画像を確認する。|
|`blind_report.html`|条件名を伏せて気に入った条件に投票し、**Reveal / 答え合わせ**で票数を見る。|
|`blind_report_detailed.html`|条件名を伏せて各画像を1〜3点で採点する。属性のチェックは点数と独立して選べる。|

詳細版で **集計・答え合わせ** → **集計CSVを保存** を押すと、条件別の合計・平均、点ごとの枚数と属性件数を出力できます。CSVの順位は合計点順で、同点は同順位です。採点数が違うときは平均点と評価可能画像数も合わせて読みます。途中入力はブラウザ内に保存されるため、評価結果を残すにはダウンロードしたCSVも比較フォルダと一緒に保管してください。実際の用途に使えるかという所見は、次の台帳に入力します。

<a id="ledger"></a>

## 研究台帳にrun・画像評価を登録し、exportする

学習の出力ルートごとに1つの `ledger/` を使います。新runは直下の `run.json` のUUIDで識別するので、同じ `base_name` の学習も別々に登録できます。台帳の正本と過去のexportはこのフォルダに残ります。新runの一覧には実効学習設定からoptimizerを表示し、一覧CSVの設定出典は `resolved` になります。旧台帳は記録済みの出典を使用します。

重み同士の共通区間を計算するときは、最終重みでもepoch途中で終わることを考慮します。一方だけが途中の場合や到達位置が異なる場合、そのepochは両側から除きます。終了位置の記録がない旧最終重みも最終epochを除き、特徴量の `range_note` に除外したepochと理由を記録します。

**runを登録する。** 次の `init` は新しい台帳を初めて作るときだけ実行します。すでに `ledger.json` があればそのまま `scan` から始めます。

```powershell
$ledger = Join-Path $outputRoot 'ledger'
if (-not (Test-Path -LiteralPath (Join-Path $ledger 'ledger.json'))) {
    & $python tools\lora_ledger.py init --ledger $ledger
}
& $python tools\lora_ledger.py scan --ledger $ledger
& $python tools\lora_ledger.py register --ledger $ledger
& $python lora_ledger_gui.py --ledger $ledger
```

`scan` は登録候補・参照状態・問題点を表示します。`register` は走査結果を台帳に反映します。`--run-id <UUID>` を加えると対象を選べます。後から診断を生成したときも再走査・登録します。登録済みrunの重みを内容のSHAまで照合したいときは `verify --full` を使います。これは照合後の参照状態を台帳に反映します。

**評価を残す。** GUIでは **比較の判断を記録** → **比較レポートから取り込む** で、画像比較の `metadata.json` を含むフォルダを選びます。評価の背景で「名前を伏せて評価」などを選び、実際に見た候補と所見を保存します。採点CSVはブラウザでの集計、台帳は本人が保存した判断を記録します。訂正は改訂履歴に残り、後日に画像を見直した評価は別に保存できます。

**評価時点の結果を書き出す。** 全登録runを含むexportは次のコマンドです。1本に絞る例は `--run-id $runIndex.run_id` を追加します。

```powershell
& $python tools\lora_ledger.py export --ledger $ledger
```

コンソールが示す `ledger/derived/<出力ID>/` に、その時点のrun・特徴量・評価・候補・問題点とmanifestなど9ファイルが並びます。再度exportすると別の出力として保存されます。GUIの **保存済み分析出力** で以前の出力を確認できます。**CSVを書き出す** は一覧・評価の詳細・訂正履歴を表計算用に保存する操作です。

**旧台帳を参照する。** 旧形式で別名のフォルダに保管している台帳は、GUIの **台帳を開く** から `ledger.json` のあるフォルダを選び、「旧台帳・読取専用」と表示されることを確認します。旧評価はその時点で保存された版、過去のexportは作成時の値を参照できます。旧台帳を新しい出力ルートの最初の台帳として引き継ぐ場合は、旧版での保存・exportを終了し、新しいルートにまだ `ledger/` がない段階で、移行用のコピー関数を使えます。次の例は新しい台帳を作るため、上記の `init` に先立って一度だけ実行します。旧台帳と新しい保存先のファイル数・内容を照合してから新しい場所で開けるようにします。

```powershell
$oldLedger = Read-Host '旧台帳のledger.jsonがあるフォルダ'
$ledger = Join-Path $outputRoot 'ledger'
& $python -c "import sys; from tools.lora_ledger_core.history_transfer import copy_legacy_ledger; copy_legacy_ledger(sys.argv[1], sys.argv[2])" $oldLedger $ledger
```

既存の新台帳と旧台帳を両方参照したい場合は、GUIからそれぞれの台帳を開きます。

<a id="tokenizer"></a>

## SDXLのTE1・TE2でタグの分割を確認する

キャプションに使いたい語を渡すと、SDXLの2つのtokenizerでそれぞれのトークン数、境界、IDが表示されます。標準ではBOS/EOSを省いて表示します。

```powershell
& $python tools\sdxl_tokenize.py --text 'paper lantern, still life'
& $python tools\sdxl_tokenize.py --search-core 'sample' --search-side suffix --search-min-add 1 --search-max-add 1
```

候補探索の例は `sample` の末尾に英小文字を1文字ずつ付け、両TEで指定の最小トークン数を満たし、分割片に1文字のものを含まない候補を比較します。結果はコンソールに出ます。既存のtokenizerを再利用する場合は `--tokenizer-cache-dir` にキャッシュの親フォルダを指定できます。対応するtokenizerがそこにあればローカルから読み、なければ取得して保存します。
