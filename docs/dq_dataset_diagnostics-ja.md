# 画像・フォルダ・タグごとのDQ診断

2026-09-05。診断用trainerのLocal計測から実データを保存し、独立したHTMLを生成する追加機能です。通常の学習trainerは変更しません。

## 実行する

普段の診断コマンドに次を追加します。

```powershell
--dq-profile-data-diagnostics=local
```

`local`は「warmup終了時の重みを固定し、量子化を適用したときの局所的な変化を調べる」という既存のLocal計測に由来します。「ローカルPCで動かす」という意味ではありません。基本の追加診断を有効にする指定です。

| 指定値 | 追加レポート | 量子化OFFの学習前後比較 | 追加のモデル評価 |
| --- | --- | --- | --- |
| `off`（既定） | なし | なし | なし |
| `local` | あり | 初期値がないため不可 | forward/backward回数は増やさない |
| `warmup` | localの内容を含む | あり | 初期状態で12V回のforward |

指定は一つです。`warmup`にすると`local`の内容も含まれるため、両方のフラグを並べる必要はありません。ここでのモードは追加診断の範囲であり、既存診断自体のwarmup学習を有効・無効にする設定ではありません。

`local`は既存Localのforward/backwardからraw MSEと直接の勾配差を取得します。モデルの評価回数は増えませんが、GPU上のMSE集約、CPUへの値の転送、保存・集計は追加されます。従来ログだけを読み直しているわけではありません。HTML内の絞り込み・フォルダやタグの集計・グラフ表示にはGPUを使いません。

モックのようにwarmup前後も比較する場合は、次を指定します。

```powershell
python -m dq_profile `
  --dq-profile-mode=standard `
  --dq-profile-data-diagnostics=warmup `
  --pretrained_model_name_or_path="D:\models\sdxl_base.safetensors" `
  --dataset_config="D:\datasets\example\dataset.toml"
```

`warmup`は学習準備直後の実際のLoRA初期状態をCPUに保存します。Localで確定したlatent・noise・target・caption/token・crop/flip条件を使い、その初期状態でもforwardします。学習対象TEのembeddingも初期TE-LoRAから再計算します。backwardやoptimizer更新は行いません。前後のRNG・network・optimizer・scheduler・scaler・guardian等を照合し、復元失敗は実行エラーとします。

既定は`off`です。小さなCPU/CUDAモデルと実際の診断用LoRAを用いたテストはありますが、全SDXLのoff/local/warmup比較は未実施です。warmupの既定有効化はその確認後に行います。`off`でも以下の52画像契約は共通です。

## 「量子化前」と「量子化後」の区別

| 状態 | 重み | 量子化 | 測定モード |
| --- | --- | --- | --- |
| A：学習開始前 | 初期LoRA | OFF | warmup |
| B：warmup終了時 | warmupで更新したLoRA | OFF | local / warmup |
| C：Bにmulを適用 | Bと同じ重み | ON | local / warmup |

**A → B** が量子化前の学習反応です。同じ固定入力に対するraw MSEと改善率を比較します。学習しやすさを考える手掛かりですが、測定した訓練入力への反応であり、将来の学習や生成品質を保証する指標ではありません。`local`ではBしかないので学習による改善は算出できません。

**B ↔ C** が量子化の影響です。Cでさらに学習した結果ではなく、Bと同じ重みに量子化を適用した瞬間の誤差・勾配を比べます。mul別の折れ線はこの比較です。レポートの旧称「後MSE」はB、すなわちwarmup後・量子化前の値でした。画面では「B warmup終了MSE」と量子化OFFの見出しを併記します。

## 2つの「画像ごとの反応」の読み方

以前の切り替え式表示で「AのMSE × 改善率」を選んだ図が、**A → B：量子化OFFの学習反応**です。現在はこの図と**B ↔ C：量子化の影響**を常時表示します。広い画面では左右、狭い画面では上下に並び、切り替え操作は不要です。

| 図 | 横軸 | 縦軸 | 見ること |
| --- | --- | --- | --- |
| A → B：量子化OFFの学習反応 | 学習開始前Aのraw MSE | `(AのMSE − BのMSE) / AのMSE` | warmupで誤差がどの割合だけ減ったか |
| B ↔ C：量子化の影響 | warmup終了時Bの量子化OFFのraw MSE | 量子化による平均勾配変化d | 同じ重みへの量子化で勾配がどれだけ変わるか |

A → Bの図は、右ほど初期誤差が大きく、上ほど誤差の減少割合が大きい、という読み方です。例えばMSEが0.30から0.24なら改善率は20%です。0%なら変化なし、負なら誤差が増えています。初期誤差が小さい画像は改善の余地も小さい場合があるので、低い改善率だけで「学習しにくい画像」と判断しないでください。

この図は**今回のwarmupでの学習反応の違い**を調べる手掛かりです。画像固有の難易度を測るものではありません。初期状態、caption、noise/timestep、warmup中の提示回数や更新回数にも左右されます。気になる点を選び、画像詳細の元caption・提示/更新/skip回数やbin別の値も一緒に見てください。例えば「同じ程度の初期誤差なのに改善が少ない画像」を見つけて、条件を確認する使い方ができます。キャラgroupの値は画像全体の誤差であり、キャラ領域だけの学習度ではありません。

画像・フォルダ・タグの選択、指定キャラタグの絞り込み、timestep bin、全体の背景表示は両図に共通です。一方の点を選ぶと、同じ画像の縁取りと詳細がもう一方にも連動します。mulを変えるとB ↔ Cの値だけが変わり、A → Bの値は変わりません。同じ物理画像でも、選択したタグ・subset・解像度のsampleが未測定なら、その別文脈の測定値を濃い点へ転用しません。別文脈の値は全体の背景にだけ表示し、背景をオフにすると消えます。両図は軸も単位も異なり、点の高さをそのまま比較するものではありません。欠測の条件が異なるため、描画される画像数が一致しない場合もあります。

`local`でもA → Bのパネルは残し、「初期評価がないため未算出」と表示します。0点や架空の初期値は補いません。A → Bも見るには`--dq-profile-data-diagnostics=warmup`を指定して再診断します。2図の同時表示によってGPU計測は増えません。

## 52画像と時間

production standard/strictは`local-body-tail-v2`、最大52実画像を使います。旧`local-body-tail-v1`の32画像契約は読み取り・再現用途に保持します。画像を複製して52枠を埋めません。従来のBody/Tail計算・Hard Safety・候補絞り込みはそのままで、母集団が増えたことによる数値変化は起こり得ます。

standardの5候補では、選択実画像数をVとしてLocalは92V回のforward/backwardです。上限を使い切る場合、32画像の2,944回から52画像の4,784回へ62.5%増えます。元から32枚以下なら上限増による増加はありません。

`warmup`はさらに12V回のforwardのみを追加します（52画像なら624回）。実行計画に別枠で表示し、未測定のforward時間係数を既存の時間見積もりへ混ぜません。standardのstandalone snapshot/prefix workerでは初期評価しません。strictのedge再走査は各Local workerで初期評価も繰り返します。

## 出力と操作

診断run直下の`data_diagnostics/dataset_report.html`を開きます。既存の量子化レポート3種類にもリンクを追加します。

mul・timestep・「キャラを指定」の操作バーはスクロール中も画面上部に残ります。下のグラフを見たまま条件を切り替えられます。「キャラを指定」を押すと設定欄へ移動し、「閉じる」で元のスクロール位置へ戻ります。

- **画像一覧**：表示名は「親フォルダ名 / ファイル名」です。散布図のポップアップ・画像比較・詳細見出しでも同じ形式を使い、`base.png`等の同名ファイルを区別します。物理画像を一行にまとめ、解像度・subsetが違うsampleを詳細で分離します。未測定sampleへ別sampleの値を転用しません。
- **フォルダ一覧**：loaderが解決した`image_dir`絶対パスで自動集計します。同名の別フォルダを混同しません。TOMLにgroupを追記する必要はありません。
- **タグ一覧**：「キャラを指定」で元captionに実在するタグを選びます。captionファイルが優先されるloaderの規則を守り、`class_tokens`を勝手に連結しません。元captionと評価時captionは別に保存します。
- **散布図**：全有効画像を薄く表示し、選択対象を強調します。軸の範囲は全体・全mulで固定し、選択変更で縮尺を動かしません。bin・filter変更時は対象に合わせて再計算します。学習反応と量子化の影響の2図を常時表示します。
- **mul比較**：画像・フォルダ・タグそれぞれ最大6対象。各対象の内部で、全mulに共通する完全なloss/勾配の有効sampleを使います。違うキャラ同士の画像集合の積集合は取りません。クリックで対象とmulを上の表示へ同期します。

`character_a,character_b,2girls`の画像は、character_aとcharacter_bをキャラとして指定すると双方に所属します。lossを二分しません。「指定キャラタグが1種類の画像のみ」は、指定キャラタグと元captionの共通部分がちょうど1種類のsampleを残します。未指定の人物やタグ漏れは判定できないので、一人だけの絵だとは断定しません。

`.txt` captionは使えます。ただしproduction入口の「4以上の有効source/image_dir」条件は維持します。文字どおり1フォルダだけの入力を新たに対応させたものではありません。

## サムネイルと並べ替え

グラフの点や画像一覧で画像を選ぶと、グラフ直下にサムネイル・親フォルダ付きの名前・元captionを表示します。クリックで拡大し、Escapeまたは「閉じる」で戻れます。複数sampleがある画像では元captionを列挙し、sample別の診断値へのリンクを表示します。

サムネイルは測定済みの物理画像ごとに1枚、縦横比を保って長辺最大512pxのJPEGとしてHTMLに埋め込みます。EXIFの向きを反映し、透明部分は白にします。画像の元ファイルを移動した後もHTML内で表示できます。これは初回サムネイル作成時の元画像の縮小版で、診断時のcrop・flipを再現したものではありません。

`thumbnails.json`にも保存し、CPU再集計やレポートの再生成で再利用します。保存済みのサムネイルは元画像の後日の変更に追従しません。元画像が見つからない・読み込めない場合は説明を表示し、数値レポートは生成します。未測定画像のサムネイルは埋め込みません。

画像・フォルダ・タグの一覧は、名前、改善率、warmup終了時MSE、初期MSE、平均d、測定画像数で並べ替えできます。昇順・降順とも未算出は末尾に置き、同値は登録順を保ちます。数値は現在のmul・bin・絞り込みを反映します。例えば画像の確認候補を探すなら「改善率・昇順」、量子化による変化を確認するなら「平均d・降順」を使えます。並べ替えを変更すると一覧の先頭ページへ戻りますが、選択中の画像・比較対象は維持します。計測や推薦結果には影響しません。

## group設定の再利用

HTMLからgroup-map JSONを保存し、次回は任意で次を指定します。

```powershell
--dq-profile-group-map="D:\datasets\dq-character-groups.json"
```

添付仕様の`dataset-groups-v1`形式を受け付けます。`tags_any`はOR、非空の`tags_all`はAND、`subset_groups`と`image_paths`は別の所属根拠としてORです。既存subset groupも自動で含めます。

Unicode NFCと前後空白だけを正規化します。caseやunderscoreを同一視しません。明示aliasは一段だけで、連鎖・循環を拒否します。Pythonで読み込む相対`image_paths`はJSONの親が基準です。明示画像が存在しない場合やgroup ID重複はエラーです。

## GPUなしで再集計する

```powershell
python -m dq_profile.dataset_diagnostics `
  --input-dir="D:\path\to\run\data_diagnostics" `
  --group-map="D:\datasets\dq-character-groups.json"
```

`inventory.jsonl`、`reference_probes.jsonl`、`quant_probes.jsonl`、`evaluation_inputs.jsonl`とmanifestを保存します。CSVはsample前後・sample量子化・group前後・group量子化の4種類です。JSONにはbin別統計、欠測理由、評価入力、group所属もあります。HTML内だけのgroup変更はCSVファイルを書き換えません。設定を保存して上記で再集計するとCSVと参考区間も更新します。

追加診断を有効にして実行した新しい結果なら、上記の保存ファイルからGPUなしでHTMLを作り直し、同じ診断フォルダに置けます。`local`で保存した結果にはAの初期評価がないため、CPU再集計で`warmup`相当にはできません。

追加診断導入前、または`off`で実行した旧結果には必要な記録がそろいません。旧ログにも`image_key`や`source_group`の対応はありますが、追加診断用のraw MSE、同じ固定入力での初期MSE、元captionやsample文脈等が不足します。残っている勾配指標で部分的な量子化グラフを作れる可能性はありますが、現実装に旧形式の取り込み機能はありません。完全な追加レポートを得るには、追加診断を指定して再実行します。

重複行の値が違う場合は平均で消さずエラーにします。旧レポートの重み付きobjectiveをraw MSEに変換することはできないため、従来の`per_image.csv`だけからの再構成は行いません。

## 数値の意味

raw MSEは全latent領域で、mask・sample weight・Min-SNR等を適用する前の誤差です。従来のobjective lossも別に保存します。量子化loss差はcandidateと同じnoise 0/1のno-quantとの差です。no-quantの3番目のnoiseを混ぜません。

sample → 画像 → groupの順で画像等重みにします。改善率は平均pre/postから求め、画像別改善率を平均しません。勾配変形は既存ExactGradientの直接の差分normを使います。参照normやloss分母が1e-12以下なら比率をnullにし、gradient topology不一致も無効にします。欠測を0点として描画しません。

画像詳細の3 noise replicasの変化幅は95%区間ではありません。groupの参考区間はsourceブロックを2,000回再標本化し、同じdrawを前後・候補で共有します。4 source未満では区間を出しません。HTML内でfilter/bin/groupやタグの別名（aliases）を変えた場合、保存済みの別母集団の区間は表示せず、CPU再集計を案内します。

これらは単一の学習runに条件づけた観測です。画質、汎化性能、別seed学習の成功確率を示しません。追加チャンネルは常に`selector_input=false`で、既存の推薦処理は読みません。

## 検証用コマンド

```powershell
python -m pytest tests/test_dq_dataset_diagnostics.py
python -m pytest tests -k dq_profile
python -m tools.check_dq_profile_copy_drift
```

`tools/validate_dq_dataset_report.cjs`はPlaywright/Edgeで52画像のテストレポートを検査する開発用ツールです。生成データは架空と明示します。全SDXL・実データでの長時間parity確認は別途必要です。
