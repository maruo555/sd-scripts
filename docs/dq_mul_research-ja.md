# 固定mul配分の研究用診断

`sdxl_dq_mul_research.py` は、事前に承認したローカルの実験契約を読み、
共通のwarmup境界で量子化対象と固定mul配分を比較する研究用入口です。
通常の学習CLIや `python -m dq_profile` のcanonical条件は変更しません。
この入口は短期学習・本学習・画像生成を実行しません。

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

優先順位は `group_overrides > components > base_mul` です。
UNet群は `unet.attn1`、`unet.attn2`、`unet.ff`、
`unet.other_projection` の排他的な4群です。未知の成分・群・モジュール、
重複指定、対象のないoverride、非有限または0以下のmulは拒否します。
OFFをmul=0で表現しません。

実際のモジュール名へ展開した設定を保存し、passの共通setterの後で適用します。
同じ測定cell内で展開済み割当が一致する別名候補は、量子化repeatごとに
一度だけF/Bを実行します。state・画像・noise・timestep・dropoutが異なるcellへは
このcacheを持ち越しません。別名と再利用元を記録し、F/B予算には実行数を使います。
snapshotはモジュールごとのON/OFFとmulも復元します。この研究runtimeでは
学習resumeやavg promoteへのpolicy導入は行っていません。

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

現在の研究スケジュールは、mul 2.70 / 3.15 / 3.75のscope比較、UNet/TEの
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
