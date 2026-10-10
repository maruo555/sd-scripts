# 標準診断と場所別mul CLIの検証

2026-10-10。対象は`canonical-v2`と、固定bits・RMSの場所別mulを直接指定する学習CLIです。
公開済み`2c2b7870e9b6bdebd3dc7ae31ff92fe149616881`へのレビューを受け、
本書と回帰テストを追加した修正版を検証しています。修正版のcommitは
`git log -1 --format=%H -- docs/dq_standard_validation-ja.md`で確認できます。

| 対象 | 検証方法 | 確認できた範囲 |
|---|---|---|
| 通常学習CLI | CPUの小型LoRAでforward/backward | policy優先順位、既存uniformとloss・勾配・乱数の一致、`scope=unet`でのTE指定、warmup/setter後の配分保持 |
| 保存・再開 | CPUで保存済みpolicy記録と照合 | 同じ宣言・配分は受理。省略・変更や異なる宣言への切替は拒否。JSONファイルの移動だけなら受理 |
| 標準診断 | 合成入力でOFF/ONの対応付きforward/backward | 5点と追加配分の重複処理、部位の加算性、比較前提の不一致検出、測定中に重み・optimizerを更新しないこと |
| 統計・公開処理 | 独立Body解析から公開HTMLを生成 | Body選出との一致、最終manifestのパス・SHA-256整合、解析stageの元記録を保持 |
| 共有JSON | MSEあり・なし・一部欠測のfixture | MSEとobjective lossの区別、欠測理由、全条件で同一の指標種別、非有限値の拒否 |
| 画像詳細 | 合成画像・caption・複数subset | 元画像を失った後のサムネイル再利用、captionの文字としての表示、複数文脈の併記、共有JSONに実パス・captionを含めないこと |
| HTML | CPUのオフラインブラウザ | 7部位×OFF/ONの描画値、条件切替、画像詳細、狭い画面幅。GPU計測の代用ではない |
| 実SDXL | **新標準の一貫したGPU受入は未実施** | 過去の研究用GPU実験を、この修正版の受入として扱わない |

汎用CPUテストは`tests/`でGit管理します。実データや個人パスを使う実験スクリプト、
GPU実行記録、ブラウザのスクリーンショットはGit管理外に保存します。

## CPUテストの再実行

プロジェクトの依存パッケージを導入済みの環境で、リポジトリ直下から実行します。
`--basetemp`には他の用途で使っていない一時フォルダを指定してください。

```powershell
$env:CUDA_VISIBLE_DEVICES=''
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
python -B -m pytest tests/test_dq_mul_policy_training.py tests/test_dq_mul_direct_cli.py tests/test_dq_spatial_standard.py tests/test_dq_spatial_report_contract.py tests/test_dq_dataset_report.py tests/test_dq_dataset_integration.py tests/test_dq_profile_production.py tests/test_dq_profile_copy_drift.py --basetemp=.tmp/dq_standard_cpu_tests -q
```

追加テストは実データもpretrained modelも読み込みません。CPUで実行した回数を
実データの診断・学習回数には数えません。テスト関数内の小型モデルと値は合成fixtureです。

## 次に必要なGPU受入

実行するcommitと入力を固定し、実行範囲・上限を確認してから行います。

1. 最低画像数・source条件を満たす実データで、新標準をsnapshot／prefix／boundaryから最終HTMLまで一度通す。dropout ONも同じ候補集合で確認する。
2. 実効mul、量子化開始境界、no-quantの再測定、候補・repeatの対応、重みの無更新、部位合算、Body選出、計画と実測F/B数、指標種別、manifestを照合する。
3. 通常学習の直接CLIでwarmup境界を越える小規模smokeを行い、対応するモードで保存・再開を確認する。既存の平均化モード等のresume制約は維持する。

比較検査が止まった場合は原因と契約を確認し、許容差を黙って広げて通過扱いにはしません。
ここを通過してから長時間の本学習へ進む手順です。研究コードの大規模統合・削除や、
乱数の識別方法・統計式の変更は、この接続部の修正とは分けて扱います。
