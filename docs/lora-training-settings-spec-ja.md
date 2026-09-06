# 学習設定の記録と診断レポート

状態：設定スナップショットと表示を実装。ユーザー実行のSDXL・1エポック学習で生成物を確認済み。

## 通常の使い方

学習batの変更は不要。`train_network.py` の共通 `NetworkTrainer` を使う学習（`sdxl_train_network.py` を含む）では、自動的に出力ディレクトリへ設定を保存する。別の学習ループを持つtrainerは対象外。

```text
output_dir/
  従来のチェックポイント・ログ
  run_records/
    <UUID>/
      manifest.json
      inputs/
        requested_args.json
        resolved_config.json
```

毎回UUIDをOS由来の乱数で発行する。同じoutput_nameで再実行・resumeしても上書きしない。学習用のPython/NumPy/PyTorch乱数は追加消費しない。ファイルを残すのは代表プロセスのみ。

`manifest.json` の `kind=training_settings`、`schema_version=1` が今回の形式。`run_id`、output_name、既存のsession ID・開始時刻、設定ファイルの相対パスを持つ。将来のrun記録全体の仕様を実装済みとするものではない。

## 保存する内容と時点

| ファイル | 内容・保存時点 |
|---|---|
| requested_args.json | CLI・TOML・既定値の統合後、trainer内の調整前の引数をコピー。accelerator初期化後、モデル読込前に保存 |
| resolved_config.json | trainer内で調整された引数、既に解決済みのDQ・平均化・勾配設定、実際のoptimizer group、データセット等の記録を学習ループ前に保存 |

requested_argsはコマンドライン原文ではなく、既定値を含むNamespaceのコピー。どの項目をユーザーが明示指定したかまでは記録しない。

resolved_config内の `args` は調整後の引数であり、すべての設定を実効値へ展開したものではない。例えばschedulerの割合指定は引数の表現を保持する。追加で確定できた値を `runtime` に分離する。

- `optimizer_groups_created`：optimizer作成直後、scheduler適用・resume復元前。TE1/TE2など、実際に作られたgroupのLRとオプション。
- `optimizer_groups_at_start`：scheduler初期化・resume復元後、学習ループ前のgroup。warmupによりLRが0の場合もそのまま記録。
- `runtime`：実際のoptimizer名、学習対象TE、process数、予定epoch数、DQ auto band/閾値・初期mul・warmup、平均化モード、勾配設定など。
- `metadata`：学習処理が既に組み立てたモデル・データセット情報。チェックポイントを後から読んで補完するものではない。

学習中のLR変更・DQ auto切替の履歴や、画質評価は含まない。画像・captionの内容hash、依存環境の完全な一覧、未コミット差分も今回の対象外。データセット構成が一致していても、入力の内容が同一とは証明できない。

パラメータ・optimizer stateは保存しない。tensorをCPUに移したり値を取り出したりしない。JSONで表現できない値は `unrecorded` として明示する。APIキー等の既知の認証情報は伏せる。

ファイルは一時ファイルから置換する。保存失敗は警告を出して学習を継続する。`settings_status=requested` は引数のみ、`resolved` は初期化後の設定保存済みを意味し、**学習の成功・完走を意味しない**。accelerator初期化より前に失敗した学習には記録が残らない。

## 診断レポートでの取得ルール

レポート生成コマンドは従来どおり。`--skip_lora_analysis` 時も設定は読み取る。設定取得は標準ライブラリのみで動作し、safetensorsはヘッダーだけ読む。

1. `input_dir/run_records/*/manifest.json` を探す。
2. チェックポイントにsession IDと開始時刻があれば、その両方に一致する記録を選ぶ。同名の新しい記録を勝手に採用しない。
3. 対象の設定記録がある場合は、その記録だけを学習設定の出典にする。
4. 対象の記録がない場合だけ、チェックポイントのメタデータから取得する。
5. 両方なければ「記録なし」。従来のグラフ・診断処理は継続する。

記録ファイルが一部だけでも、欠けた項目をチェックポイントのメタデータで穴埋めしない。壊れたファイル・未対応version・run ID不一致はエラー表示とし、自動切替しない。読めないmanifestがあり対象を判定できない場合も、無条件のメタデータ切替を避ける。

checkpoint側のepoch/step/hash等は設定とは別枠。session照合は既存メタデータ上の対応確認であり、チェックポイントの内容hashによる同一性検証ではない。チェックポイントのメタデータやtensorは今回変更しない。

`--no_metadata`、checkpoint欠落等でsession情報がない場合、同名の記録が1つなら「対応未検証」として表示する。複数あれば未確定とし、日時で選ばない。移動済みの記録等は明示指定できる。

```bat
python tools\make_lora_diagnostic_report.py --base_name RUN_NAME --input_dir ..\lora_output --training_settings ..\lora_output\run_records\RUN_UUID\manifest.json
```

明示指定しても、読めるcheckpointのsession情報と不一致なら採用しない。

## 表示・過去互換性

診断JSONの `training_settings` に値・取得元・状態を追加する。グラフと診断値の計算は変更しない。

- 単独HTML：「学習設定」を折りたたみ表示。主要項目、optimizer作成時のLR、全引数・確定値・取得元を確認できる。
- 比較HTML：「学習設定の比較」を折りたたみ表示。差分と一部未記録の項目を既定で表示し、全項目に切替可能。JSON読込だけで動き、保存した比較HTMLにも設定を保持する。
- 旧JSON：グラフは従来どおり表示し、設定は記録なし。設定を追加するにはレポート生成を再実行する。旧JSONを一括補完するCLIは未実装。
- `None`（指定なし）・`false`・数値0・未記録を区別する。未記録をOFFや現行の既定値とみなさない。

過去のTLS2の例では、共通TE LR=0.0002は記録されているがTE1/TE2個別の指定はない。メタデータの共通TE LRをTE1の実効LRとみなさない。DQ presetや平均化の全設定も復元できない。新しい記録では引数と実際のoptimizer groupを分けて確認できる。

## 検証と次の区切り

標準ライブラリのテストで保存・読み込み・照合・欠損・破損・型・情報源の分離・乱数と引数の非変更を確認。既存の表示テスト、比較HTMLのブラウザテスト、過去235件のメタデータ読み込みと通常CLIも確認した。さらにユーザー実行のSDXL・1エポック（290 step）について、設定3ファイル、最終LoRA・final_rawとのsession照合、実ログのTE1/TE2/UNet学習率、生成済み診断JSON・HTMLへの反映を確認した。複数GPUやresume等を網羅した実行検証ではない。

今回で「設定保存 → 診断JSON → 単独表示・比較表示」まで完結する。attempt/実更新/skip台帳、checkpoint artifact履歴、生成条件・画像評価との連携は、このブランチを基点とする別ブランチで実装するのが適切。将来はrun_idを引き継げるが、今回はそれらのファイルや更新フックを追加しない。
