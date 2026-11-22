# 独自機能の回帰テスト

リポジトリのルートで実行します。学習の設定、DQ、Guardian、平均重み、ログ、診断・画像比較・研究台帳を合成入力で確認できます。個人のデータセットや学習済みのrunを用意する必要はありません。

```powershell
.\venv\Scripts\python.exe -m pytest tests/migration
```

CUDAやQtなど、実行環境に依存するケースがあります。実行結果の `skipped` は個別に確認してください。CSVの列順、旧項目の保持・復元・廃止を照合する期待値は [csv-contract.json](fixtures/csv-contract.json) にあります。採点用サンプルの選び方は [promote-bank.json](fixtures/promote-bank.json) の合成ケースで照合します。

## 診断比較画面のブラウザ試験

Node.jsからPlaywrightとChromiumを使用します。別途インストールしたブラウザを使う場合は `LORA_VIEWER_BROWSER` に実行ファイルのパスを指定してください。レポート2件の読込、条件別の表示、HTMLの保存と再読込を試験します。

```powershell
node tests/migration/test_diagnostic_viewer.cjs
```

## 任意の手動試験

| 試験 | 入力と確認内容 |
|---|---|
| [LoRAの更新比較](../local/migration_lora_reference.py) | 合成した小さなLoRAを3回更新。CPUまたはCUDAで勾配・optimizer状態・乱数を記録し、同じ機器・環境で得た基準値と比較します。 |
| [重みの読戻し](check_lora_load.py) | 手元のSDXLモデルと対応するLoRA重みを指定し、CPUでパラメータの形状・値・倍率を照合します。結果の書出し先には新しいファイル名を指定します。 |
| [GuardianとAMPの更新順](guardian_step_probe.py) | SDXLの通常の学習引数で、Guardian棄却とAMP overflowを含む4試行を実行します。手元のモデル・データセットとfp16のGradScalerが必要です。入力に応じて勾配が変わるため、試験で期待する更新・棄却の並びも確認してください。 |
| [終了後診断と手動診断の一致](verify_diagnostic_run.py) | `--run_dir` に、重みのepoch推移を生成できる完走runを指定します。診断レポートを2経路で実際に再生成し、生成日時・生成IDを除いた分析値を比べます。 |

たとえば、CPU用のLoRA更新基準値を取得して照合する場合は、比較元・比較先の各リポジトリのルートで対応するコマンドを実行します。基準ファイルは両者から見える同じパスを指定してください。

```powershell
$reference = Join-Path $env:TEMP 'lora-reference-cpu.safetensors'
.\venv\Scripts\python.exe tests/local/migration_lora_reference.py record --device cpu --reference $reference
.\venv\Scripts\python.exe tests/local/migration_lora_reference.py compare --device cpu --reference $reference
```
