# 独自改造ファイル一覧

本家 sd-scripts 0.13.0（`8c088eb`）を基準とした**初回移植の公開対象**を記録する。表に載せたのは本家から変更・追加したファイル。各リンクからリポジトリ内の実物を開ける。移植元と設計判断は[選択移植の記録](migration/MIGRATION_PLAN.md)にまとめた。

| 本家からの差分 | 初回の構成 |
|---|---:|
| 変更した本家ソース／日本語README | 12件／1件 |
| 追加した実行用ソース・画面 | 68件 |
| 追加したテスト・検証コード | 55件 |
| 追加した文書（本一覧・テスト手順を含む） | 7件 |
| 追加したテスト用固定データ・依存環境・ライセンス | 6件 |
| **合計** | **149件** |

- [変更した本家ファイル](#upstream-changes)
- [追加した実行用ソース・画面](#runtime-files)
- [追加したテスト・検証コード](#validation-files)
- [追加した文書](#user-docs)
- [固定データと依存環境](#supporting-files)

<a id="upstream-changes"></a>

## 1. 変更した本家ファイル

| ファイル | 役割 |
|---|---|
| [train_network.py](../train_network.py) | 学習ループに勾配更新の採否、観測、epoch末の保存・平均、run controlとの接続点を追加。 |
| [sdxl_train_network.py](../sdxl_train_network.py) | SDXLで独自CLIを解決し、TE設定・学習ループ・観測を適用。 |
| [networks/lora.py](../networks/lora.py) | LoRAの対象層とLR、重みの読込・生成、DQのforwardへの接続。 |
| [sdxl_gen_img.py](../sdxl_gen_img.py) | 生成時のTE1・TE2・UNet別強度、LBW、動的プロンプトを接続。 |
| [library/args.py](../library/args.py) | 独自optimizerの引数案内と、設定ファイルを読み込んだ後の引数の再解析。 |
| [library/config_util.py](../library/config_util.py) | dataset設定にsubsetのgroupを追加し、倍率・バッチ等の実効値を検査。 |
| [library/subset.py](../library/subset.py) | 各形式のsubsetにgroupを保持。 |
| [library/dataset.py](../library/dataset.py) | promoteの採点で必要なときにバッチへclass tokensを渡す。 |
| [library/controlnet_dataset.py](../library/controlnet_dataset.py) | ControlNetのsubsetから内部のDreamBoothデータセットへgroupを引き継ぐ。 |
| [library/optimizer.py](../library/optimizer.py) | AdamW8bitFastとAdamWBnbの生成を本家のoptimizer選択に接続。 |
| [library/sdxl_original_unet.py](../library/sdxl_original_unet.py) | strict時のnorm・attentionの演算精度と、拡散時刻埋め込みのfreq shiftを適用。 |
| [library/logging_util.py](../library/logging_util.py) | 学習ループが計測した初期化時間を表示し、最初の学習バッチで進捗バーの速度計測をリセットする。 |
| [README-ja.md](../README-ja.md) | 本家の日本語READMEから、独自機能の利用ガイドとファイル一覧へ案内する。 |

<a id="runtime-files"></a>

## 2. 追加した実行用ソース・画面

### 実行用ソース・画面の内訳

学習・生成・診断・研究台帳で実行または読み込む68件。Python 64件、Windowsの起動バッチ1件、画面・テンプレートのHTML 3件。

### 学習設定・更新の制御・LoRA平均（11件）

| ファイル | 役割 |
|---|---|
| [library/maruo_training_config.py](../library/maruo_training_config.py) | 独自CLIとTOMLの既定値・優先順・組合せを解決する。 |
| [library/sdxl_training_options.py](../library/sdxl_training_options.py) | strict有効時、fp16入力のLayerNorm・空間GroupNorm・attentionのsoftmaxをfp32で計算し、元の型に戻す。 |
| [library/te_training.py](../library/te_training.py) | TE1・TE2の各MLPからfc1・fc2を選び、旧LoRAを初期重みとして読み込む際にキーと倍率を照合する。 |
| [library/adamw8bit_fast.py](../library/adamw8bit_fast.py) | bitsandbytesの更新処理の同期をまとめ、標準経路への切替も扱う。 |
| [library/adamw_bnb.py](../library/adamw_bnb.py) | 高い8bit切替閾値で使用するAdamWとその状態精度を定義する。 |
| [library/grad_norm_guardian.py](../library/grad_norm_guardian.py) | 勾配ノルムを観測し、その試行でoptimizerを更新するか決める。 |
| [library/avg_ckpt_util.py](../library/avg_ckpt_util.py) | 重みをCPUに複製し、平均に使う窓を保持する。 |
| [library/promote_bank.py](../library/promote_bank.py) | raw重みと平均重みの採点に使うサンプルを選択・保持する。 |
| [library/avg_training.py](../library/avg_training.py) | epoch末の候補の採点・昇格と、指定時のoptimizer状態リセットを行う。 |
| [library/group_loss.py](../library/group_loss.py) | 実際のsubsetに対応するグループ別lossを観測する。 |
| [library/lora_rank_stats.py](../library/lora_rank_stats.py) | 更新後のLoRA重みから実効rankを測る。 |

### DQの量子化・倍率・Triton（6件）

| ファイル | 役割 |
|---|---|
| [library/rounding_util.py](../library/rounding_util.py) | Torch経路で量子化スケールと丸めを計算する。 |
| [library/dq_mul_policy.py](../library/dq_mul_policy.py) | 固定の共通倍率・場所別倍率を割り当て、設定から復元する。 |
| [library/dq_training.py](../library/dq_training.py) | 学習試行ごとの量子化適用範囲・状態・観測を管理する。 |
| [library/dq_auto.py](../library/dq_auto.py) | 観測したclip率とEMAから自動倍率を更新する。 |
| [library/dq_triton.py](../library/dq_triton.py) | 乱数の整合を保ちながらTriton経路とfallbackを切り替える。 |
| [library/triton_quant.py](../library/triton_quant.py) | 量子化と統計収集に使うTritonカーネル。 |

### run、重み、ログの保存（8件）

| ファイル | 役割 |
|---|---|
| [library/training_run.py](../library/training_run.py) | runフォルダを確保し、run ID・索引・状態を管理する。 |
| [library/training_settings.py](../library/training_settings.py) | 要求・実効設定とGitリビジョン・依存環境の照合情報をrunに保存。 |
| [library/training_run_weights.py](../library/training_run_weights.py) | 重みを保存し、役割・保存時点・世代整理の結果を索引に登録する。 |
| [library/lora_run_schema.py](../library/lora_run_schema.py) | 6種類のCSVの列、型、変換規則を定義する。 |
| [library/training_diagnostics.py](../library/training_diagnostics.py) | 各観測をrunのCSVに書き出す。 |
| [library/training_run_output.py](../library/training_run_output.py) | 学習ループからrunの作成・観測・出力確定を呼び出す。 |
| [library/training_run_control.py](../library/training_run_control.py) | run controlのコマンド処理とサンプル生成の前後で、Pythonの乱数状態を保存・復元する。 |
| [library/training_run_report.py](../library/training_run_report.py) | 学習終了後、指定されたrunの診断を別プロセスで起動する。 |

### 画像生成・条件比較・blind評価（12件）

| ファイル | 役割 |
|---|---|
| [library/generation_dynamic_prompt.py](../library/generation_dynamic_prompt.py) | 直接生成と画像比較で動的プロンプトを同じ規則で展開する。 |
| [library/generation_lora_strengths.py](../library/generation_lora_strengths.py) | TE1・TE2・UNet別の強度指定を解釈して表現を揃える。 |
| [library/generation_lbw.py](../library/generation_lbw.py) | SDXLブロック別強度（LBW）のプリセットと入力値を検査する。 |
| [networks/lora_lbw.py](../networks/lora_lbw.py) | 画像生成用のLoRAにブロック別倍率を適用する。 |
| [sdxl_lora_report_cui.py](../sdxl_lora_report_cui.py) | コマンドラインで画像比較を開始する。 |
| [sdxl_lora_report_gui.py](../sdxl_lora_report_gui.py) | 画像比較の条件を設定し、Qt画面でキューの実行順・取消・再試行を管理する。 |
| [sdxl_lora_report_worker.py](../sdxl_lora_report_worker.py) | 比較ジョブの生成を子プロセスで実行する。 |
| [tools/lora_report_plan.py](../tools/lora_report_plan.py) | 比較条件・生成ジョブ・対象画像の組合せを確定する。 |
| [tools/lora_report_queue.py](../tools/lora_report_queue.py) | 比較ジョブのプロセスと結果を照合し、生成プロセスの起動・取消を扱う。 |
| [tools/lora_report_result.py](../tools/lora_report_result.py) | 比較結果のメタデータとPNGの対応を検証する。 |
| [tools/lora_report_views.py](../tools/lora_report_views.py) | 通常比較・blind比較のHTMLを描画する。 |
| [lora_report_templates/blind_report_detailed.html](../lora_report_templates/blind_report_detailed.html) | blind採点と詳細CSV出力に使う画面テンプレート。 |

### 診断・density・tokenizer（16件）

| ファイル | 役割 |
|---|---|
| [tools/lora_run_selection.py](../tools/lora_run_selection.py) | 指定されたrunフォルダ、または出力ルートと名前から診断対象を選ぶ。 |
| [tools/lora_run_reader.py](../tools/lora_run_reader.py) | runの索引・設定・CSV・重みを読んで、同じrunのものか照合する。 |
| [tools/lora_run_metrics.py](../tools/lora_run_metrics.py) | CSV整理で移した68項目を設定と観測値から復元する。 |
| [tools/make_lora_diagnostic_report.py](../tools/make_lora_diagnostic_report.py) | runを選び、診断を生成するコマンドの入口。 |
| [tools/lora_diagnostic_analysis.py](../tools/lora_diagnostic_analysis.py) | 6種類のCSVの診断集計をまとめる。 |
| [tools/lora_diagnostic_dq.py](../tools/lora_diagnostic_dq.py) | scopeと倍率を考慮し、通常DQ・自動DQの観測を集計する。 |
| [tools/lora_diagnostic_rank.py](../tools/lora_diagnostic_rank.py) | rankの集計とヒートマップ用の値を作る。 |
| [tools/lora_diagnostic_numbers.py](../tools/lora_diagnostic_numbers.py) | 診断で共用する数値の変換、移動平均、欠測処理。 |
| [tools/lora_diagnostic_charts.py](../tools/lora_diagnostic_charts.py) | 集計値をグラフ用の系列・参考指標に変換する。 |
| [tools/lora_diagnostic_display.py](../tools/lora_diagnostic_display.py) | 診断結果をHTMLの各表示欄に組み立てる。 |
| [tools/lora_report_publication.py](../tools/lora_report_publication.py) | 生成した診断JSON・HTMLの対応を検証して公開する。 |
| [tools/lora_report_assets/diagnostic.html](../tools/lora_report_assets/diagnostic.html) | 診断レポートの表示テンプレート。 |
| [tools/lora_comparison.html](../tools/lora_comparison.html) | 複数のrunの診断をオフラインで比較する画面。 |
| [tools/analyze_lora_density.py](../tools/analyze_lora_density.py) | LoRAの層・ブロック別densityを解析するコマンド。 |
| [tools/lora_density_display.py](../tools/lora_density_display.py) | densityの解析結果をHTMLに表示する。 |
| [tools/sdxl_tokenize.py](../tools/sdxl_tokenize.py) | 両TEでトークン分割を表示し、指定条件を満たす候補を探索する。 |

### 研究台帳（15件）

| ファイル | 役割 |
|---|---|
| [tools/lora_ledger.py](../tools/lora_ledger.py) | 研究台帳を操作するCLI。 |
| [lora_ledger_gui.py](../lora_ledger_gui.py) | 研究台帳のQt画面を起動する。 |
| [lora_ledger_gui.bat](../lora_ledger_gui.bat) | Windowsで研究台帳GUIを起動するバッチ。 |
| [tools/lora_ledger_core/__init__.py](../tools/lora_ledger_core/__init__.py) | 研究台帳のPythonパッケージ定義。 |
| [tools/lora_ledger_core/output_root.py](../tools/lora_ledger_core/output_root.py) | 学習出力ルートと、そこで使用する台帳を結び付ける。 |
| [tools/lora_ledger_core/scanner.py](../tools/lora_ledger_core/scanner.py) | runを発見し、重み・ログ・診断との対応を調べて登録する。 |
| [tools/lora_ledger_core/storage.py](../tools/lora_ledger_core/storage.py) | 台帳のJSONと変更履歴を整合させて保存する。 |
| [tools/lora_ledger_core/features.py](../tools/lora_ledger_core/features.py) | 各重みの保存時点に対応する研究指標を算出する。 |
| [tools/lora_ledger_core/reviews.py](../tools/lora_ledger_core/reviews.py) | 比較結果を取り込み、条件間の評価を記録する。 |
| [tools/lora_ledger_core/notes.py](../tools/lora_ledger_core/notes.py) | runや重みごとの手動メモを保存する。 |
| [tools/lora_ledger_core/exporting.py](../tools/lora_ledger_core/exporting.py) | 選択した指標・評価をCSV等の一式として書き出す。 |
| [tools/lora_ledger_core/history_transfer.py](../tools/lora_ledger_core/history_transfer.py) | 旧台帳の履歴を引き継ぎ、保存済みexportを参照する。 |
| [tools/lora_ledger_core/presentation.py](../tools/lora_ledger_core/presentation.py) | 記録から表示用のdataset名・時刻を組み立てる。 |
| [tools/lora_ledger_core/gui.py](../tools/lora_ledger_core/gui.py) | 研究台帳のQtメイン画面。 |
| [tools/lora_ledger_core/dialogs.py](../tools/lora_ledger_core/dialogs.py) | 台帳のrun登録・画像比較の取込に使うダイアログと、取り消し可能なバックグラウンド処理を提供する。 |

<a id="validation-files"></a>

## 3. 追加したテスト・検証コード

追加した検証コード55件の内訳は、pytest用49件、Node.jsで実行するブラウザ試験1件、任意の手動試験4件、optimizer比較の共通コード1件。別途、試験手順のMarkdown 1件と固定データ2件を収録した。本家0.13.0の既存テスト関連ファイル56件はそのまま利用する。

自動テストは合成入力や一時ファイルを使う。手動試験では、重みの読戻しにはSDXLモデルとLoRA、Guardianの更新順の確認にはモデルとデータセット、診断の再生成には完走したrunを実行者が指定する。小さなLoRAの更新比較には合成テンソルを使う。

### 学習・optimizer・DQの回帰テスト（21件）

| ファイル | 役割 |
|---|---|
| [tests/migration/test_config.py](../tests/migration/test_config.py) | 独自CLI・TOMLの優先順位、既定値、固定倍率・自動倍率等の設定と不正入力を検査。 |
| [tests/migration/test_dataset_options.py](../tests/migration/test_dataset_options.py) | datasetのgroup継承、実効倍率、batchや対象データの制約。 |
| [tests/migration/test_te_training.py](../tests/migration/test_te_training.py) | TE1/TE2のFC選択・LR・既存重み・dropout。 |
| [tests/migration/test_lora_factory.py](../tests/migration/test_lora_factory.py) | LoRA生成時・学習時の重み読込とインスタンス間の独立。 |
| [tests/migration/test_sdxl_precision.py](../tests/migration/test_sdxl_precision.py) | strict時のnorm・attentionの演算精度と勾配。実CUDAの経路も含む。 |
| [tests/migration/test_sdxl_timestep.py](../tests/migration/test_sdxl_timestep.py) | freq shiftが拡散時刻のみに効くこととdeep shrinkの挙動。 |
| [tests/migration/test_delta_quant.py](../tests/migration/test_delta_quant.py) | DQの丸め、channel RMS、統計、場所別mulの割当。 |
| [tests/migration/test_dq_auto.py](../tests/migration/test_dq_auto.py) | 自動mulの帯判定・EMA・warmup・次のforwardへの反映。 |
| [tests/migration/test_dq_training.py](../tests/migration/test_dq_training.py) | 量子化対象、dropout、Guardian棄却・採点時の統計と乱数。 |
| [tests/migration/test_dq_triton.py](../tests/migration/test_dq_triton.py) | Triton経路とfallback時の丸め・乱数・統計。 |
| [tests/migration/test_grad_norm_guardian.py](../tests/migration/test_grad_norm_guardian.py) | Guardianの閾値、非有限値、勾配ノルムの計算。 |
| [tests/migration/test_guardian_optimizer.py](../tests/migration/test_guardian_optimizer.py) | Guardian棄却とAMP overflowを区別し、更新順を検査。 |
| [tests/migration/test_adamw8bit_fast.py](../tests/migration/test_adamw8bit_fast.py) | Fastと標準optimizerの各更新・全状態・fallbackの一致。 |
| [tests/migration/test_adamw_bnb.py](../tests/migration/test_adamw_bnb.py) | AdamWBnbの8bit切替閾値、32bit状態、棄却とreset。 |
| [tests/migration/test_avg_window.py](../tests/migration/test_avg_window.py) | 平均窓への追加・破棄・元重みとの独立。 |
| [tests/migration/test_promote_bank.py](../tests/migration/test_promote_bank.py) | promote採点用のサンプル選択・両TE・乱数状態。 |
| [tests/migration/test_epoch_averaging.py](../tests/migration/test_epoch_averaging.py) | epoch末のraw・平均重みの保存順、promote/reset、失敗処理。 |
| [tests/migration/test_epoch_boundary_observations.py](../tests/migration/test_epoch_boundary_observations.py) | epoch最終試行での観測・保存・promoteの順序。 |
| [tests/migration/test_rank_group_observers.py](../tests/migration/test_rank_group_observers.py) | 更新後のrankと重み付きgroup lossの観測。 |
| [tests/migration/test_console_progress.py](../tests/migration/test_console_progress.py) | 初期化時間を学習速度から除く進捗表示。 |
| [tests/migration/test_sample_python_rng.py](../tests/migration/test_sample_python_rng.py) | 学習中のサンプル生成前後でPythonの乱数状態を保つ。 |

### run・重み・CSVの回帰テスト（12件）

| ファイル | 役割 |
|---|---|
| [tests/migration/test_training_run.py](../tests/migration/test_training_run.py) | runフォルダの同時作成、索引の更新、失敗時の保護。 |
| [tests/migration/test_training_run_weights.py](../tests/migration/test_training_run_weights.py) | 重みの登録・世代整理・保存名、sampleの保存先。 |
| [tests/migration/test_training_settings_provenance.py](../tests/migration/test_training_settings_provenance.py) | clone・worktreeで設定に使うGitリビジョンを取得。 |
| [tests/migration/test_run_control_output.py](../tests/migration/test_run_control_output.py) | run controlのメタデータと本家の共有sample出力先。 |
| [tests/migration/test_run_output.py](../tests/migration/test_run_output.py) | 学習から設定とログへの接続、棄却時・無効時の出力。 |
| [tests/migration/test_lora_run_csv.py](../tests/migration/test_lora_run_csv.py) | 6 CSVの型・引用符・改行・不完全な行の処理。 |
| [tests/migration/test_csv_field_decisions.py](../tests/migration/test_csv_field_decisions.py) | 固定したCSVの列順・保持・廃止の期待値と、実装上の6ヘッダーを照合。 |
| [tests/migration/test_lora_run_metrics.py](../tests/migration/test_lora_run_metrics.py) | 復元対象68項目を固定データと照合し、実際の値・欠測を確認。 |
| [tests/migration/test_reader_csv_fingerprint.py](../tests/migration/test_reader_csv_fingerprint.py) | 複数行のCSVレコードで、完結した部分だけハッシュ化。 |
| [tests/migration/test_reader_weight_identity.py](../tests/migration/test_reader_weight_identity.py) | 重みとrun索引のUUID・役割・保存時点を照合。 |
| [tests/migration/test_run_discovery.py](../tests/migration/test_run_discovery.py) | 直下のrun検出、重複UUID・不正な索引・リンクの扱い。 |
| [tests/migration/test_run_selection.py](../tests/migration/test_run_selection.py) | 同名runの選択、名前省略時の検索範囲。 |

### 診断・生成比較・研究台帳の回帰テスト（17件）

| ファイル | 役割 |
|---|---|
| [tests/migration/test_diagnostic_metrics.py](../tests/migration/test_diagnostic_metrics.py) | 診断用の集計単位、rank・group・promoteの欠測処理。 |
| [tests/migration/test_diagnostic_publication.py](../tests/migration/test_diagnostic_publication.py) | 診断の途中失敗、正しいrunの自動選択・公開。 |
| [tests/migration/test_diagnostic_rank_contract.py](../tests/migration/test_diagnostic_rank_contract.py) | 場所別mul割当とrank正規化に必要な入力。 |
| [tests/migration/test_report_input_snapshot.py](../tests/migration/test_report_input_snapshot.py) | 診断中に設定や重みが差し替わった場合の検出。 |
| [tests/migration/test_run_density.py](../tests/migration/test_run_density.py) | 実際の保存epochと重みからdensityを一度だけ解析。 |
| [tests/migration/test_diagnostic_viewer.cjs](../tests/migration/test_diagnostic_viewer.cjs) | 比較ビューアで診断を開き、条件を分けてオフライン保存。 |
| [tests/migration/test_generation_lbw.py](../tests/migration/test_generation_lbw.py) | ブロック別強度（LBW）の適用・merge・復元。 |
| [tests/migration/test_generation_strengths.py](../tests/migration/test_generation_strengths.py) | TE1/TE2/UNet別強度の引数解釈と適用。 |
| [tests/migration/test_lora_report_plan.py](../tests/migration/test_lora_report_plan.py) | 比較ジョブの確定・CUIとworkerの接続・dry-run。 |
| [tests/migration/test_comparison_queue.py](../tests/migration/test_comparison_queue.py) | Qt GUIの比較処理をキャンセル・再試行した時の整合。 |
| [tests/migration/test_comparison_result.py](../tests/migration/test_comparison_result.py) | 比較metadataと生成PNGの強度・seed・LBWの対応。 |
| [tests/migration/test_comparison_browser.py](../tests/migration/test_comparison_browser.py) | Qt WebEngine上でblind採点し、CSVをダウンロード。 |
| [tests/migration/test_ledger_features.py](../tests/migration/test_ledger_features.py) | 重み保存時点に対応する6 CSVと診断から20指標を算出。 |
| [tests/migration/test_ledger_registration.py](../tests/migration/test_ledger_registration.py) | run登録、重複・移動・再登録と手動編集の保持。 |
| [tests/migration/test_ledger_history.py](../tests/migration/test_ledger_history.py) | 旧台帳のID・評価・exportの引継ぎと中断時の復旧。 |
| [tests/migration/test_ledger_export.py](../tests/migration/test_ledger_export.py) | 比較評価・履歴の書出し、Qtでの保存・再読込。 |
| [tests/migration/test_sdxl_tokenize.py](../tests/migration/test_sdxl_tokenize.py) | 両TEの分割・ID・候補順位を小語彙で検査。 |

### 手動試験・検証の補助と共通コード（5件）

| ファイル | 役割 |
|---|---|
| [tests/migration/bnb_step_oracle.py](../tests/migration/bnb_step_oracle.py) | 標準bitsandbytesと各optimizerの更新・状態を突合する共通コード。 |
| [tests/local/migration_lora_reference.py](../tests/local/migration_lora_reference.py) | 小さいLoRAで複数stepの値・勾配・更新・乱数を記録し本家と照合。 |
| [tests/migration/check_lora_load.py](../tests/migration/check_lora_load.py) | 実SDXLへ保存済みLoRAをCPUで読み戻し、全パラメータを検査。 |
| [tests/migration/guardian_step_probe.py](../tests/migration/guardian_step_probe.py) | 4試行でGuardian棄却とAMP overflowを意図的に発生させて比較。 |
| [tests/migration/verify_diagnostic_run.py](../tests/migration/verify_diagnostic_run.py) | 完走runの終了時診断と、手動再生成した診断を比較。 |

<a id="user-docs"></a>

## 4. 追加した文書

| ファイル | 役割 |
|---|---|
| [docs/maruo_cli_ja.md](maruo_cli_ja.md) | 独自CLIの既定値・適用条件・各機能への索引。 |
| [docs/maruo_training_features_ja.md](maruo_training_features_ja.md) | 学習・DQ・Guardian・promote・ログなどの機能別解説。 |
| [docs/maruo_tools_ja.md](maruo_tools_ja.md) | 診断・density・画像生成／比較・台帳・tokenizerの操作ガイド。 |
| [docs/maruo_features_ja.md](maruo_features_ja.md) | WindowsでのTritonの任意導入、量子化方式の使い分け、学習CLIに渡す設定例。 |
| [docs/migration/MIGRATION_PLAN.md](migration/MIGRATION_PLAN.md) | 移植元と本家の基準、選択移植時の設計判断・検証範囲を記録。 |
| [tests/migration/README.md](../tests/migration/README.md) | 個人のデータセットを必要としない回帰テストと、任意の手動試験の手順。 |
| [docs/maruo_file_inventory_ja.md](maruo_file_inventory_ja.md) | 本一覧。初回の公開構成と以後のファイルの変更を記録。 |

<a id="supporting-files"></a>

## 5. テスト用固定データ・依存環境・ライセンス

| ファイル | 役割 |
|---|---|
| [tests/migration/fixtures/csv-contract.json](../tests/migration/fixtures/csv-contract.json) | 6 CSVの列順と旧項目の保持・復元・廃止の期待値。自動テストから参照。 |
| [tests/migration/fixtures/promote-bank.json](../tests/migration/fixtures/promote-bank.json) | 旧版のpromote採点用サンプル選択の期待値。 |
| [docs/migration/baseline/windows-cu130.lock.txt](migration/baseline/windows-cu130.lock.txt) | 検証したWindows/CUDAの依存一覧。学習時にこのファイルのハッシュを記録。 |
| [requirements-gui.txt](../requirements-gui.txt) | 研究台帳・比較GUIなどの追加依存パッケージ。 |
| [requirements-triton-windows.txt](../requirements-triton-windows.txt) | Windows x64・PyTorch 2.9系でTritonを追加するための依存指定。 |
| [third_party/bitsandbytes-LICENSE.txt](../third_party/bitsandbytes-LICENSE.txt) | bitsandbytesに由来するoptimizer実装に同梱するライセンス。 |

## 更新方法

初回移植の比較基準を残し、対象ファイルの追加・削除・移動・役割変更と同じ変更で、この一覧、リンク、件数を更新する。パスはリポジトリのルートを起点に記載する。初回後に構成が変わったら日付と増減を追記する。

照合にはリポジトリのルートで次のコマンドを使う。本家を更新した場合は、比較先をその本家コミットに替え、新たに取り込んだ本家ファイルと独自ファイルを区別する。

```powershell
git diff --name-status --find-renames 8c088eb --
git ls-files --others --exclude-standard
```
