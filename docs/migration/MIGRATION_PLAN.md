# SDXL LoRA 選択移植の記録

2026年11月、旧 `maruo-main` のSDXL LoRA向け独自機能を、本家 sd-scripts 0.13.0 の学習・生成処理に合わせて選択的に移植した。各ファイルの役割は[独自改造ファイル一覧](../maruo_file_inventory_ja.md)、利用できる引数は[独自CLIリファレンス](../maruo_cli_ja.md)にある。

## ソースの基準

| 時点 | コミット | 用途 |
|---|---|---|
| 旧 `maruo-main` が本家から分岐した時点 | `a21b6a917e8ca2d0392f5861da2dddb510e389ad` | 本家に対する旧版の変更を調べた起点。 |
| 移植元 `maruo-main` | `898d1f77b19b424cd489f927ab915f6343a69701` | 独自機能の実効値、対象層、保存順、ツールの入出力を照合した版。 |
| 移植実装の本家基準 | `8c088eba9cd5d10a5d00dd58f0d83063fa9d57ca` | sd-scripts 0.13.0。学習ループ、run control、SDXLのモデル・optimizer・生成処理の接続先。 |

## 本家0.13.0で取り込んだ変更とライブラリの更新

旧版と新版の分岐点の間には、標準LoRAの学習にも使われるデータ処理・テキストエンコーダ・学習制御の更新がある。下表は2026-11-23に各Windows仮想環境から読み取った実測値。PyTorchやbitsandbytesを含め、表の全バージョンを本家の必須指定としているわけではない。本家の依存指定と検証環境の再現にはそれぞれの記録を参照する。[本家の旧依存指定](https://github.com/kohya-ss/sd-scripts/blob/a21b6a9/requirements.txt)・[0.13.0の依存指定](https://github.com/kohya-ss/sd-scripts/blob/8c088eb/requirements.txt)・[移植検証時のlock](baseline/windows-cu130.lock.txt)

| ライブラリ | 旧 `maruo-main` 環境 | 新 `maruo-main-2` 環境 | 関係する処理 |
|---|---|---|---|
| PyTorch | 2.9.1+cu130 | 2.9.1+cu130 | GPU演算・自動混合精度の基盤。両環境で同じビルドを使用。 |
| Accelerate | 0.30.0 | 1.15.0 | 学習ループの混合精度・勾配同期・state保存。 |
| Transformers | 4.44.0 | 5.17.0 | TE1・TE2とCLIP tokenizer。本家は文字正規化とTE1の層名の互換性を補った。 |
| Diffusers | 0.25.0 | 0.40.0 | VAEと拡散ノイズのscheduler。 |
| bitsandbytes | 0.48.2 | 0.50.2 | AdamW8bitFastが利用する更新カーネルとoptimizer状態。 |
| huggingface-hub | 0.24.5 | 1.32.0 | tokenizerなどの取得とローカルキャッシュ。 |
| safetensors | 0.4.2 | 0.8.0 | モデル・LoRAの重みの読み書き。 |
| NumPy / OpenCV / Pillow | 1.26.4 / 4.8.1.78 / 11.3.0 | 2.2.6 / 4.10.0.84 / 12.3.0 | 画像読み込み・bucketに合わせたリサイズと切り抜き。 |

- **共通コードの整理。** 本家PR #2372は121ファイルを変更し、その前後で `library/train_util.py` は6,888行から218行になった。dataset、optimizer、loss、保存、サンプル生成などをそれぞれのモジュールに分離した。両分岐点で標準 `networks/lora.py` を比較すると、本家側の差は保存時のハッシュ関数の参照先で、LoRAのforward・rank dropoutの式は共通だった。[本家PR #2372](https://github.com/kohya-ss/sd-scripts/pull/2372)
- **テキスト条件。** Transformers 5系のtokenizerには旧CLIPの文字正規化を補い、TE1はラッパーで旧LoRAの層名を保つ。SDXLが使うTE1のhidden state、TE2の最後から2層目のhidden state、EOS位置のpooled出力という構成は両端で共通。[tokenizerの互換処理](https://github.com/kohya-ss/sd-scripts/blob/8c088eb/library/clip_tokenizer.py)・[TE1の互換処理](https://github.com/kohya-ss/sd-scripts/blob/8c088eb/library/clip_text_model.py)
- **入力条件によって効く修正。** 新版の既定補間では、片方の辺が同寸でも他方が縮小ならAREAを使う。また、タグを並べ直す際に設定した区切り文字を尊重する。cached latentは旧版で学習dtypeに明示変換していたが、新版は入力dtypeを保持してからスケーリングとノイズ生成に進む。fp32で保存したディスクキャッシュなどでは差が出うる。[リサイズの修正](https://github.com/kohya-ss/sd-scripts/commit/b822b7e)・[タグ区切りの修正](https://github.com/kohya-ss/sd-scripts/commit/409ce9d)・[本家の新しい学習ループ](https://github.com/kohya-ss/sd-scripts/blob/8c088eb/train_network.py)

### 固定入力での比較（2026-11-23）

CPUでは13枚でbucket・crop・前処理後の画素と両TEのtoken IDが一致した。fp16の固定latent、seed 39、noise offset 0.15、adaptive noise scale 0.1で32回比較し、ノイズ、timestep、ノイズ付加後のlatent、固定予測値に対するL2も一致した。

GPUでは同じRTX 5080上で両環境を個別に起動し、同一チェックポイントと1組の画像・キャプションで比較した。学習前のVAEと両TEをeval・fp16で使い、TEはautocastを有効にした。入力画素（高さ832×幅576）、両TEの77トークン、VAEのサンプル用ノイズの一致を確認した。

| 出力 | 照合した値 | 最大絶対差 / 平均絶対差 |
|---|---|---|
| VAE | 平均・標準偏差・同一ノイズでのサンプル。それぞれ 1×4×104×72 | 各 0 / 0 |
| TE1 | 学習で使うhidden state、1×77×768 | 0 / 0 |
| TE2 | 学習で使うhidden state、1×77×1280、およびpooled出力、1×1280 | 各 0 / 0 |
| AdamW8bitFast | 同じfp32初期値・勾配で学習率3.5e-4の1更新。3テンソル、計14,592要素の更新量と更新後の重み | 各 0 / 0 |

optimizerは小さいテンソルのfp32状態と大きいテンソルの8bit状態を両方通した。モーメント、量子化テーブル、ブロック別スケールの数値も一致し、保存して比較したstepの値はいずれも1だった。これは固定入力のforwardとoptimizer初回更新の検証であり、学習中のdropout、UNetのbackward、多数step後の重みや画質の同等性は扱っていない。

## 移植時の判断

- **学習。** SDXL LoRAにTE別学習率とMLPのFC層選択、UNetのfp16演算と拡散時刻埋め込みのオプション、追加optimizerを接続した。学習を変える独自設定を無効にした場合は本家の計算経路を用いる。
- **量子化と更新順。** LoRAの出力差分に固定・自動倍率の量子化を適用し、場所別の倍率とTorch/Tritonの実行経路を分けた。Guardianの判定をoptimizer更新より前に置き、同期試行数と実際の更新数を区別する。promoteではepoch末の現重みと候補を同じ入力で採点する。[学習機能の解説](../maruo_training_features_ja.md)
- **出力。** 1回の学習につき1つのrunフォルダを作る。保存重み、設定、索引、最大6種類の観測CSVを同じrunに結び付ける。本家のrun controlから指定するサンプル生成には、本家の共有サンプル保存先を引き継いだ。
- **ツール。** 診断・density・研究台帳はrun索引を起点に設定・CSV・重みを照合する。研究台帳は学習出力ルートごとに1つ置く。生成・比較ではTE1・TE2・UNetの強度とブロック別強度を扱い、GUIとCUIで生成条件を共有する。[ツールの操作ガイド](../maruo_tools_ja.md)
- **互換性。** 独自設定を使用する学習では、旧LoRA重みを初期値にして新しいrunを開始できる。独自optimizerなどの途中状態からの再開は初回移植の対象外とした。過去の独自ログを読む場合は旧版のツールを使用する。

## 検証した範囲

移植時には、独自設定を無効にした本家との数値照合、機能ごとの更新・棄却・採点・保存順、Torch/Tritonの量子化と乱数の扱いを確認した。量子化なし・固定倍率・自動倍率の学習については、保存重みの読戻し、診断・台帳への登録、同一条件での生成画像の比較を実施した。過去の評価はそのときのモデル・データ・環境に対するものなので、別の学習での画質は新たに生成した結果で判断する。検証時のパッケージ構成は[環境lock](baseline/windows-cu130.lock.txt)に記録している。

## 共通学習ループの整理（2026-11-23）

[共通trainer](../../train_network.py) が更新・保存・run controlの順序を決める。[SDXLの入口](../../sdxl_train_network.py) は標準LoRAの学習1回ごとに [SdxlLoraTraining](../../library/sdxl_lora_training.py) を作る。このクラスが独自機能の状態を持ち、共通ループから渡された値を使って既存の各機能を呼ぶ。

| 現在の処理 | 共通ループに残すもの | SDXL側へ移すもの・再利用先 |
|---|---|---|
| 開始・データセット・出力準備 | 引数検証、dataset・Accelerator・networkの生成順、総step数の確定、main processの区別。保存間隔は本家の引数上で決めた後、保存用引数を得る。 | SDXLの入口で解決した設定を所有する。subset group・倍率の検査、promote用ラベル、run予約・設定スナップショット、初回バッチの速度計測、DQ・Guardian・平均の生成を、必要な実物が揃った段階で行う。実体は既存の `DQTraining`、`GradNormGuardian`、`EpochAveraging`、`RunOutput` を使う。 |
| forwardと採点用入力 | tokenize、text encoding、ノイズ・target作成、UNet、loss、backwardの計算式を1つに保つ。既存の `forward_backward` と `on_step_start` を再利用する。 | SDXLの学習forward直前にDQ試行を開く。通常forwardで実際に使ったnoisy latent・target・timestep・重み付きtoken等だけを必要な場合に渡す。共通側の計算箇所には引き渡し点だけを残し、`_promote_capture`、bankへの登録、group用timestepの保持はSDXL側で所有する。採点は `score_promote_bank` から同じtext条件とlossを再利用し、再抽選や観測への再登録を防ぐ。 |
| backward後の更新 | 更新可否を問い合わせ、許可時のall-reduce・clip・optimizer・scheduler、最後のzero_gradを本家と同じ位置で実行する。既定は常に許可する。同期試行数 `global_step` とAMPのoverflow処理は共通側が所有する。 | Guardian判定、勾配ログ、`skipped` の累計と表示を担当する。Guardianの棄却とAMPによる更新見送りは別の状態として扱う。ログから実際の更新数を逆算しない。 |
| 同期試行の終了 | 本家の重みノルム正規化を通り、同期時だけ進捗と `global_step` を進める。続くサンプル・step保存・run control・validationの位置を維持する。 | 進んだstepを受け、group loss → DQ統計確定・次回倍率決定 → DQ/rankの記録の順で呼ぶ。棄却時もDQ試行を閉じる。統計の採否は各観測器の既存仕様に従う。次のforwardまで倍率を遡及させない。 |
| epoch末・終了時 | 周期判定、raw重みとstateの保存・世代削除、サンプル、optimizerのtrain/eval切替、run control、停止分岐、最終重みの保存順。通常・手動とも同じ保存入口を使う。 | raw保存・サンプル・optimizerのtrain復帰の後に平均とpromoteを実行し、採用時のみ設定に従いmomentsをresetする。promote CSVとepochのflushをrun controlのepoch終了通知より前に行う。重みの索引・保存時点、最終raw別保存、run正常／失敗／停止、診断起動は既存のrun関連モジュールに委ねる。 |

**実装境界。** 共通側の新しい差し込み位置は、既存のメソッドで表せない「backward直後の更新可否」「同期試行確定後」「epochのraw保存とサンプル終了後」を中心にする。基本実装は許可または何もしない。保存は既存の `_save_model`・`_remove_model` の経路を使い、SDXL側が発行先と重み索引を担当する。共通側が渡すのはepoch、同期試行数、実バッチ位置、必要なモデル等、その場で確定した値に限る。非SDXLでは独自ログ、bank、乱数消費、GPU→CPU同期を増やさない。SDXL＋標準LoRAで独自学習を全OFFにした場合にも、現在のrun出力と本家のstate保存間隔の両方を維持する。

**今回変えない挙動。** Guardian棄却時はoptimizer・schedulerを見送る。別途 `--scale_weight_norms` を指定した場合は、その後に本家の重み正規化が動く。これも止める案は学習仕様の変更なので、責務を移す作業から切り離す。run controlの途中停止では通常のepoch末promoteに進まず、手動サンプル・定期サンプルは従来の共有出力先を使う。

### 整理後に確認した境界

- **共通trainerとSDXL。** [CPU統合試験](../../tests/migration/test_epoch_boundary_observations.py) は、小さなモデルで実際のtrainerを3 epoch実行する。2回目に大きな勾配を渡すと、共通trainerは更新し、Guardian付きSDXLは1回目と同じ重みを保存する。どちらも3回目は更新する。観測のみ・勾配蓄積2では6 microbatchから同期3回分のgroup lossが出る。停止コマンドのstep保存、例外時のrun終了状態も検証する。
- **epoch末。** 同じ試験で、SDXLのepoch末処理に入った時点で第1・第2 epochの定期重みがすでにディスクにあることを確認する。検証用にメモリ上で順序を逆転するとこの試験が失敗する。別の固定入力試験では採点前のraw、採用した候補の値、次のepochの観測を照合する。
- **GPU実学習。** 量子化なし・固定倍率・自動倍率を各39ステップ（3 epoch）実行した。各runの索引、4個の重み（定期2個・最終・final raw）の読戻しと有限値、観測CSV、診断レポートを確認した。通常の平均窓4では3 epochで採点に達しない。追加の自動倍率試験では開始比率0、窓2、bank目標2とし、第3 epochでrawと候補の両方を採点した。この試行ではrawが選ばれた。
- **GPUでの更新棄却。** 4試行の実学習で順に通常更新、Guardian棄却、AMP overflow、通常更新を起こした。schedulerが進んだ回数は順に1・0・0・1で、GradScalerのscaleはoverflow時だけ半減した。
- **回帰テスト。** CUDA有効の環境で `tests/migration` の527件が成功した。学習結果について今回確かめた範囲は短い実行での更新・保存・観測・採点である。
