# SDXL LoRA学習の機能別解説

SDXL LoRAの独自機能を使った学習で、どの処理が重みを変え、どの数値を観測しているかを説明します。実際に起動する設定は[学習設定の使用例](maruo_features_ja.md)、引数名・既定値・選択肢は[独自CLIリファレンス](maruo_cli_ja.md)から引けます。

|知りたいこと|読む章|
|---|---|
|どの学習で使えるか。1試行はどう進むか|[対象と処理の順序](#scope)|
|学習する層、学習率、計算精度を変える|[TE1・TE2](#encoders)、[SDXL UNet](#unet)|
|重みを更新する、見送る、epoch末に候補を採用する|[optimizer](#optimizer)、[Guardian](#guardian)、[平均とpromote](#averaging)|
|LoRAの出力を量子化する|[固定倍率・bits](#dq)、[自動倍率](#dq-auto)、[Triton](#triton)|
|何が起きたか記録とファイルで確かめる|[6種類のCSV](#logging)、[run・設定・重み](#run-output)、[実行中の操作](#control)|

<a id="scope"></a>

## 対象と1試行の順序

学習挙動を変える独自機能は通常のAccelerateで使用します。DeepSpeedはbackward内で更新するため、CLI・TOMLやAccelerateの起動設定でDeepSpeedを有効にした場合、独自学習との併用は開始時にエラーになります。独自学習をすべてOFFにした場合は本家のDeepSpeed経路を使用できます。公開されている独自学習引数は、CLI・TOMLのどちらで指定しても、実効のネットワークモジュールが `networks.lora` であることが必要です。

学習の入口は `sdxl_train_network.py --network_module networks.lora` です。独自機能で学習の挙動を変える構成は、1プロセス、勾配蓄積1、datasetのbatch size 1・実効 `network_multiplier=1.0` で、新しいrunとして開始します。datasetで同じ画像を何回学習するかは `num_repeats` で設定できます。TE出力のキャッシュ、inpainting学習、subsetの `diff_output_preservation`、optimizer途中状態からの `--resume` は、この構成の対象外です。常用例の `--cache_latents` は使用できます。本家の学習にrankやgroup lossなどの観測だけを加えた場合は、本家の `--resume` で再開でき、再開後の観測は新しいrunに記録します。既存LoRAを `--network_weights` で読み込む操作は、その重みを初期値とする新規学習です。

1. **forward前**に、完了済みの試行数から、その試行で使うDQのbits、開始条件、統計を取る部位を決めます。
2. **forward・backward**でTEとUNetを通し、対象のLoRA出力にDQを適用します。lossを計算し、勾配を作ります。promote用の入力を収集中なら、ここで実バッチを控えます。
3. **更新前**にGuardianが勾配を判定します。通過すれば通常の勾配クリップ・optimizer・LR schedulerの処理に進み、棄却すれば更新を見送ります。どちらも次の試行のために勾配をクリアします。
4. **試行末**に進捗を1つ進めます。対象の試行ではgroup loss、DQ統計・倍率の判断、rankを記録します。自動倍率の決定は以後のforwardに反映されます。この境界で定期サンプル・保存・外部操作も処理します。

画面の `steps` は同期試行の数です。Guardianが見送った回も進むため、たとえば「100試行ごと」の観測時点と、「optimizerが100回更新された時点」は別々に数えます。

<a id="encoders"></a>

## TE1・TE2の対象層と学習率

`--te_mlp_fc_only` はSDXLの2本のテキストエンコーダーについて、各Transformerブロックの `mlp.fc1` と `mlp.fc2` にLoRAを付けます。先頭から末尾ブロックまでを対象とし、TE内のself-attentionや最終projectionは含みません。TE1が12ブロック・TE2が32ブロックのモデルでは、対象はそれぞれ24層・64層です。UNet側の対象層はこの指定とは独立です。

各TEの基準LRは、`--learning_rate` → 共通の `--text_encoder_lr` → 該当する `--text_encoder_lr1` または `--text_encoder_lr2` の順に決まります。たとえば共通 `2e-4`、TE1 `3e-4`、TE2省略なら、TE1は `3e-4`、TE2は `2e-4` です。本家の `--text_encoder_lr` に2値を与えた場合はそれが各TEの指定になるため、同じTEに異なる個別値を重ねると開始時にエラーになります。

LRが0のグループはoptimizerの更新から外れます。すでに取り付けたTEのLoRAはforwardに残るので、`--dq_delta_scope both` ならその出力も量子化の対象です。run内の `training_settings.json` の `resolved.text_encoder_lrs` で基準LR、`resolved.optimizer_groups_at_start` で実際に作られたグループを確かめられます。

<a id="unet"></a>

## SDXL UNetの精度とtimestep埋め込み

|機能|変わる演算|適用範囲|
|---|---|---|
|fp16での正規化 `--fp16_safe_norms_mode strict`|入力がfp16のとき、ResNet・出力部のGroupNorm、空間TransformerのGroupNorm、TransformerブロックのLayerNormをfp32で計算し、演算直後に元のfp16へ戻します。通常attentionのsoftmaxもfp32で計算してスコアのdtypeへ戻します。|SDXL UNet内の対象演算のみ。SDPAやxformersを使うattention本体は選んだバックエンドで実行します。bf16入力にはstrictのfp16専用分岐は掛かりません。|
|周波数シフト `--downscale_freq_shift`|拡散ノイズのtimestepをsin/cosで埋め込むときの周波数シフトを0から1にします。|UNetの拡散timestep埋め込みのみ。SDXLが別に使う元画像サイズ・crop位置・目標サイズの埋め込みは本家の計算を使います。|

どちらの指定も今回のrunの `training_settings.json` の `resolved.sdxl_unet` に記録されます。TEの学習対象やoptimizerの状態精度は、それぞれ専用の設定で決まります。

<a id="optimizer"></a>

## LoRAを更新するoptimizer

`AdamW8bitFast` はbitsandbytesのAdamW8bitの更新と状態形式を使います。対応する単一CUDAデバイスでは、小さいLoRAテンソルを1つ更新するたびにCPUが待つ処理をまとめ、optimizerの1ステップ末に同期します。条件に合わないケースではbitsandbytes標準の更新経路を使います。コンソールに選択した経路が表示されます。

`AdamWBnb` も同じ同期の仕組みを使いますが、8bit状態へ切り替えるテンソルサイズの閾値を高く固定します。通常のLoRAでは32bitのoptimizer状態を使う選択肢です。両者ともoptimizer内部の選択であり、学習中のLoRA出力のbitsは[別途DQ](#dq)で選びます。たとえば `AdamW8bitFast` はDQを無効にした学習でも使えます。

<a id="guardian"></a>

## Guardianによる勾配更新のスキップ

Guardianはbackward直後、AMPのunscaleと勾配クリップより前の勾配ノルムを見ます。初期閾値は200,000、履歴が200件たまると現在の観測を含む窓の「平均＋2.5×標準偏差」で判定します。常用の `stable_no_threshoff` は有限値で窓を作り、閾値の上限を200,000とします。`stable` は非有限値も窓に含めるため、影響が後続の閾値にも及び得ます。`gamble` は有限値で窓を作り、動的閾値に上限を置かず、非有限の勾配ノルムは直ちに棄却します。

棄却でも `gradient.csv` にその回のノルム、閾値、loss、AMPスケールが残り、進捗の `skipped` が増えます。DQ統計と自動倍率の更新、rank観測、group lossのEMAはその回を見送ります。たとえば100試行目が棄却なら、100試行目のrank行は書かれず、通常の次の候補は200試行目です。`skipped` はGuardianだけの累計です。AMP側による見送りもあるため、実更新回数を `steps − skipped` で決めることはできません。

<a id="averaging"></a>

## epoch末の平均とpromote採点

`--avg_cp` を使うと、開始割合 `--avg_begin` に達した各epoch末に、LoRAの重みをCPUのfp32で窓に保管します。`uniform` は窓全体の単純平均、`ema` は窓の中の新しいepochを重くする平均です。40 epoch・開始割合0.6・窓4なら24 epoch目から窓に入り、27 epoch目に最初の平均候補ができます。

|モード|窓が揃った後の処理|
|---|---|
|`live`|平均した重みを採用し、次の学習に使います。|
|`promote`|同じ採点用バッチ集合（bank）で現在の重みと平均候補を比べます。候補の平均lossが所定の改善幅を超えて下がり、指定の連勝回数に達すると採用します。|

bankには訓練中に得たノイズ付き潜在、目標、timestep、トークンなどを保存し、クラスや画像の偏りを抑えて固定します。収集はGuardian判定前です。採点時にはTE1・TE2で各候補のテキスト条件を再計算します。2候補は同じTorch乱数状態から採点し、終わったら学習用の乱数位置に戻します。採点にはoptimizer更新、DQの統計収集・倍率更新、rank・group loss観測を含めません。bankの判定値は、この固定入力での学習lossの比較です。画像の仕上がりは、保存した重みで生成して確認します。[画像比較の手順](maruo_tools_ja.md#compare)

平均採用時、既定のresetは存在する `exp_avg`、`exp_avg_sq`、`exp_avg_max` をゼロにします。既存のstepとbitsandbytesの `state1`・`state2` は保持します。`optimizer_reset_applied=true` はこの処理を通った記録です。常用例の `--no-avg_reset_stats` ではreset自体を省きます。採用・保留の理由を調べる際は `promote.csv` の `row_type=decision` を選び、`status`、2つのproxy loss、`win_streak`、`promote_applied` を順に見ます。`window_epoch` 行は候補に入ったepochの内訳です。

<a id="dq"></a>

## DQで量子化する値、開始時期、固定倍率

DQは学習forwardでLoRAが元の層に加える**出力差分（delta）**に作用します。LoRAのdown・dropout・upを通し、倍率とscaleを掛けた後に差分を量子化し、元の層の出力へ足します。保存するLoRA重みは通常の保存形式・精度のままです。

1. チャンネル別RMSから、指定した `range_mul` に応じて表現できる幅と刻みを決めます。8bitなら対称な整数段階は−127から127です。
2. 各値を隣の2段階のどちらかへ確率的に丸め、範囲の外は端にクリップします。backwardでは量子化前の差分へ勾配を通します。
3. 倍率を大きくすると表現範囲が広がり、同じbitsで刻みも大きくなります。`dq.csv` の `clip_rate_raw` と `quant_error_rms_raw` を一緒に見ると、端に達した割合と丸めを含む誤差の両方を確認できます。

量子化は固定bits、または進捗に応じたbitsスケジュールで有効にします。最低は2bitです。`--dq_delta_begin_after_lr_warmup` は、正のLR warmup指定から**予定の試行数**で開始点を決め、`--dq_delta_begin` より優先します。2,000試行で `--lr_warmup_steps 0.05` なら、100試行を終えた直後の101試行目からです。Guardian棄却で実際のLR更新が遅れても、開始位置は同じです。

固定倍率の例では基本倍率2.70、UNetのattn2と両TEは3.75です。TE1・TE2の個別倍率を指定した側は、TE共通倍率より優先します。場所別上書きは固定bits専用で、bitsスケジュールや自動倍率との組合せは開始前に検査されます。`training_settings.json` の `resolved.dq.mul` には設定、対象名、割り当て結果を照合する指紋が残ります。

<a id="dq-auto"></a>

## DQ自動倍率で何を判定するか

|対象を決める引数|自動倍率の設定例での動作|
|---|---|
|`--dq_delta_scope both`|UNetとTE1・TE2のLoRA出力を量子化します。|
|`--dq_delta_log_scope` を省略|量子化scopeを引き継ぎ、ログの観測時点でUNetとTEの集計を別行に記録します。|
|`--dq_delta_auto_scope` を省略|UNetも学習するこの構成ではUNetの観測で自動倍率を判断します。決定した共通倍率は、両TEを含む量子化対象で使います。|

自動倍率の例では `clip_rate_high` の目標帯（0.3〜0.5%）を使い、目標帯から初期倍率を求めます。既定では50試行ごとに観測し、該当する試行のforwardで集めたclip率のEMAを見ます。`--dq_delta_auto_use_raw` もあれば、EMAとその回の生のclip率が同じ側に外れた場合に倍率を調整します。上側に外れれば倍率を上げて範囲を広げ、下側なら下げます。判断後の値は次のforwardから使います。

開始直後はauto自身のwarmupで観測を蓄えます。これはLR warmupとは別に進み、有効な測定でEMAが帯内に3回続けば早期に終了します。`dq_auto.csv` の `event_type=init` と `check`、`warmup_active`、`auto_reason`、`common_mul_before`／`common_mul_after` で、「未判定」「観測して維持」「倍率変更」を区別できます。初期倍率そのものは `resolved.dq.auto.init.mul` に記録します。`clip_rate_low_auto` は低帯で開始し、所定の進捗区間で量子化誤差の条件が続いた場合に一度だけ中帯へ移行できる方式です。

<a id="triton"></a>

## DQのTriton経路

`--dq_delta_use_triton` は、対応するCUDAテンソルのRMS・確率丸めにTritonカーネルを使います。`--dq_delta_triton_stats` を併用すると、詳細な統計を取る回には丸めと集計の融合も試します。形状・dtype・メモリ配置などに応じて、融合、通常のTriton丸め、PyTorch参照実装へ段階的に切り替えます。経路を切り替えても、そのforwardで用意した丸め乱数を再利用します。

Triton指定があるだけで、すべての層が融合カーネルを通ったとは判定できません。`training_settings.json` で指定を、`dq.csv` と `dq_auto.csv` で学習中に観測した値を確認します。計算経路の性能を比較するときはbits・scope・観測間隔を揃えます。

<a id="logging"></a>

## 6種類のCSVの読み分け

各CSVは機能を有効にしたrunに作られます。欠けた観測や空欄を数値0として補わず、`run.json` の記録状態と併せて読みます。

|ファイル|1行が表すもの|最初に見る値|
|---|---|---|
|`gradient.csv`|Guardianが判定する前に求めた勾配ノルムと、その回のしきい値・loss。棄却回も記録|epoch内の `batch_index`、`grad_norm`、`threshold`、`grad_scale`|
|`dq.csv`|指定間隔の同期試行で集めたdelta統計。両側を記録する場合は部位ごとに1行|同じ `attempt_step` の `scope`、`bits`、clip率と誤差|
|`dq_auto.csv`|自動倍率の初期化または判定1回|観測scope、倍率の前後、warmup、判定理由|
|`rank.csv`|100同期試行ごと、Guardianを通過した回のUNet LoRA 1モジュール|有効rank／設定rankの `rank_sat`、最も強い成分のエネルギー割合 `rank_top1`、総エネルギー `rank_energy`|
|`group_loss.csv`|勾配を同期した回の最後のバッチのloss。Guardianが棄却した回とlossが非有限の回は省く|groupとsubset、`timestep`、`bucket_reso`、loss、groupごとのEMAと観測数|
|`promote.csv`|1 epochの候補判断または平均窓の内訳|採点・採用の件数を数えるときは `decision` 行を使用|

group lossは `--group_loss_log` を指定し、dataset TOMLの `[[datasets.subsets]]` に `group = "standing"` のように名前を付けます。勾配蓄積が1なら各バッチが記録対象です。蓄積が2なら、2バッチごとに最後の1バッチのgroupとlossを記録します。グループごとの観測数とEMA（直前の値の重みは0.98）は、記録した行だけで更新し、同名のgroupを持つsubset間で共有します。`group` は `[general]` → `[[datasets]]` → `[[datasets.subsets]]` の順で内側の指定を優先します。継承元にも指定がない場合、または内側で空白名を指定した場合は `__ungrouped__` にまとまります。rankのTE1・TE2 LR列は実optimizerグループから得た学習率で、rank観測自体の対象はUNetです。run間で推移を並べるときの手順は[診断レポート](maruo_tools_ja.md#report)と[診断比較](maruo_tools_ja.md#viewer)にあります。

<a id="run-output"></a>

## runの識別、設定、どの重みを選ぶか

学習1回ごとに `<output_dir>/YYYYMMDD-HHMMSS_<output_name>/` ができます。同名・同秒なら連番が付きます。`run.json` のUUIDで学習回を識別し、`training_settings.json` の `requested` で指定内容、`resolved` で実際の設定を確認できます。台帳を作る前でも、このrun単独で設定と結果を読めます。

|選ぶ重み|使いどころ|
|---|---|
|step、epoch|途中の特定時点を比較する。定期epoch保存は、そのepochの平均・promote決定前。40 epoch・毎epoch保存の例では1〜39のepochファイルが該当します。|
|final|通常終了後に使う標準のLoRA。最終epochの平均・promoteの判断を反映した重みです。|
|final_raw|平均を有効にして `--avg_save_final_raw` を指定した場合の、最終epoch判定直前の重み。それまでのepochで採用した平均の影響は含まれます。|

重みの役割・epoch・試行・保存状態は `run.json` に記録されます。`--report_after_training` を指定すれば、正常終了してログが確定した後にこのrunの `diagnostic_report/` に診断を生成します。新runから評価を積み上げる方法は[研究台帳](maruo_tools_ja.md#ledger)を参照してください。

<a id="control"></a>

## 実行中の観測・サンプル・保存・停止

初期化が終わると「学習開始前の初期化時間」を表示し、進捗バーは実際の学習に入る境界から計時します。Guardian使用時の `skipped` とあわせて、起動時の待ち時間と学習中の進み方を読み分けられます。

`--run_control` を有効にすると、本家の操作CLIからサンプル画像生成・重み保存・一時停止・再開・停止を要求できます。実行中のコマンドは同期試行の境界で処理します。定期・手動サンプル画像は共有の `<output_dir>/sample/` に置き、ファイル名にrunの識別子を含めます。サンプル生成には解像度や生成条件に応じた時間と追加のVRAMが必要です。[本家のRun Control操作](run_control.md)

一時停止からの再開は、同じ学習プロセスを続行する操作です。stopで終了すると、その操作で要求したstep重み・stateを保存し、runの状態は `stopped` になります。通常終了のfinal・epoch末判定・終了後レポートはこの経路では作成しません。旧LoRAや保存したstep重みから新しい学習を始めるには `--network_weights` を使います。Guardianの履歴・平均窓・DQの自動倍率などを含む独自学習の途中状態を復元する機能は、現在のサポート範囲外です。
