# rank_log 仕様（暫定）

`train_network.py` の `--rank_log` 系オプションで、LoRA重みから推定した rank 飽和指標をCSV出力します。

## 目的
- `dq_delta`（フェイク量子化）の有無に依存せず、LoRAの rank 使用状態を時系列で観測する。
- `dq_delta_log` とは独立に、rank 指標のみを記録する。

## オプション
- `--rank_log`
  - rankログを有効化。既定値: `OFF`。
- `--rank_log_every <int>`
  - 記録間隔（optimizer step）。既定値: `100`。
- `--rank_log_mode {summary,per_module}`
  - 既定値: `summary`。
  - `summary`: UNet全体の集約指標。
  - `per_module`: モジュールごとの指標。
- `--rank_log_file <path>`
  - 出力先。既定値: `None`。
  - 未指定時は `output_dir/rank_logs+<output_name>.txt`。

## 出力スキーマ

### summary
`Epoch,TrainStep,Scope,RankDim,RankSatWMean,RankSatP50,RankSatP95,RankSatMax,RankTop1P95,RankEnergySum`

### per_module
`Epoch,TrainStep,Scope,Module,RankDim,RankSat,RankTop1,RankEnergy`

## 指標の意味（要点）
- `RankSat*`: 実効rankの飽和度（高いほど rank を使い切りやすい）。
- `RankTop1*`: 上位1成分の支配度（高いほど rank1 偏り傾向）。
- `RankEnergy*`: LoRA更新の総エネルギー。

## 補足
- 現状の集計対象 `Scope` は `unet` のみです。
- 値は `networks/lora.py` の `compute_rank_stats()` に基づきます。
