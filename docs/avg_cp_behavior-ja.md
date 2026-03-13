# --avg_cp の挙動整理（sdxl_train_network.py）

## 対象コードと役割

- `train_network.py`  
  エポック終端で `--avg_cp` の判定・平均化・再ロード・optimizer 統計のリセットを行う。
- `library/avg_ckpt_util.py`  
  LoRA 重みだけを抽出し（`lora_` で始まるキーのみ）、平均（`uniform`/`ema`/`metric`）を計算する。
- `README.md`  
  オプション説明とフロー（raw 保存 → 平均 → 次エポック開始）が記載されている。

## 実装フロー（エポック終端の順番）

1. **raw ckpt の保存**（`--save_every_n_epochs` が有効な場合）  
   - `epoch+1 < num_train_epochs` のときのみ保存されるため、**最終エポックの raw は保存されない**。
2. **画像サンプルの生成**（`self.sample_images`）
3. **平均化 (`--avg_cp`)**
   - 条件: `(epoch + 1) / num_train_epochs >= avg_begin`
   - LoRA の `state_dict` を `cp_window`（最大 `avg_window`）に追加。
   - 窓が満杯になったら `average_state_dicts` で平均し、**平均後の重みをそのまま学習モデルにロード**。
   - `--avg_reset_stats` 指定時、Optimizer の `exp_avg` / `exp_avg_sq` / `exp_avg_max` をゼロ化（`step` は保持）。
4. 次エポックは **平均後の重み** から開始。

### `--avg_mode ema` の重みのかかり方（窓サイズ N の例）

`alpha = 2 / (N + 1)` で、古い重みから順に EMA 更新する。  
`N=4` の場合 `alpha=0.4` となり、重み配分は以下の比率になる。

```
epoch_t-3 : 0.216
epoch_t-2 : 0.144
epoch_t-1 : 0.240
epoch_t   : 0.400
```

（古いほど重みが小さく、新しいほど大きい）

## 40エポック + 指定設定の具体例

前提:
- `max_train_epochs = 40`
- `avg_begin = 0.6`
- `avg_window = 4`
- `avg_mode = ema`
- `save_every_n_epochs = 1`

### 平均開始エポック

`(epoch + 1) / 40 >= 0.6` となる最初のエポックは **24**。  
よって **24 エポック目の終了時**から `cp_window` に溜まり始める。

### 具体的な平均対象

| エポック終端 | cp_window に入る | 平均の発動 | 平均対象 |
|---|---|---|---|
| 24 | 24 | しない | - |
| 25 | 24, 25 | しない | - |
| 26 | 24, 25, 26 | しない | - |
| 27 | 24, 25, 26, 27 | **する** | 24〜27 |
| 28 | 25, 26, 27, 28 | **する** | 25〜28 |
| 29 | 26, 27, 28, 29 | **する** | 26〜29 |
| … | … | … | … |
| 40 | 37, 38, 39, 40 | **する** | 37〜40 |

補足:
- 平均は **エポック終端で実行**されるため、**次のエポックは平均済み重み**から開始される。
- `save_every_n_epochs=1` の場合、raw ckpt は **1〜39 エポック**まで保存される。  
  40 エポック目の raw は保存されず、最終保存は **平均後の状態**で行われる。

## 「平均なし」を同時に作る方法はあるか

### 現状の挙動

- **raw ckpt は平均前に保存される**ため、同一エポックの「平均前スナップショット」は得られる。
- ただし平均が入った時点で学習モデル自体が変わるため、**「平均なしで最後まで学習した結果」とは一致しない**。

### 可能性のあるアプローチ

1. **別実行で比較する（最も正確）**  
   - `--avg_cp` あり／なしで 2 走させ、最終結果を比較。
2. **コード改修で“影の平均”を保存する**  
   - 平均を計算して別名で保存するが、学習モデルにはロードしない。  
   - もしくは平均後に保存してから元の重みへ戻す（学習挙動を変えない）。  
   - 現状そのためのフラグや分岐は **未実装**。
3. **学習後に後処理で平均する**  
   - 学習時は `--avg_cp` を使わず raw を保存。  
   - その後 `library/avg_ckpt_util.py` 相当で複数 ckpt を平均し、  
     「平均版」を別途生成する（学習中の平均化とは厳密には別）。

## まとめ

- `--avg_cp` は **LoRA 部分のみ**を対象に、学習の後半で窓平均を適用し、  
  **平均後の重みで学習を継続**する設計。
- 「平均あり」と「平均なし」を**完全に同時生成**する仕組みは現状ない。  
  比較には **別実行**が確実。  
  もしくは **平均版だけ別保存する改修**が必要。

## `--avg_cp_mode shadow` の追加仕様

### この機能の目的と位置づけ

この `shadow` は、いきなり「平均した重みを本採用して学習を続ける」のではなく、まず  
**平均候補が本当に raw より良いのかを、安全に観測するための段階**として追加された。

考え方は次のとおり:

- 学習本線はこれまでどおり `raw` のまま進める
- epoch 終端でだけ、`raw` と「直近 `avg_window` epoch を `avg_mode` で平均した center」を比較する
- 比較は backward を行わない軽い追加計算（forward-only scoring）で行う
- その結果を JSONL に記録し、
  - 平均候補が raw より勝つことが多いのか
  - margin を入れても勝つのか
  - 連勝する傾向があるのか
  を観測する

つまり、これは「avg_cp を自動採用する機能」の完成版ではなく、  
**平均を本採用する価値があるかを同一 run 内で検証するための観測モード**である。

将来的には、この観測結果が十分よさそうなら:

- score の良い center を実際に学習モデルへ promote して次 epoch に進む

という拡張につなげる想定だが、今回の実装ではそこまでは行わない。  
今回はあくまで **raw vs center を同条件で比較し、best を保存し、ログを取るところまで** を実装している。

### 目的

- **学習本線は raw のまま進める**
- 同一 run・同一 seed 条件で、`raw` と `current avg_mode で作る center 1本` を比較する
- VRAM 増加は常時ほぼゼロに寄せ、追加コストは epoch 終端の forward-only 採点時間と CPU RAM に寄せる

### CLI

- `--avg_cp_mode {live,shadow}`
  - 既定値は `live`
  - `live` は従来どおり、平均後重みをそのまま学習モデルにロードする
  - `shadow` は平均候補を**採点と保存にだけ使い、学習モデルにはロードしない**
- `--avg_shadow_bank_size`
  - 既定値は `12`
  - train_proxy bank に保持する**学習バッチ数**
- `--avg_shadow_margin`
  - 既定値は `0.003`
  - `center_score < raw_score * (1 - margin)` のときだけ `center` 勝ち
- `--avg_shadow_patience`
  - 既定値は `2`
  - virtual streak 計算用のみ。**actual promote はしない**
- `--avg_shadow_log_jsonl`
  - 既定値は `True`
  - `output_dir/avg_shadow+<output_name>.jsonl` に epoch ごとの JSONL を出す

### 実装フロー

#### 1. 学習 step 中の train_proxy bank 収集

条件:

- `--avg_cp`
- `--avg_cp_mode shadow`
- `(epoch + 1) / num_train_epochs >= avg_begin`
- まだ `avg_shadow_bank_size` に達していない

収集タイミング:

- `target` 算出後
- backward 前

保持する情報:

- `noisy_latents`
- `target`
- `timesteps`
- `huber_c`
- `loss_weights`
- `input_ids`
- `input_ids2`（SDXL のとき）
- `network_multipliers`（存在するとき）
- `alpha_masks`（存在するとき）
- `original_sizes_hw`
- `crop_top_lefts`
- `target_sizes_hw`
- `text_encoder_outputs1_list`
- `text_encoder_outputs2_list`
- `text_encoder_pool2_list`
- `captions`（weighted captions 経路で必要なとき）

実装上の保存方針:

- bank は **CPU 保持のみ**
- `noisy_latents` / `target` / `alpha_masks` / cached TE 出力は CPU `fp16` に圧縮
- `timesteps` など整数テンソルはそのまま CPU に detach/clone
- `avg_shadow_bank_size` に達した時点で **freeze** し、その後は更新しない
- `clean latents` は保持しない
- VAE 再実行もしない
- ノイズの再サンプリングもしない

#### 2. epoch end の raw snapshot / center 作成

epoch 終端の順番は以下:

1. 通常の `save_every_n_epochs` による raw ckpt 保存
2. `sample_images`
3. `clean_memory_on_device()` で一時メモリ解放
4. `avg_begin` 到達後なら、現在の LoRA state を `raw_sd` として CPU に抽出
5. `cp_window` と `cp_window_epochs` に追加

ここでの分岐:

- `avg_cp_mode=live`
  - 従来どおり `average_state_dicts()` で平均し、**平均後重みを学習モデルにロード**
  - `avg_reset_stats` も従来どおり有効
- `avg_cp_mode=shadow`
  - **学習モデルは raw のまま**
  - `cp_window` が `avg_window` に満ち、かつ bank 完成後のみ採点する

#### 3. proxy scoring

shadow の採点は epoch 終端で行う。

比較対象:

- `raw_sd`
- `avg_mode` で `cp_window` から作った `center_sd`

採点手順:

1. 現在の raw モデルで `raw_score` を計算
2. RNG state を元に戻す
3. `center_sd` をモデルに一時ロード
4. 同じ bank で `center_score` を計算
5. 採点後に **必ず raw に戻す**
6. RNG state も元に戻し、shadow scoring が次 epoch の乱数系列を汚さないようにする

VRAM 方針:

- raw と center を GPU に同時常駐させない
- bank item は 1 件ずつ GPU に送る
- backward なし
- optimizer 更新なし
- `torch.no_grad()` で forward-only 採点

#### 4. score の定義

score は `train_proxy bank` 上の **mean loss**。

実装では、学習 step と shadow scoring の両方が同じ helper を使う:

- `NetworkTrainer._get_text_conds_for_batch()`
- `NetworkTrainer._compute_batch_loss()`

この helper 内で再利用している処理:

- `call_unet()`
- `train_util.conditional_loss()`
- `apply_masked_loss()`
- `loss_weights` 乗算
- `apply_snr_weight()`
- `scale_v_prediction_loss_like_noise_prediction()`
- `add_v_prediction_like_loss()`
- `apply_debiased_estimation()`

つまり、judge の loss 式は**通常学習の loss と同じ系統**を使っている。

補足:

- SDXL では `sdxl_train_network.py` の `get_text_cond()` が outer の grad mode を尊重するように修正されており、shadow scoring 中に不要な graph を強制生成しない

#### 5. winner 判定

- `center_score < raw_score * (1 - avg_shadow_margin)` のときだけ `winner = "center"`
- それ以外は `winner = "raw"`

あわせて以下を epoch ごとに計算・記録する:

- `virtual_margin_ok`
- `virtual_win_streak`
- `virtual_would_promote = (virtual_win_streak >= avg_shadow_patience)`

ただし:

- **actual promote は未実装**
- 学習モデルを center に切り替えることはしない

#### 6. 保存物

shadow では、score が更新されたときに以下を直接保存する:

- `best_raw.safetensors`
- `best_center.safetensors`
- `best_auto.safetensors`

定義:

- `best_raw`: 観測済み raw の中で proxy score 最小
- `best_center`: 観測済み center の中で proxy score 最小
- `best_auto`: その時点までに観測した raw / center 全体で proxy score 最小

保存実装:

- 現在の in-memory LoRA state dict から直接 `.safetensors` を保存
- `save_every_n_epochs` に依存しない
- LoRA 部分だけを保存する
- SAI metadata と safetensors hash metadata も付与する

#### 7. JSONL logging

shadow 有効時は epoch ごとに 1 行の JSONL を出す。

主な出力項目:

- `epoch`
- `progress`
- `bank_ready`
- `bank_size`
- `avg_mode`
- `avg_window`
- `raw_proxy_loss`
- `center_proxy_loss`
- `delta_abs`
- `delta_pct`
- `winner`
- `virtual_margin_ok`
- `virtual_win_streak`
- `virtual_would_promote`
- `window_epochs`
- `status`

`status` の例:

- `before_avg_begin`
- `bank_collecting`
- `waiting_window`
- `scored`

#### 8. 実際の制約

- 今回の実装で比較する center は **現在の `args.avg_mode` で作る 1 本だけ**
- `val_bank` は未実装
- `adaptive promote` は未実装
- resume 時は `cp_window` を既存 epoch ckpt から復元するが、**shadow bank / best_* / streak は復元しない**
- 分散学習では score/save/log は main process で行う
  - ただし bank の all-gather はしていないため、**proxy bank は main process の local shard ベース**

### まとめ

- `shadow` は「学習は raw のまま」「平均候補は epoch end に観測だけ」のモード
- 判定 loss は学習と同じ helper 経路に寄せている
- 常時増える VRAM はほぼなく、追加コストは CPU bank と epoch end forward に寄せている
- `best_raw` / `best_center` / `best_auto` を **1 run から取得できる**
