# SDXL Self-Distill v2

この fork には、SDXL LoRA のための軽量 self-distill v2 基盤を追加しています。通常の `train_network.py` / `sdxl_train_network.py` の学習フローはそのまま残し、自己蒸留系だけを別導線で再構成しています。

v2 の目的は「Teacher LoRA の挙動を、軽量な single-timestep cache 学習で keep / suppress し直すこと」です。通常の画像 LoRA 学習とは異なり、画像 dataset を直接学習しません。代わりに `Base model` と `Teacher LoRA` の差分観測を cache 化し、その cache を学習データとして使います。

## v2 の要点

- 学習時は `Teacher/Base live forward なし`
- 学習時は `Student U-Net 1回 forward`
- `Student` は基本的に `Teacher` と同じ LoRA 重みから開始
- `U-Net only train`
- `Teacher` に Text Encoder LoRA が含まれる場合も対応
- `export_te_mode=preserve` が既定
- `xt_source_mode=teacher_rollout` が MVP 本線
- `--network_train_text_encoder_only` は未対応で hard fail します
- warm start する preset では LoRA の rank / shape は `dim_from_weights=true` で元 weights から読みます

## 追加された主なファイル

- `tools/build_prompt_bank.py`
- `tools/build_self_distill_cache.py`
- `tools/cache_audit.py`
- `tools/eval_self_distill.py`
- `sdxl_self_distill_network.py`
- `library/self_distill_cache.py`
- `library/self_distill_dataset.py`
- `library/self_distill_losses.py`
- `library/self_distill_targets.py`
- `library/self_distill_sampler.py`
- `library/lbw_profile.py`
- `configs/self_distill/*.toml`

## 通常の画像 LoRA 学習との違い

共通点:

- base model / LoRA / optimizer / scheduler / mixed precision の扱いは通常の sd-scripts に寄せています
- safetensors 出力、LoRA warm start、LBW-like profile を使えます

違い:

- 画像 dataset ではなく cache manifest を読む
- Teacher/Base の観測は cache build 時だけ行う
- 学習時は prompt を live で TE に通さず、cached conditioning を使う
- keep / suppress の条件別に loss を切り替える
- `cache_conditioning=false` のとき、cache build 中の TE forward は CPU のまま行い、embeddings だけ必要 device に移します
- cache build は base bundle と teacher bundle を CPU に持ち、record ごとに片方ずつ GPU に載せます

## style distill と composition distill

`style_distill`:

- 主題や構図を変えても Teacher の画風差分を残したいとき向け
- `high_pass_delta_loss` は optional
- leak を減らしつつ混ぜやすい LoRA を目指す preset

`composition_distill`:

- exact pose 再現ではなく、柔らかい構図傾向を残したいとき向け
- `coarse_preservation_loss` と `low_pass_delta_loss` を弱めに使う preset
- ControlNet 的な厳密空間制御の代替ではありません

## prompt bank の役割

prompt bank は「元の学習タグの再現」ではなく、「Teacher-base 差分が安定して出る条件を観測する bank」です。

標準 variant:

- `keep_strong`
- `keep_weak`
- `off_null`
- `frontier`
- `suppress_trigger_*`

variant ごとに `conditioning_source` と `loss_role` を持ちます。

MVP の conditioning 方針:

- `keep_*`: `teacher`
- `off_null`: `base`
- `suppress_trigger_*`: default は `teacher`
- `frontier_tags` が空なら `frontier` variant は生成されません

### tag 群の役割

- keep trigger: 残したい中心 trigger。最初は 1 個推奨
- suppress trigger / nuisance tag: 弱めたい trigger や nuisance
- support tag: Teacher の良い差分を見えやすくする補助タグ
- template: prompt bank の本体
- frontier tag: いまは弱いが将来伸ばしたい方向

### template 軸

- `carrier_families`: 6〜10
- `shot_types`: 3〜4
- `lighting_envs`: 3〜4
- 開始時は template 50〜80、seed 3〜4、variant 4〜6 が実用ラインです

### holdout split

- template 単位で 15〜20% を `holdout` に分離できます
- 学習 prompt 専用の偽物を避けるため、eval は `holdout` を基本にしてください

## cache の構成

manifest 1行は `prompt variant × seed` の record です。NPZ には複数 timestep を入れます。

必須:

- `target_timesteps`
- `x_t`
- `teacher_target`
- `base_target`
- `base_prompt_embeds`
- `base_negative_prompt_embeds`
- `base_pooled_prompt_embeds`
- `base_negative_pooled_prompt_embeds`

`teacher_te_included=true` のとき必須:

- `teacher_prompt_embeds`
- `teacher_negative_prompt_embeds`
- `teacher_pooled_prompt_embeds`
- `teacher_negative_pooled_prompt_embeds`

任意:

- `initial_noise_latent`
- `teacher_final_latent`
- preview path

### strict manifest check

cache 読み込み時に以下を hard fail で検証します。

- base model identifier/hash
- teacher LoRA identifier/hash
- LBW profile hash
- scheduler / prediction type
- resolution
- `xt_source_mode`
- timestep sampling mode
- prompt bank hash
- cache schema version

cache header の `scheduler` / `prediction_type` は cache build 時の CLI 既定値ではなく、prompt bank 側の `generation_settings` から確定します。

## Teacher TE を含む場合

Teacher に Text Encoder LoRA が含まれる場合、trigger 応答は TE と U-Net の連携で成立します。v2 ではこれを正式対応とし、teacher/base の conditioning cache を保存します。

- `export_te_mode=preserve` が既定です
- 学習時は TE を更新しません
- export 時は TE LoRA を凍結保持したまま student safetensors に含めます
- `drop` は将来拡張扱いで、trigger 応答が変わる可能性があります
- `teacher_te_included=true` の cache に対して `export_te_mode=drop` は未対応で、train 開始時に hard fail します
- `--lbw_profile` を付けた評価では、retain/suppress/drift の基準も `teacher+LBW` に切り替わります
- `teacher_te_included=true` の cache では、student 初期重みが U-Net-only でも student 側に TE LoRA を作って保持します
- `dim_from_weights=true` かつ U-Net-only init でも、teacher TE 付き cache なら U-Net 側の rank/alpha を引き継いで TE LoRA を生成します

## x_t と target

MVP は `xt_source_mode=teacher_rollout` です。

1. same prompt / same seed / same scheduler で Teacher を rollout
2. 指定した inference step の latent を `x_t` として保存
3. その同じ `x_t` 上で Teacher/Base の guided prediction を target 化
4. target は `eps` または `v` に変換して保存

## loss

コア:

- `keep_delta_loss`
- `suppress_to_base_loss`
- `weight_anchor_loss`

optional:

- `coarse_preservation_loss`

experimental:

- `high_pass_delta_loss`
- `low_pass_delta_loss`
- `sparse_loss`

`suppress_trigger_*` の MVP 成功条件は「Teacher より明確に弱まる」「base 側に寄る」です。`export_te_mode=preserve` のまま完全 suppress は目標にしません。

## optimizer preset と attention backend

optimizer preset:

- `adamw8bit`
- `adafactor_fixedlr`
- `adagrad8bit`
- `rmsprop8bit`

使い分け:

- まずは `adamw8bit`
- 低 lr 固定で比較したいなら `adafactor_fixedlr`
- `adagrad8bit` / `rmsprop8bit` は bitsandbytes 依存です。未対応環境では hard fail します

attention backend:

- `auto`
- `sdpa`
- `xformers`

`auto` は `xformers > sdpa > default` の順で選びます。

## coverage と step 数

定義:

- `T = template 数`
- `R = seeds / template`
- `V = variant 数`
- `K = timesteps / record`
- `N_records = T × R × V`
- `N_items = N_records × K`
- `coverage rho = (S × B × A) / N_items`

目安:

- `rho = 0.2〜0.4`: smoke test
- `rho = 0.5〜1.0`: 初回本命
- `rho = 1.0〜2.0`: 強めの suppress / forgetting
- `rho > 3.0`: 上書きしすぎのリスク

Teacher が十分学習済みなら、v2 の開始 lr は `teacher lr / 10〜30` を目安にしてください。

- keep 重視: `1e-5〜2e-5`
- keep + suppress 強め: `2e-5〜3.5e-5`
- frequency 実験: `5e-6〜1e-5`

step 数は Teacher の元 step をそのまま使わず、coverage で決めます。

- smoke / first-pass: `500〜1500`
- 本命: `1500〜3500`

## ディスク容量の考え方

latent は fp16 で概算します。

- 640 解像度の latent は約 `4 × 80 × 80 × 2 bytes ≒ 50KB`
- 768 解像度の latent は約 `4 × 96 × 96 × 2 bytes ≒ 72KB`
- record 1件で `x_t[K]`, `teacher_target[K]`, `base_target[K]`, conditioning を持つため、実サイズはこれより大きくなります
- 実用セットでは prompt conditioning が支配的になりやすいので、まず small bank で実測してください

## 16GB 向け基本方針

- `train_batch_size = 1`
- `gradient_accumulation_steps = 1 or 2`
- `mixed_precision = fp16`
- `optimizer_preset = adamw8bit`
- `attention_backend = auto` または `xformers`
- `gradient_checkpointing = true`
- `resolution = 640`
- preview generation during train は off
- Teacher/Base は学習時に live forward しない

危険設定:

- `resolution > 768`
- `batch_size > 1`
- `sample_steps` を大きくしすぎる
- conditioning cache なしで学習しようとする構成

## 最小実行例

prompt bank:

```bash
python tools/build_prompt_bank.py \
  --output outputs/self_distill/prompt_bank.json \
  --keep_triggers "my_style" \
  --support_tags "diffused light,wet surface" \
  --carrier_families "portrait photo of a person,product photo of a ceramic cup,rock and water landscape material" \
  --shot_types "close-up,bust shot,material study" \
  --lighting_envs "soft studio light,overcast daylight,rim light" \
  --seed_list "101,102,103" \
  --variant_quota '{"keep_strong":0.4,"keep_weak":0.2,"off_null":0.3,"frontier":0.1}'
```

cache build:

```bash
python tools/build_self_distill_cache.py \
  --config_file configs/self_distill/base.toml \
  --prompt_bank outputs/self_distill/prompt_bank.json \
  --output_dir outputs/self_distill/cache \
  --teacher_lora_weights path/to/teacher.safetensors \
  --pretrained_model_name_or_path path/to/sdxl_base.safetensors \
  --sample_sampler euler_a
```

cache audit:

```bash
python tools/cache_audit.py \
  --cache_manifest outputs/self_distill/cache/manifest.jsonl \
  --output_dir outputs/self_distill/audit \
  --output_csv
```

train:

```bash
python sdxl_self_distill_network.py \
  --config_file configs/self_distill/base.toml \
  --cache_manifest outputs/self_distill/cache/manifest.jsonl \
  --teacher_lora_weights path/to/teacher.safetensors \
  --student_init_weights path/to/teacher.safetensors \
  --output_dir outputs/self_distill/train \
  --output_name self_distill_v2
```

eval:

```bash
python tools/eval_self_distill.py \
  --config_file configs/self_distill/base.toml \
  --pretrained_model_name_or_path path/to/sdxl_base.safetensors \
  --eval_prompts outputs/self_distill/prompt_bank.json \
  --teacher_lora_weights path/to/teacher.safetensors \
  --student_lora_weights outputs/self_distill/train/self_distill_v2-step001500.safetensors \
  --output_dir outputs/self_distill/eval
```

dry run:

```bash
python tools/build_self_distill_cache.py \
  --config_file configs/self_distill/base.toml \
  --prompt_bank outputs/self_distill/prompt_bank.json \
  --output_dir outputs/self_distill/cache \
  --teacher_lora_weights path/to/teacher.safetensors \
  --pretrained_model_name_or_path path/to/sdxl_base.safetensors \
  --dry_run

python sdxl_self_distill_network.py \
  --config_file configs/self_distill/base.toml \
  --cache_manifest outputs/self_distill/cache/manifest.jsonl \
  --teacher_lora_weights path/to/teacher.safetensors \
  --student_init_weights path/to/teacher.safetensors \
  --dry_run
```

## smoke test の目安

- templates: 24
- seeds: 2
- variants: 3〜5
- timesteps per record: 2
- resolution: 640
- batch_size: 1
- grad_accum: 1
- steps: 200〜500

## 既知の制限

- `export_te_mode=drop` は将来拡張扱いです
- `suppress_trigger_*` は MVP では U-Net 側抑制です
- exact pose 再現や ControlNet 的厳密空間制御は対象外です
- high/low frequency 機能は optional な実験機能です
- rank 圧縮、TE 学習、full rollout 学習は v2 本線に含めません
- `eval_split` に該当する record が 0 件なら `eval_self_distill.py` は明示エラーで止まります

## 補足

- `--resume_cache` は既存 manifest header と現在設定が一致する場合だけ再利用されます
- train 時の strict manifest check で teacher 一致まで見たい場合は `--teacher_lora_weights` を渡してください
- `gradient_accumulation_steps > 1` のときも requested `max_train_steps` に届くように sampler 長を調整しています
- `--logging_dir` を使う場合、self-distill train でも tracker を初期化して `accelerator.log` へ流します
- `pretrained_model_name_or_path` は単一ファイルだけでなく diffusers directory や model identifier でも manifest hash を計算できます
