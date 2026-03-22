# SDXL Self-Distill v1

この fork には、SDXL LoRA の自己蒸留実験用の最小構成を追加しています。通常の `train_network.py` / `sdxl_train_network.py` の学習フローはそのまま残し、自己蒸留用の新規 script と補助 module を別系統で追加しています。

通常の SDXL LoRA 学習との違いは、画像 dataset を直接学習に使わず、`base model` と `teacher LoRA` の観測結果を事前に disk cache 化しておき、学習中はその cache を読みながら `student LoRA` だけを更新する点です。v1 では `U-Net only`、`text encoder frozen`、`teacher/base live forward なし` を前提にしています。

## 追加された主なファイル

- `tools/build_prompt_bank.py`
- `tools/build_self_distill_cache.py`
- `sdxl_self_distill_network.py`
- `tools/eval_self_distill.py`
- `library/self_distill_cache.py`
- `library/self_distill_dataset.py`
- `library/self_distill_losses.py`
- `library/lbw_profile.py`
- `configs/self_distill/*.toml`

## フロー

1. `build_prompt_bank.py` で single trigger 前提の prompt bank を作る
2. `build_self_distill_cache.py` で `base model` と `teacher LoRA` の観測を cache 化する
3. `sdxl_self_distill_network.py` で cached target に対して `student LoRA` を学習する
4. `eval_self_distill.py` で `base / teacher / student` を比較する

v1 は軽量構成で、teacher/base は cache build 時だけ使います。学習時は cache に入っている `initial_noise_latent`、生成設定、teacher/base の final latent を読み、student だけを rollout します。

## 最小実行例

prompt bank 作成:

```bash
python tools/build_prompt_bank.py \
  --output outputs/self_distill/prompt_bank.json \
  --trigger_token sks_character \
  --carrier_families "1girl,anime illustration" \
  --shot_types "bust shot,close-up" \
  --lighting_envs "studio lighting,soft rim light" \
  --seed_list "101,102"
```

cache 作成:

```bash
python tools/build_self_distill_cache.py \
  --config_file configs/self_distill/base.toml \
  --prompt_bank outputs/self_distill/prompt_bank.json \
  --output_dir outputs/self_distill/cache \
  --teacher_lora_weights path/to/teacher.safetensors \
  --pretrained_model_name_or_path path/to/sdxl_base.safetensors \
  --cache_prompt_embeddings
```

train 実行:

```bash
python sdxl_self_distill_network.py \
  --config_file configs/self_distill/base.toml \
  --cache_manifest outputs/self_distill/cache/manifest.jsonl \
  --student_init_weights path/to/teacher.safetensors \
  --output_dir outputs/self_distill/train \
  --output_name self_distill_run
```

eval 実行:

```bash
python tools/eval_self_distill.py \
  --pretrained_model_name_or_path path/to/sdxl_base.safetensors \
  --eval_prompts outputs/self_distill/prompt_bank.json \
  --teacher_lora_weights path/to/teacher.safetensors \
  --student_lora_weights outputs/self_distill/train/self_distill_run-step000100.safetensors \
  --output_dir outputs/self_distill/eval
```

## 各 script の役割

### `tools/build_prompt_bank.py`

- single trigger LoRA 向けに prompt bank を JSON で生成します
- `carrier_families × shot_types × lighting_envs` をベースに template を作ります
- `strong / weak / off / support_only / frontier` を variant として展開します

主な入力:

- `--trigger_token`
- `--support_tags`
- `--frontier_tags`
- `--carrier_families`
- `--shot_types`
- `--lighting_envs`
- `--seed_list`

### `tools/build_self_distill_cache.py`

- prompt bank を読み、base model と teacher LoRA の観測 cache を作ります
- `teacher_final_latent` と `base_final_latent` を保存します
- 必要に応じて prompt embeddings と preview 画像も保存できます
- LBW-like profile を teacher 側に適用できます
- teacher LoRA に Text Encoder 重みが含まれる場合は `--cache_prompt_embeddings` が必須です

cache の必須項目:

- `prompt_text`
- `variant_type`
- `seed`
- `generation_settings`
- `initial_noise_latent`
- `teacher_final_latent`
- `base_final_latent`

cache の任意項目:

- `prompt_embeds`
- `negative_prompt_embeds`
- `pooled_prompt_embeds`
- `negative_pooled_prompt_embeds`
- `preview_image_path`

### `sdxl_self_distill_network.py`

- cache から prompt / latent / 設定を読みます
- student LoRA を既存 LoRA 重みから warm start できます
- `--dim_from_weights` 相当の初期化を使えます
- 学習中は teacher/base を live で回しません
- loss は `library/self_distill_losses.py` で差し替えやすくしています

対応している loss:

- positive high-pass delta loss
- coarse preservation loss
- off loss
- anchor loss
- optional sparse loss

high-pass:

- `dog`
- `laplacian`
- `gaussian_residual`

low-pass:

- `avg`
- `gaussian`
- `identity`

### `tools/eval_self_distill.py`

- fixed prompt suite と fixed seed で `base / teacher / student` を比較します
- preview grid を保存します
- JSON metrics を保存します

v1 で出す proxy:

- `retain_proxy`
- `leakage_proxy`
- `drift_proxy`

## 16GB VRAM で危険な設定

- `resolution > 768`
- `batch_size > 1`
- `gradient_checkpointing` 無効
- `cache_prompt_embeddings` 無効で、長い prompt を毎 step 再エンコードする構成
- `sample_steps` を大きくしすぎる構成

## OOM 対策

- `configs/self_distill/vram16.toml` を基準にする
- `batch_size=1` を維持する
- `gradient_accumulation_steps` を増やす
- `gradient_checkpointing` と `xformers` を有効にする
- cache を disk に置き、prompt embeddings も cache する
- まず `sample_steps=8` 前後の short smoke test で詰める

## 差し替えポイント

- loss を差し替える: `library/self_distill_losses.py`
- prompt bank のロジックを差し替える: `tools/build_prompt_bank.py`
- cache schema や rollout helper を拡張する: `library/self_distill_cache.py`
- LBW-like profile を調整する: `library/lbw_profile.py`

## v1 で未実装の点

- text encoder 学習
- teacher/base を学習時に live で回す mode
- intermediate timestep target
- perceptual / image loss
- A1111 完全互換の LBW
