@echo off
setlocal

rem ============================================
rem SDXL Self-Distill v1 minimal 4-step template
rem For cmd.exe / .bat
rem ============================================

rem 1. Required settings
set BASE=C:\path\to\sdxl_base.safetensors
set TEACHER=C:\path\to\teacher_lora.safetensors
set TRIGGER=mytrg

rem 2. Optional settings
set SUPPORT_TAGS=
set FRONTIER_TAGS=

rem 3. Working directories
set WORKDIR=outputs\self_distill_test
set PROMPT_BANK=%WORKDIR%\prompt_bank.json
set CACHE_DIR=%WORKDIR%\cache
set TRAIN_DIR=%WORKDIR%\train
set EVAL_DIR=%WORKDIR%\eval

rem 4. Experiment settings
set RESOLUTION=768
set SAMPLE_STEPS=8
set GUIDANCE=7.5
set SAMPLER=euler_a
set NUM_TEMPLATES=12
set SEEDS=101,102
set MAX_TRAIN_STEPS=500
set GRAD_ACCUM=4
set LEARNING_RATE=1e-4
set OUTPUT_NAME=mytrg_sd_test

rem 5. Attention backend
rem Use --sdpa for SDPA
rem Use --xformers for xformers
set ATTN=--sdpa

rem 6. Activate venv if needed
rem call venv\Scripts\activate.bat

if not exist "%WORKDIR%" mkdir "%WORKDIR%"
if not exist "%CACHE_DIR%" mkdir "%CACHE_DIR%"
if not exist "%TRAIN_DIR%" mkdir "%TRAIN_DIR%"
if not exist "%EVAL_DIR%" mkdir "%EVAL_DIR%"

echo.
echo ============================================
echo Step 1/4: build prompt bank
echo ============================================
python tools\build_prompt_bank.py ^
  --output "%PROMPT_BANK%" ^
  --trigger_token "%TRIGGER%" ^
  --support_tags "%SUPPORT_TAGS%" ^
  --frontier_tags "%FRONTIER_TAGS%" ^
  --carrier_families "1girl,portrait,anime illustration" ^
  --shot_types "close-up,bust shot,upper body" ^
  --lighting_envs "studio lighting,soft rim light,outdoor daylight" ^
  --seed_list "%SEEDS%" ^
  --num_templates %NUM_TEMPLATES% ^
  --width %RESOLUTION% ^
  --height %RESOLUTION% ^
  --sample_steps %SAMPLE_STEPS% ^
  --guidance_scale %GUIDANCE% ^
  --sample_sampler %SAMPLER%
if errorlevel 1 goto :error

echo.
echo ============================================
echo Step 2/4: build self-distill cache
echo ============================================
python tools\build_self_distill_cache.py ^
  --pretrained_model_name_or_path "%BASE%" ^
  --prompt_bank "%PROMPT_BANK%" ^
  --teacher_lora_weights "%TEACHER%" ^
  --output_dir "%CACHE_DIR%" ^
  --network_module networks.lora ^
  --resolution %RESOLUTION% ^
  --cache_prompt_embeddings ^
  --mixed_precision fp16 ^
  %ATTN%
if errorlevel 1 goto :error

echo.
echo ============================================
echo Step 3/4: train self-distill student
echo ============================================
python sdxl_self_distill_network.py ^
  --pretrained_model_name_or_path "%BASE%" ^
  --cache_manifest "%CACHE_DIR%\manifest.jsonl" ^
  --student_init_weights "%TEACHER%" ^
  --output_dir "%TRAIN_DIR%" ^
  --output_name "%OUTPUT_NAME%" ^
  --network_module networks.lora ^
  --dim_from_weights ^
  --network_train_unet_only ^
  --train_batch_size 1 ^
  --gradient_accumulation_steps %GRAD_ACCUM% ^
  --max_train_steps %MAX_TRAIN_STEPS% ^
  --learning_rate %LEARNING_RATE% ^
  --unet_lr %LEARNING_RATE% ^
  --optimizer_type AdamW8bit ^
  --mixed_precision fp16 ^
  --save_precision fp16 ^
  %ATTN% ^
  --gradient_checkpointing ^
  --max_data_loader_n_workers 1 ^
  --require_cached_prompt_embeddings ^
  --positive_high_pass_delta_weight 1.0 ^
  --coarse_preservation_weight 0.25 ^
  --off_loss_weight 1.0 ^
  --anchor_loss_weight 0.1 ^
  --high_pass_mode dog ^
  --low_pass_mode avg ^
  --save_every_n_steps 100
if errorlevel 1 goto :error

set STUDENT=%TRAIN_DIR%\%OUTPUT_NAME%-step000500.safetensors
if not exist "%STUDENT%" (
  echo.
  echo Expected student checkpoint not found:
  echo   %STUDENT%
  echo Check MAX_TRAIN_STEPS and OUTPUT_NAME.
  goto :error
)

echo.
echo ============================================
echo Step 4/4: evaluate base / teacher / student
echo ============================================
python tools\eval_self_distill.py ^
  --pretrained_model_name_or_path "%BASE%" ^
  --eval_prompts "%PROMPT_BANK%" ^
  --teacher_lora_weights "%TEACHER%" ^
  --student_lora_weights "%STUDENT%" ^
  --output_dir "%EVAL_DIR%" ^
  --network_module networks.lora ^
  --resolution %RESOLUTION% ^
  --sample_sampler %SAMPLER% ^
  --mixed_precision fp16 ^
  %ATTN%
if errorlevel 1 goto :error

echo.
echo ============================================
echo Completed
echo ============================================
echo Prompt bank : %PROMPT_BANK%
echo Cache       : %CACHE_DIR%\manifest.jsonl
echo Student     : %STUDENT%
echo Eval dir    : %EVAL_DIR%
goto :eof

:error
echo.
echo ============================================
echo Failed
echo ============================================
exit /b 1
