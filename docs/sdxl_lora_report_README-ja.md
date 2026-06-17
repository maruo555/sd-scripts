# SDXL LoRA Report

`sdxl_lora_report` は、SDXL LoRAの比較用画像をまとめて生成し、ブラウザで見やすいHTMLレポートを作るための補助ツールです。

主な目的は、LoRAの差分確認、LBW(LoRA Block Weight)の効き方確認、複数LoRAの重ね掛け比較を、同じprompt/seed条件で効率よく行うことです。

## できること

- 1つ以上のLoRA条件を指定して画像を一括生成する。
- 同じprompt、同じseedで、LoRAだけ違う画像を並べて比較する。
- 1つの比較条件に複数LoRAを入れて、LoRAの重ね掛けを比較する。
- LoRAごとに `strength` と `lbw` を指定する。
- 必要に応じてLoRAなしの `baseline` 画像も生成する。
- 生成結果を1つの出力フォルダにまとめる。
- HTMLレポートでLoRA、prompt、seedの表示切り替えや画像サイズ変更を行う。

## ファイル構成

```text
sdxl_lora_report_gui.py
  PySide6 GUI。通常はこちらを使う。

sdxl_lora_report_cui.py
  config JSONを読んでジョブを展開し、workerを呼び、metadataとHTMLを作る。

sdxl_lora_report_worker.py
  1回の sdxl_gen_img.py 起動で複数LoRA条件を切り替えながら生成するworker。

lora_report_samples/
  CUI用のサンプルconfigとpromptファイル。
```

通常の実行経路は以下です。

```text
sdxl_lora_report_gui.py
  -> sdxl_lora_report_cui.py
    -> sdxl_lora_report_worker.py
      -> sdxl_gen_img.py
```

## GUIの使い方

PySide6が必要です。`requirements.txt` に追加されていますが、既存環境を壊したくない場合は、必要なvenvに手動で `PySide6` だけ入れてください。

起動例:

```powershell
python sdxl_lora_report_gui.py
```

画面上部で以下を指定します。

- `Model`: SDXLモデルファイル。`.safetensors` または `.ckpt` をドラッグ&ドロップできます。
- `Output root`: レポートの出力先フォルダ。フォルダをドラッグ&ドロップできます。
- `Prompt file`: prompt一覧のテキストファイル。`.txt` をドラッグ&ドロップできます。
- `Run name`: 出力フォルダ名に使う識別名。

左側の `LoRA assets` に `.safetensors` のLoRAをドラッグ&ドロップします。

中央の `Comparison conditions` が、レポート上で比較される条件です。

- `Make single conditions`: 選択中のLoRAから、単体LoRA条件をまとめて作ります。
- `Add condition`: 空の比較条件を作ります。
- `Add selected LoRA`: 選択中のLoRAを、選択中の比較条件に追加します。
- 1つの条件にLoRAが1個なら単体LoRA、2個以上なら重ね掛け条件です。
- 条件内のLoRA行で `strength` と `LBW` を編集できます。

右側で生成設定を指定します。

- `Width` / `Height`
- `Steps`
- `Sampler`
- `Scale`
- `Batch size`
- `Images / prompt`
- `Common args`
- `Seeds`
- `Include baseline`

`Include baseline` をオンにすると、LoRAなし画像も同じprompt/seedで生成します。

## 前回設定の復元

GUI実行時には `.tmp/lora_report_gui_last.json` が作られます。

これはGUIからCUIへ渡す実行用configですが、次回GUI起動時に以下の設定を復元するためにも使われます。

- Model
- width
- height
- steps
- sampler
- scale
- batch_size
- images_per_prompt
- common_args

LoRA条件やprompt fileまでは自動復元しません。実験条件を勝手に持ち越す事故を避けるためです。

`.tmp/` は `.gitignore` に入っています。実モデルパスや個人環境のLoRAパスが入るため、コミットしないでください。

## Prompt File

promptファイルは1行1promptです。

単純な書き方:

```text
1girl, standing, smile
1girl, sitting, smile
```

詳細指定:

```text
prompt_id | prompt | negative prompt | width | height
```

例:

```text
standing_smile | 1girl, standing, smile | low quality, worst quality | 1024 | 1024
```

空行と `#` で始まる行は無視されます。

## CUIの使い方

GUIを使わずにconfig JSONから実行することもできます。

サンプル:

```powershell
python sdxl_lora_report_cui.py --config lora_report_samples\lora_report_sample.json --dry-run
```

本番実行:

```powershell
python sdxl_lora_report_cui.py --config path\to\config.json
```

既存画像をスキップする場合:

```powershell
python sdxl_lora_report_cui.py --config path\to\config.json --skip-existing
```

## Config JSON

サンプルは `lora_report_samples/lora_report_sample.json` です。

主な項目:

```json
{
  "output_root": "../../lora_reports",
  "run_name": "sample_lora_compare",
  "prompt_file": "lora_report_prompts_sample.txt",
  "sdxl_gen_img": {
    "ckpt": "D:/models/sdxl_model.safetensors",
    "width": 1024,
    "height": 1024,
    "steps": 30,
    "sampler": "euler_a",
    "scale": 7.0,
    "batch_size": 1,
    "images_per_prompt": 1,
    "common_args": ["--xformers", "--bf16"]
  },
  "seeds": {
    "values": [12345],
    "random_count": 0
  },
  "include_baseline": true,
  "loras": []
}
```

LoRA単体条件:

```json
{
  "id": "sample_xlmlt1",
  "name": "Sample LoRA XLMLT1",
  "path": "D:/loras/sample_lora.safetensors",
  "strength": 1.0,
  "lbw": "XLMLT1"
}
```

LoRA重ね掛け条件:

```json
{
  "id": "sample_stack",
  "name": "Sample LoRA stack",
  "items": [
    {
      "name": "character",
      "path": "D:/loras/sample_character.safetensors",
      "strength": 0.8,
      "lbw": "XLMLT1"
    },
    {
      "name": "style",
      "path": "D:/loras/sample_style.safetensors",
      "strength": 0.5,
      "lbw": "ALL"
    }
  ]
}
```

LBWを使う条件では、その条件内のすべてのLoRA itemに `lbw` を指定してください。

## HTML Report

出力先には日時つきフォルダが作られます。

```text
output_root/
  20260617_120000_run_name/
    report.html
    metadata.json
    config.json
    prompts.txt
    prompts.parsed.json
    images/
    worker/
```

`report.html` では以下ができます。

- X軸/Y軸の入れ替え
- LoRA条件、prompt、seedの表示/非表示切り替え
- 画像表示サイズのスライダー変更
- 画像クリックで元画像表示

## 実装コンセプト

このツールは、`sdxl_gen_img.py` の生成機能を直接大きく改造せず、比較レポート用の薄い制御層として作っています。

重要な考え方:

- GUIは生成ロジックを持たない。
- GUIはconfig JSONを作り、CUIを起動するだけにする。
- CUIはconfigを正規化し、prompt/seed/LoRA条件をジョブへ展開する。
- workerは `sdxl_gen_img.py --from_file --sequential_file_name` を1回だけ起動し、各prompt行の `--am` でLoRA倍率を切り替える。
- LoRAの読み込み回数とモデルロード回数を減らし、比較条件が増えても扱いやすくする。

workerは全LoRA条件で必要なLoRAをスロット化します。

同じ `(module, path, lbw)` のLoRAは1つのスロットとしてまとめられます。各画像生成時には、条件に応じて `--am` の倍率を変えます。

例:

```text
LoRA A only:  --am 0.8 0.0
LoRA B only:  --am 0.0 0.8
LoRA A+B:     --am 0.8 0.8
baseline:     --am 0.0 0.0
```

これにより、単体LoRA、複数LoRA重ね掛け、baselineを同じ生成プロセス内で扱えます。

## 制約

- 現在のworkerは `images_per_prompt=1` を前提にしています。
- SDXL LoRA LBW利用を主目的にしているため、LBWを使う場合は `networks.lora_lbw` が使われます。
- GUIの前回復元対象はモデルと生成設定だけです。LoRA条件やprompt fileは自動復元しません。
- `.tmp/` は個人環境のパスを含むためコミット対象外です。

## 今後の開発メモ

次に拡張するなら、以下が候補です。

- GUIのLoRA条件ツリーへのドラッグ&ドロップ操作をさらに直感的にする。
- LBWプリセット管理画面を追加する。
- 1つのLoRAに対して複数LBWを自動展開するLBW sweep機能を追加する。
- strength sweep機能を追加する。
- promptごとの画像サイズや個別negative prompt編集をGUI上で行えるようにする。
- レポートに条件メモや評価コメント欄を追加する。
- 生成済みレポートをGUIから開く履歴機能を追加する。
