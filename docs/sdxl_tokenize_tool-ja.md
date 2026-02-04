# SDXLトークン分割ツール（sdxl_tokenize.py）

## ツールの説明
SDXL の2つのテキストエンコーダー（TE1/TE2）で、入力タグ文字列がどのようにトークン分割されるかを表示します。  
さらに、コア語の前後に1～数文字を付け足して「単文字トークンを含まない」「2トークン以上」の候補を探索できます。

## 全オプション一覧（モード別）
**共通（どのモードでも有効）**
| オプション | 説明 | 既定値 |
|---|---|---|
| `--tokenizer-cache-dir` | トークナイザのキャッシュディレクトリ | なし |
| `--add-special-tokens` | BOS/EOS などの特殊トークンを含める | 無効 |

**`--text` モード（単純分割表示）**
| オプション | 説明 | 既定値 |
|---|---|---|
| `--text` | 解析する入力文字列 | なし |

**`--search-core` モード（候補探索）**
| オプション | 説明 | 既定値 |
|---|---|---|
| `--search-core` | 探索対象のコア語（例: `yuzu`） | なし |
| `--search-side` | 追加文字の位置（`prefix`/`suffix`/`both`） | `both` |
| `--search-min-add` | 追加文字数の最小（`0` の場合は追加なしも探索） | `1` |
| `--search-max-add` | 追加文字数の最大 | `3` |
| `--search-alphabet` | 追加文字に使う文字集合 | `abcdefghijklmnopqrstuvwxyz` |
| `--search-limit` | 表示する候補数 | `20` |
| `--search-either` | TE1/TE2 のどちらかが条件を満たせば採用 | 無効 |

注意:
- `--search-core` を指定すると探索モードになり、`--text` は無視されます。
- 探索モードの条件は「2トークン以上」かつ「単文字トークンを含まない」です。
- `--search-either` を付けない場合は **TE1/TE2 両方**で条件を満たす候補のみ出力します。

## 使い方の例1: `--text` で分割を調べる
```bash
python tools/sdxl_tokenize.py --text "stnc,xa"
```
TE1/TE2 それぞれの `boundary`（`|`区切り）とトークン一覧が表示されます。

## 使い方の例2: `--search-core` で候補探索
```bash
python tools/sdxl_tokenize.py --search-core "yuzu" --search-side both --search-min-add 0 --search-max-add 3
```
`yuzu` の前後に 1～3 文字を付けた候補を探索し、条件に合う最短候補を表示します。  
TE1/TE2 どちらかだけ条件を満たせば良い場合は `--search-either` を付けてください。
