"""Exercise SDXL token splitting and candidate ranking without downloading a model."""
import pytest

from library.clip_tokenizer import CLIPTokenizer
from library.strategy_sdxl import TOKENIZER1_PATH, TOKENIZER2_PATH
from tools import sdxl_tokenize as cli


@pytest.fixture
def tokenizer_cache(tmp_path):
    # TE1 can merge ab into one piece. TE2 must return separate one-character pieces.
    vocab = {token: i for i, token in enumerate(["<|startoftext|>", "<|endoftext|>", "a", "b</w>", "ab</w>"])}
    for model, merges in ((TOKENIZER1_PATH, [("a", "b</w>")]), (TOKENIZER2_PATH, [])):
        CLIPTokenizer(vocab=vocab, merges=merges).save_pretrained(tmp_path / model.replace("/", "_"))
    return str(tmp_path)


def test_raw_ids_boundaries_and_special_tokens(tokenizer_cache, capsys):
    cli.main(["--text", "ＡＢ", "--tokenizer-cache-dir", tokenizer_cache])
    printed = capsys.readouterr().out
    assert "token_count: 1\nnormalized: ab\nboundary: ab" in printed
    assert "token=ab</w>  id=4  piece='ab'" in printed
    assert "token_count: 2\nnormalized: ab\nboundary: a|b" in printed

    cli.main(["--text", "ＡＢ", "--tokenizer-cache-dir", tokenizer_cache, "--add-special-tokens"])
    printed = capsys.readouterr().out
    assert "boundary: <|startoftext|>|ab|<|endoftext|>" in printed
    assert "boundary: <|startoftext|>|a|b|<|endoftext|>" in printed


def test_core_with_no_additions_and_different_encoder_results(tokenizer_cache, capsys):
    args = [
        "--tokenizer-cache-dir", tokenizer_cache, "--text", "ignored", "--search-core", "ab",
        "--search-min-add", "0", "--search-max-add", "0", "--search-min-tokens", "1",
    ]
    cli.main(args)
    assert capsys.readouterr().out == "no candidates found\n"
    cli.main(args + ["--search-either"])
    printed = capsys.readouterr().out
    assert "best_candidates: 1" in printed
    assert "text: ab  (none +)\n  TE1: 1  ab\n  TE2: 2  a|b" in printed
    cli.main(args + ["--search-either", "--search-limit", "0"])
    printed = capsys.readouterr().out
    assert "best_candidates: 0" in printed and "text: ab " not in printed


def test_search_order_uses_larger_token_count_then_addon_length_then_text(monkeypatch):
    args = cli._parser().parse_args(["--search-core", "xy", "--search-alphabet", "ba", "--search-max-add", "2"])
    te1, te2 = object(), object()
    # "xya" has fewer TE1 tokens but its TE2 is longer. "bbxy" uses a longer add-on.
    counts = {"bxy": (2, 2), "axy": (2, 2), "xya": (2, 3), "bbxy": (2, 2)}

    def tokenize(tokenizer, text, special):
        count = counts.get(text, (1, 1))[int(tokenizer is te2)]
        return cli.Split(list(range(count)), ["piece"] * count, ["piece"] * count, text)

    monkeypatch.setattr(cli, "_tokenize", tokenize)
    assert [candidate.text for candidate in cli._search_candidates(args, te1, te2)] == ["axy", "bxy"]
    args.search_side = "suffix"
    assert [candidate.text for candidate in cli._search_candidates(args, te1, te2)] == ["xya"]


def test_missing_input_fails_before_loading_any_vocabulary(monkeypatch):
    monkeypatch.setattr(cli, "_load_tokenizer", lambda *args: pytest.fail("unexpected vocabulary access"))
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == 2
