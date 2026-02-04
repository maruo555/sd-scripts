#!/usr/bin/env python3
import argparse
import os
from typing import Iterable, List, Optional, Tuple

from transformers import CLIPTokenizer

TOKENIZER1_PATH = "openai/clip-vit-large-patch14"
TOKENIZER2_PATH = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"


def _load_tokenizer(model_id: str, cache_dir: Optional[str], fix_pad_token_id: bool) -> CLIPTokenizer:
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        local_dir = os.path.join(cache_dir, model_id.replace("/", "_"))
        if os.path.exists(local_dir):
            tokenizer = CLIPTokenizer.from_pretrained(local_dir)
        else:
            tokenizer = CLIPTokenizer.from_pretrained(model_id)
            tokenizer.save_pretrained(local_dir)
    else:
        tokenizer = CLIPTokenizer.from_pretrained(model_id)

    if fix_pad_token_id:
        tokenizer.pad_token_id = 0

    return tokenizer


def _tokenize(
    tokenizer: CLIPTokenizer, text: str, add_special_tokens: bool
) -> Tuple[List[str], List[int], List[str], str]:
    encoded = tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        return_attention_mask=False,
        return_tensors=None,
    )
    input_ids = encoded["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    pieces = [tokenizer.convert_tokens_to_string([t]) for t in tokens]
    normalized = tokenizer.convert_tokens_to_string(tokens)
    return tokens, input_ids, pieces, normalized


def _print_tokens(label: str, model_id: str, tokenizer: CLIPTokenizer, text: str, add_special_tokens: bool) -> None:
    tokens, ids, pieces, normalized = _tokenize(tokenizer, text, add_special_tokens)
    boundary = "|".join(pieces)

    print(f"{label} ({model_id})")
    print(f"token_count: {len(tokens)}")
    print(f"normalized: {normalized}")
    print(f"boundary: {boundary}")
    print("tokens:")
    for i, (tok, tid, piece) in enumerate(zip(tokens, ids, pieces)):
        print(f"{i:>3}  token={tok}  id={tid}  piece={piece!r}")
    print("")


def _piece_len(piece: str) -> int:
    stripped = piece.strip()
    if stripped == "":
        return 0
    return len(stripped)


def _has_single_char_piece(pieces: List[str]) -> bool:
    return any(_piece_len(p) == 1 for p in pieces)


def _iter_additions(alphabet: str, min_add: int, max_add: int) -> Iterable[str]:
    if max_add < min_add:
        return []
    if max_add < 1:
        return []
    start_len = 1 if min_add < 1 else min_add

    def _recurse(prefix: str, depth: int):
        if depth >= start_len:
            yield prefix
        if depth == max_add:
            return
        for ch in alphabet:
            yield from _recurse(prefix + ch, depth + 1)

    return _recurse("", 0)


def _search_candidates(
    core: str,
    side: str,
    alphabet: str,
    min_add: int,
    max_add: int,
    tokenizer1: CLIPTokenizer,
    tokenizer2: CLIPTokenizer,
    add_special_tokens: bool,
    require_both: bool,
):
    candidates = []
    if min_add == 0:
        # include the core itself if it already satisfies the condition
        t1_core = _tokenize(tokenizer1, core, add_special_tokens)
        t2_core = _tokenize(tokenizer2, core, add_special_tokens)
        t1_tokens, _, t1_pieces, _ = t1_core
        t2_tokens, _, t2_pieces, _ = t2_core

        t1_ok = (len(t1_tokens) >= 2) and (not _has_single_char_piece(t1_pieces))
        t2_ok = (len(t2_tokens) >= 2) and (not _has_single_char_piece(t2_pieces))

        if (t1_ok and t2_ok) if require_both else (t1_ok or t2_ok):
            candidates.append(
                {
                    "text": core,
                    "direction": "none",
                    "addon": "",
                    "addon_len": 0,
                    "t1_tokens": t1_tokens,
                    "t1_pieces": t1_pieces,
                    "t2_tokens": t2_tokens,
                    "t2_pieces": t2_pieces,
                }
            )

    additions = list(_iter_additions(alphabet, min_add, max_add))
    for add in additions:
        texts = []
        if side in {"prefix", "both"}:
            texts.append((add + core, "prefix", add))
        if side in {"suffix", "both"}:
            texts.append((core + add, "suffix", add))

        for text, direction, addon in texts:
            t1 = _tokenize(tokenizer1, text, add_special_tokens)
            t2 = _tokenize(tokenizer2, text, add_special_tokens)
            t1_tokens, _, t1_pieces, _ = t1
            t2_tokens, _, t2_pieces, _ = t2

            t1_ok = (len(t1_tokens) >= 2) and (not _has_single_char_piece(t1_pieces))
            t2_ok = (len(t2_tokens) >= 2) and (not _has_single_char_piece(t2_pieces))

            if require_both:
                ok = t1_ok and t2_ok
            else:
                ok = t1_ok or t2_ok

            if not ok:
                continue

            candidates.append(
                {
                    "text": text,
                    "direction": direction,
                    "addon": addon,
                    "addon_len": len(addon),
                    "t1_tokens": t1_tokens,
                    "t1_pieces": t1_pieces,
                    "t2_tokens": t2_tokens,
                    "t2_pieces": t2_pieces,
                }
            )
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show token splits for SDXL text encoders (TE1/TE2).",
    )
    parser.add_argument("--text", help="Input tag string, e.g. \"stnc,xa\"")
    parser.add_argument(
        "--tokenizer-cache-dir",
        default=None,
        help="Directory for tokenizer cache (optional).",
    )
    parser.add_argument(
        "--add-special-tokens",
        action="store_true",
        help="Include special tokens (BOS/EOS).",
    )
    parser.add_argument(
        "--search-core",
        default=None,
        help="Core word to wrap with extra chars for searching (e.g. \"yuzu\").",
    )
    parser.add_argument(
        "--search-side",
        choices=["prefix", "suffix", "both"],
        default="both",
        help="Where to add extra chars around the core word.",
    )
    parser.add_argument(
        "--search-min-add",
        type=int,
        default=1,
        help="Minimum number of chars to add.",
    )
    parser.add_argument(
        "--search-max-add",
        type=int,
        default=3,
        help="Maximum number of chars to add.",
    )
    parser.add_argument(
        "--search-alphabet",
        default="abcdefghijklmnopqrstuvwxyz",
        help="Alphabet to use when generating additions.",
    )
    parser.add_argument(
        "--search-limit",
        type=int,
        default=20,
        help="Maximum number of results to print.",
    )
    parser.add_argument(
        "--search-either",
        action="store_true",
        help="Allow either TE1 or TE2 to meet conditions (default: require both).",
    )
    args = parser.parse_args()

    tokenizer1 = _load_tokenizer(TOKENIZER1_PATH, args.tokenizer_cache_dir, fix_pad_token_id=False)
    tokenizer2 = _load_tokenizer(TOKENIZER2_PATH, args.tokenizer_cache_dir, fix_pad_token_id=True)

    if args.search_core:
        candidates = _search_candidates(
            core=args.search_core,
            side=args.search_side,
            alphabet=args.search_alphabet,
            min_add=args.search_min_add,
            max_add=args.search_max_add,
            tokenizer1=tokenizer1,
            tokenizer2=tokenizer2,
            add_special_tokens=args.add_special_tokens,
            require_both=not args.search_either,
        )

        if not candidates:
            print("no candidates found")
            return

        def score(c):
            t1_len = len(c["t1_tokens"])
            t2_len = len(c["t2_tokens"])
            return (max(t1_len, t2_len), c["addon_len"], c["text"])

        candidates.sort(key=score)
        best_score = score(candidates[0])[:2]
        filtered = [c for c in candidates if score(c)[:2] == best_score]

        print(f"core: {args.search_core}")
        print(f"side: {args.search_side}")
        print(f"alphabet: {args.search_alphabet}")
        print(f"add_range: {args.search_min_add}..{args.search_max_add}")
        print(f"require_both: {not args.search_either}")
        print("")
        print(f"best_candidates: {min(len(filtered), args.search_limit)}")
        print("")

        for c in filtered[: args.search_limit]:
            t1_boundary = "|".join(c["t1_pieces"])
            t2_boundary = "|".join(c["t2_pieces"])
            print(f"text: {c['text']}  ({c['direction']} +{c['addon']})")
            print(f"  TE1: {len(c['t1_tokens'])}  {t1_boundary}")
            print(f"  TE2: {len(c['t2_tokens'])}  {t2_boundary}")
            print("")
        return

    if not args.text:
        parser.error("either --text or --search-core is required")

    print(f"input: {args.text}")
    print("")
    _print_tokens("TE1", TOKENIZER1_PATH, tokenizer1, args.text, args.add_special_tokens)
    _print_tokens("TE2", TOKENIZER2_PATH, tokenizer2, args.text, args.add_special_tokens)


if __name__ == "__main__":
    main()
