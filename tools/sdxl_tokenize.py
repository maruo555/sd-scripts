#!/usr/bin/env python3
"""Show raw SDXL CLIP token boundaries and search for tag spellings."""
import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path


@dataclass
class Split:
    ids: list[int]
    tokens: list[str]
    pieces: list[str]
    normalized: str

    def qualifies(self, min_tokens):
        return len(self.ids) >= min_tokens and all(len(piece.strip()) != 1 for piece in self.pieces)

    @property
    def boundary(self):
        return "|".join(self.pieces)


def _tokenize(tokenizer, text, add_special_tokens):
    ids = tokenizer(text, add_special_tokens=add_special_tokens, return_attention_mask=False)["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(ids)

    def decode(token_ids):
        # CLIP's decoder restores word boundaries; convert_tokens_to_string
        # in transformers 5 leaves </w> intact and changes the search filter.
        return tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    return Split(ids, tokens, [decode([token_id]) for token_id in ids], decode(ids))


def _spellings(core, side, alphabet, min_add, max_add):
    if min_add == 0:
        yield core, "none", ""
    for size in range(max(1, min_add), max_add + 1):
        for chars in product(alphabet, repeat=size):
            addon = "".join(chars)
            if side in {"prefix", "both"}:
                yield addon + core, "prefix", addon
            if side in {"suffix", "both"}:
                yield core + addon, "suffix", addon


@dataclass
class Candidate:
    text: str
    direction: str
    addon: str
    te1: Split
    te2: Split

    @property
    def score(self):
        return max(len(self.te1.ids), len(self.te2.ids)), len(self.addon)


def _search_candidates(args, tokenizer1, tokenizer2):
    best_score = None
    best = []
    spellings = _spellings(args.search_core, args.search_side, args.search_alphabet, args.search_min_add, args.search_max_add)
    for text, direction, addon in spellings:
        te1 = _tokenize(tokenizer1, text, args.add_special_tokens)
        te2 = _tokenize(tokenizer2, text, args.add_special_tokens)
        checks = (te1.qualifies(args.search_min_tokens), te2.qualifies(args.search_min_tokens))
        if not (any(checks) if args.search_either else all(checks)):
            continue
        candidate = Candidate(text, direction, addon, te1, te2)
        score = candidate.score
        if best_score is None or score < best_score:
            best_score = score
            best.clear()
        if score == best_score:
            best.append(candidate)
    return sorted(best, key=lambda candidate: candidate.text)


def _load_tokenizer(tokenizer_class, model_id, cache_dir):
    local = Path(cache_dir) / model_id.replace("/", "_") if cache_dir else None
    if local is not None and local.exists():
        return tokenizer_class.from_pretrained(local)
    tokenizer = tokenizer_class.from_pretrained(model_id)
    if local is not None:
        tokenizer.save_pretrained(local)
    return tokenizer


def _parser():
    parser = argparse.ArgumentParser(description="Show token splits for SDXL text encoders (TE1/TE2).")
    parser.add_argument("--text", help='Input tag string, e.g. "grid, circle".')
    parser.add_argument("--tokenizer-cache-dir", help="Load or save each tokenizer in this directory.")
    parser.add_argument("--add-special-tokens", action="store_true", help="Include BOS/EOS in splits and search.")
    parser.add_argument("--search-core", help="Core word to wrap with extra characters.")
    parser.add_argument("--search-side", choices=["prefix", "suffix", "both"], default="both")
    parser.add_argument("--search-min-add", type=int, default=1, help="Use 0 to also try the core itself.")
    parser.add_argument("--search-max-add", type=int, default=3, help="Maximum number of characters to add.")
    parser.add_argument("--search-min-tokens", type=int, default=2, help="Minimum number of tokens in an accepted split.")
    parser.add_argument("--search-alphabet", default="abcdefghijklmnopqrstuvwxyz")
    parser.add_argument("--search-limit", type=int, default=20, help="Maximum number of tied best candidates to print.")
    parser.add_argument("--search-either", action="store_true", help="Allow either TE1 or TE2 to meet the conditions.")
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if not (args.search_core or args.text):
        parser.error("either --text or --search-core is required")

    from library.clip_tokenizer import CLIPTokenizer
    from library.strategy_sdxl import TOKENIZER1_PATH, TOKENIZER2_PATH

    te1 = _load_tokenizer(CLIPTokenizer, TOKENIZER1_PATH, args.tokenizer_cache_dir)
    te2 = _load_tokenizer(CLIPTokenizer, TOKENIZER2_PATH, args.tokenizer_cache_dir)
    te2.pad_token_id = 0  # SDXL uses 0 for the second encoder.
    if args.search_core:
        best = _search_candidates(args, te1, te2)
        if not best:
            print("no candidates found")
            return
        print(f"core: {args.search_core}")
        print(f"side: {args.search_side}")
        print(f"alphabet: {args.search_alphabet}")
        print(f"add_range: {args.search_min_add}..{args.search_max_add}")
        print(f"min_tokens: {args.search_min_tokens}")
        print(f"require_both: {not args.search_either}")
        print()
        print(f"best_candidates: {min(len(best), args.search_limit)}")
        print()
        for candidate in best[: args.search_limit]:
            print(f"text: {candidate.text}  ({candidate.direction} +{candidate.addon})")
            print(f"  TE1: {len(candidate.te1.ids)}  {candidate.te1.boundary}")
            print(f"  TE2: {len(candidate.te2.ids)}  {candidate.te2.boundary}")
            print()
        return

    print(f"input: {args.text}")
    print()
    for label, model_id, tokenizer in (("TE1", TOKENIZER1_PATH, te1), ("TE2", TOKENIZER2_PATH, te2)):
        split = _tokenize(tokenizer, args.text, args.add_special_tokens)
        print(f"{label} ({model_id})")
        print(f"token_count: {len(split.ids)}")
        print(f"normalized: {split.normalized}")
        print(f"boundary: {split.boundary}")
        print("tokens:")
        for i, (token, tid, piece) in enumerate(zip(split.tokens, split.ids, split.pieces)):
            print(f"{i:>3}  token={token}  id={tid}  piece={piece!r}")
        print()


if __name__ == "__main__":
    main()
