import argparse
import itertools
import json
import os
import random
import sys
from typing import Any, Dict, List

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from library import self_distill_cache


def _load_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def _compose(parts: List[str]) -> str:
    return ", ".join([part for part in parts if part])


def _support_subset(tags: List[str], rng: random.Random, max_per_prompt: int) -> List[str]:
    if not tags:
        return []
    max_pick = min(len(tags), max_per_prompt)
    target = rng.randint(0, max_pick)
    shuffled = tags[:]
    rng.shuffle(shuffled)
    return shuffled[:target]


def _build_templates(args, rng: random.Random) -> List[str]:
    carriers = _load_list(args.carrier_families)
    shots = _load_list(args.shot_types)
    lightings = _load_list(args.lighting_envs)
    templates = []
    for carrier, shot, lighting in itertools.product(carriers, shots, lightings):
        templates.append(_compose([carrier, shot, lighting, args.template_suffix]))
    if args.num_templates and len(templates) > args.num_templates:
        rng.shuffle(templates)
        templates = templates[: args.num_templates]
    return templates


def build_prompt_bank(args: argparse.Namespace) -> Dict[str, Any]:
    rng = random.Random(args.prompt_seed)
    keep_triggers = _load_list(args.keep_triggers)
    suppress_triggers = _load_list(args.suppress_triggers)
    support_tags = _load_list(args.support_tags)
    frontier_tags = _load_list(args.frontier_tags)
    seeds = [int(seed) for seed in _load_list(args.seed_list)]
    templates = _build_templates(args, rng)

    num_holdout = int(round(len(templates) * float(args.holdout_ratio)))
    holdout_templates = set(rng.sample(templates, k=min(num_holdout, len(templates)))) if num_holdout > 0 else set()

    metadata_variants = {
        "keep_strong": {"conditioning_source": "teacher", "loss_role": "keep"},
        "keep_weak": {"conditioning_source": "teacher", "loss_role": "keep"},
        "off_null": {"conditioning_source": "base", "loss_role": "off"},
        "frontier": {"conditioning_source": "teacher", "loss_role": "keep"},
    }
    for suppress_trigger in suppress_triggers:
        metadata_variants[f"suppress_trigger_{suppress_trigger}"] = {
            "conditioning_source": args.suppress_conditioning_source,
            "loss_role": "suppress",
        }

    records = []
    template_records = []
    for template_index, template in enumerate(templates):
        split = "holdout" if template in holdout_templates else "train"
        for seed in seeds:
            support_subset = _support_subset(support_tags, rng, args.max_support_tags_per_prompt)
            frontier_tag = frontier_tags[(template_index + seed) % len(frontier_tags)] if frontier_tags else ""
            variants = {
                "keep_strong": _compose(keep_triggers + support_subset + [template]),
                "keep_weak": _compose(keep_triggers + [template]),
                "off_null": _compose([template]),
                "frontier": _compose(keep_triggers + ([frontier_tag] if frontier_tag else []) + [template]),
            }
            for suppress_trigger in suppress_triggers:
                variants[f"suppress_trigger_{suppress_trigger}"] = _compose([suppress_trigger, template])

            for variant_type, prompt_text in variants.items():
                variant_meta = metadata_variants[variant_type]
                record = {
                    "record_id": f"pb_{template_index:05d}_{seed}_{variant_type}",
                    "template_index": template_index,
                    "template": template,
                    "split": split,
                    "seed": seed,
                    "variant_type": variant_type,
                    "conditioning_source": variant_meta["conditioning_source"],
                    "loss_role": variant_meta["loss_role"],
                    "prompt_text": prompt_text,
                    "negative_prompt": args.negative_prompt,
                    "support_tags": support_subset,
                    "frontier_tag": frontier_tag,
                    "generation_settings": {
                        "width": args.width,
                        "height": args.height,
                        "sample_steps": args.sample_steps,
                        "scale": args.guidance_scale,
                        "sample_sampler": args.sample_sampler,
                        "negative_prompt": args.negative_prompt,
                        "prediction_target": args.prediction_target,
                    },
                }
                records.append(record)
                template_records.append({"template": template, "split": split})

    payload = {
        "version": self_distill_cache.PROMPT_BANK_VERSION,
        "metadata": {
            "keep_triggers": keep_triggers,
            "suppress_triggers": suppress_triggers,
            "support_tags": support_tags,
            "frontier_tags": frontier_tags,
            "carrier_families": _load_list(args.carrier_families),
            "shot_types": _load_list(args.shot_types),
            "lighting_envs": _load_list(args.lighting_envs),
            "seed_list": seeds,
            "num_templates": len(templates),
            "holdout_ratio": float(args.holdout_ratio),
            "prediction_target": args.prediction_target,
            "variant_definitions": metadata_variants,
            "variant_quota": self_distill_cache.parse_mapping_arg(args.variant_quota),
            "template_hash": self_distill_cache.object_sha256(template_records),
        },
        "records": records,
    }
    return payload


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--keep_triggers", type=str, required=True)
    parser.add_argument("--suppress_triggers", type=str, default="")
    parser.add_argument("--support_tags", type=str, default="")
    parser.add_argument("--frontier_tags", type=str, default="")
    parser.add_argument("--carrier_families", type=str, required=True)
    parser.add_argument("--shot_types", type=str, required=True)
    parser.add_argument("--lighting_envs", type=str, required=True)
    parser.add_argument("--seed_list", type=str, required=True)
    parser.add_argument("--template_suffix", type=str, default="")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--num_templates", type=int, default=72)
    parser.add_argument("--max_support_tags_per_prompt", type=int, default=2)
    parser.add_argument("--holdout_ratio", type=float, default=0.18)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--sample_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--sample_sampler", type=str, default="euler_a")
    parser.add_argument("--prediction_target", type=str, choices=["eps", "v"], default="eps")
    parser.add_argument("--variant_quota", type=str, default="")
    parser.add_argument("--suppress_conditioning_source", type=str, choices=["teacher", "base"], default="teacher")
    parser.add_argument("--prompt_seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    prompt_bank = build_prompt_bank(args)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(prompt_bank, f, ensure_ascii=False, indent=2)
