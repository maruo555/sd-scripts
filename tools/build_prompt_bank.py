import argparse
import itertools
import os
import random
import sys
import json
from typing import Any, Dict, List

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

PROMPT_BANK_VERSION = 1


def _load_list_arg(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _choose_support_tags(support_tags: List[str], rng: random.Random, max_tags: int) -> List[str]:
    if not support_tags:
        return []
    count = min(len(support_tags), max_tags)
    target = rng.randint(1, count)
    shuffled = support_tags[:]
    rng.shuffle(shuffled)
    return shuffled[:target]


def _compose_prompt(parts: List[str]) -> str:
    return ", ".join([part for part in parts if part])


def build_prompt_bank(args: argparse.Namespace) -> Dict[str, Any]:
    rng = random.Random(args.prompt_seed)

    support_tags = _load_list_arg(args.support_tags)
    frontier_tags = _load_list_arg(args.frontier_tags)
    carriers = _load_list_arg(args.carrier_families)
    shots = _load_list_arg(args.shot_types)
    lightings = _load_list_arg(args.lighting_envs)
    seeds = [int(seed) for seed in _load_list_arg(args.seed_list)]

    templates = []
    for carrier, shot, lighting in itertools.product(carriers, shots, lightings):
        template = _compose_prompt([carrier, shot, lighting, args.template_suffix])
        templates.append(template)

    if args.num_templates and len(templates) > args.num_templates:
        rng.shuffle(templates)
        templates = templates[: args.num_templates]

    records = []
    record_index = 0
    for template in templates:
        for seed in seeds:
            support_subset = _choose_support_tags(support_tags, rng, args.max_support_tags_per_prompt)
            frontier_tag = frontier_tags[record_index % len(frontier_tags)] if frontier_tags else ""

            variants = {
                "strong": _compose_prompt([args.trigger_token] + support_subset + [template]),
                "weak": _compose_prompt([args.trigger_token, template]),
                "off": template,
                "support_only": _compose_prompt(support_subset + [template]) if support_subset else template,
                "frontier": _compose_prompt([args.trigger_token, frontier_tag, template]) if frontier_tag else _compose_prompt([args.trigger_token, template]),
            }

            for variant_type, prompt_text in variants.items():
                records.append(
                    {
                        "record_id": f"pb_{record_index:05d}_{variant_type}",
                        "template_index": record_index,
                        "seed": seed,
                        "variant_type": variant_type,
                        "prompt_text": prompt_text,
                        "negative_prompt": args.negative_prompt,
                        "support_tags": support_subset,
                        "frontier_tag": frontier_tag,
                        "template": template,
                        "generation_settings": {
                            "width": args.width,
                            "height": args.height,
                            "sample_steps": args.sample_steps,
                            "scale": args.guidance_scale,
                            "sample_sampler": args.sample_sampler,
                            "negative_prompt": args.negative_prompt,
                        },
                    }
                )
            record_index += 1

    payload = {
        "version": PROMPT_BANK_VERSION,
        "metadata": {
            "trigger_token": args.trigger_token,
            "support_tags": support_tags,
            "frontier_tags": frontier_tags,
            "carrier_families": carriers,
            "shot_types": shots,
            "lighting_envs": lightings,
            "seed_list": seeds,
            "num_templates": len(templates),
        },
        "records": records,
    }
    return payload


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True, help="output JSON path")
    parser.add_argument("--trigger_token", type=str, required=True, help="single trigger token")
    parser.add_argument("--support_tags", type=str, default="", help="comma separated support tags")
    parser.add_argument("--frontier_tags", type=str, default="", help="comma separated frontier tags")
    parser.add_argument("--carrier_families", type=str, required=True, help="comma separated carrier families")
    parser.add_argument("--shot_types", type=str, required=True, help="comma separated shot types")
    parser.add_argument("--lighting_envs", type=str, required=True, help="comma separated lighting environments")
    parser.add_argument("--seed_list", type=str, required=True, help="comma separated seeds")
    parser.add_argument("--template_suffix", type=str, default="", help="extra suffix appended to every template")
    parser.add_argument("--negative_prompt", type=str, default="", help="negative prompt")
    parser.add_argument("--num_templates", type=int, default=96, help="cap templates after cartesian expansion")
    parser.add_argument("--max_support_tags_per_prompt", type=int, default=2, help="max support tags in strong prompts")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--sample_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--sample_sampler", type=str, default="euler_a")
    parser.add_argument("--prompt_seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    prompt_bank = build_prompt_bank(args)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(prompt_bank, f, ensure_ascii=False, indent=2)
