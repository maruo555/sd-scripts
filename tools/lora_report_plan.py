"""Resolve comparison inputs to a portable job graph without starting Qt, Torch or a generator."""
import base64
import copy
import csv
import json
import os
import random
import re
import uuid
from pathlib import Path
from library.generation_dynamic_prompt import RE_DYNAMIC_PROMPT, handle_dynamic_prompt_variants
from library.generation_lora_strengths import component_strengths_from_spec, serialize_strength_spec, normalize_strength_spec, flatten_strength_specs, resolve_flat_strengths
from library.generation_lbw import parse_sdxl_lbw_weights

DEFAULT_LBW = "ALL"
REPORT_LORA_MODULE = "networks.lora_lbw"

def sanitize_id(value: str, fallback: str) -> str:
    value = (value or "").strip()
    if not value:
        value = fallback
    value = re.sub(r"[^\w.-]+", "_", value, flags=re.ASCII)
    value = value.strip("._-")
    value = value or fallback
    if value.split(".",1)[0].upper() in {"CON","PRN","AUX","NUL",*(f"COM{i}" for i in range(1,10)),*(f"LPT{i}" for i in range(1,10))}:
        raise ValueError(f"reserved Windows output name: {value}")
    return value


def resolve_path(path: str | None, base_dir: Path) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def parse_prompt_file(path: Path) -> list[dict]:
    if path.suffix.lower() == ".tsv" or looks_like_prompt_tsv(path):
        return parse_prompt_tsv(path)
    return parse_prompt_pipe_text(path)


def looks_like_prompt_tsv(path: Path) -> bool:
    if path.suffix.lower() != ".txt":
        return False
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "\t" not in line:
                    return False
                fields = [normalize_prompt_field(part) for part in next(csv.reader([line], delimiter="\t"))]
                return "prompt" in fields
    except OSError:
        return False
    return False


def parse_prompt_pipe_text(path: Path) -> list[dict]:
    prompts = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            # A vertical bar inside an SDXL dynamic prompt belongs to that prompt.
            fields = re.split(r"(\{[^{}]*\})", line)
            parts = [""]
            for i, field in enumerate(fields):
                if i % 2:
                    parts[-1] += field
                else:
                    head,*tail = field.split("|")
                    parts[-1] += head
                    parts.extend(tail)
            parts = [part.strip() for part in parts]
            if len(parts) > 5:
                raise ValueError(f"{path}:{line_no}: at most five pipe-separated fields are supported")
            if len(parts) == 1:
                prompt_id = f"p{len(prompts) + 1:03d}"
                prompt = parts[0]
                negative = ""
                width = None
                height = None
            else:
                prompt_id = parts[0] or f"p{len(prompts) + 1:03d}"
                prompt = parts[1] if len(parts) > 1 else ""
                negative = parts[2] if len(parts) > 2 else ""
                width = int(parts[3]) if len(parts) > 3 and parts[3] else None
                height = int(parts[4]) if len(parts) > 4 and parts[4] else None

            if not prompt:
                raise ValueError(f"Prompt is empty at {path}:{line_no}")

            prompts.append(
                {
                    "id": sanitize_id(prompt_id, f"p{len(prompts) + 1:03d}"),
                    "prompt": prompt,
                    "negative": negative,
                    "width": width,
                    "height": height,
                    "line_no": line_no,
                }
            )

    if not prompts:
        raise ValueError(f"No prompts found: {path}")
    return prompts


def parse_prompt_tsv(path: Path) -> list[dict]:
    prompts = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        lines = []
        line_numbers = []
        found_header = False
        for line_no, raw_line in enumerate(f, 1):
            if not found_header:
                stripped = raw_line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                found_header = True
            lines.append(raw_line)
            line_numbers.append(line_no)

        reader = csv.DictReader(lines, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"No TSV header found: {path}")
        normalized_fields = {normalize_prompt_field(name): name for name in reader.fieldnames}
        if "prompt" not in normalized_fields:
            raise ValueError(f"TSV prompt file requires a 'prompt' column: {path}")

        for row_offset, row in enumerate(reader, 1):
            row_index = line_numbers[row_offset] if row_offset < len(line_numbers) else row_offset + 1
            if is_empty_prompt_row(row):
                continue
            prompt = cell(row, normalized_fields, "prompt")
            if not prompt or prompt.startswith("#"):
                continue

            prompt_id = cell(row, normalized_fields, "id") or f"p{len(prompts) + 1:03d}"
            negative = cell(row, normalized_fields, "negative")
            width = parse_optional_int(cell(row, normalized_fields, "width"), path, row_index, "width")
            height = parse_optional_int(cell(row, normalized_fields, "height"), path, row_index, "height")

            prompts.append(
                {
                    "id": sanitize_id(prompt_id, f"p{len(prompts) + 1:03d}"),
                    "prompt": prompt,
                    "negative": negative,
                    "width": width,
                    "height": height,
                    "line_no": row_index,
                }
            )

    if not prompts:
        raise ValueError(f"No prompts found: {path}")
    return prompts


def normalize_prompt_field(name: str | None) -> str:
    value = (name or "").strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "prompt_id": "id",
        "negative_prompt": "negative",
        "neg": "negative",
        "w": "width",
        "h": "height",
    }
    return aliases.get(value, value)


def cell(row: dict, fields: dict[str, str], name: str) -> str:
    source = fields.get(name)
    if source is None:
        return ""
    return (row.get(source) or "").strip()


def is_empty_prompt_row(row: dict) -> bool:
    return not any((value or "").strip() for value in row.values())


def parse_optional_int(value: str, path: Path, line_no: int, column: str) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid {column} value at {path}:{line_no}: {value}") from exc


def validate_image_size(width: int | None, height: int | None, source: str):
    if width is None or height is None:
        raise ValueError(f"{source}: width and height are required")
    for name, value in (("width", width), ("height", height)):
        if int(value) <= 0:
            raise ValueError(f"{source}: {name} must be positive: {value}")
        if int(value) % 64 != 0:
            raise ValueError(
                f"{source}: {name} must be a multiple of 64 for SDXL generation: {value}. "
                "Use a nearby value such as 896, 960, or 1024."
            )


def build_seeds(seed_config: dict) -> list[int]:
    values = [int(v) for v in seed_config.get("values", [])]
    negative_values = [value for value in values if value < 0]
    if negative_values:
        raise ValueError(f"Seed values must be non-negative because sdxl_gen_img.py prompt seeds do not accept negatives: {negative_values}")

    random_count = int(seed_config.get("random_count", 0) or 0)
    if random_count < 0:
        raise ValueError("seeds.random_count must be nonnegative")
    if random_count > 0:
        rnd = random.Random(seed_config.get("random_source_seed"))
        min_seed = int(seed_config.get("random_min", 0))
        max_seed = int(seed_config.get("random_max", 2**32 - 1))
        if min_seed < 0:
            raise ValueError(f"seeds.random_min must be non-negative: {min_seed}")
        if max_seed < min_seed:
            raise ValueError(f"seeds.random_max must be greater than or equal to seeds.random_min: {max_seed} < {min_seed}")
        values.extend(rnd.randint(min_seed, max_seed) for _ in range(random_count))
    if not values:
        raise ValueError("At least one seed is required. Set seeds.values or seeds.random_count.")
    if len(set(values)) != len(values):
        raise ValueError("Duplicate comparison seeds")
    return values


def normalize_lora_item(item: dict, config_dir: Path, force_lbw_module: bool, condition_id: str, item_index: int) -> dict:
    path = resolve_path(item.get("path"), config_dir)
    if path is None:
        raise ValueError(f"LoRA condition '{condition_id}' item {item_index} requires path")

    lbw = item.get("lbw")
    if force_lbw_module and lbw is None:
        raise ValueError(f"LoRA condition '{condition_id}' item {item_index} requires lbw when the condition uses LBW")

    module = item.get("module")
    if not module:
        module = "networks.lora_lbw" if force_lbw_module or lbw is not None else "networks.lora"

    return {
        "name": item.get("name") or Path(path).stem,
        "path": str(path),
        "strength": serialize_strength_spec(item.get("strength", 1.0)),
        "lbw": lbw,
        "module": module,
    }


def build_lora_conditions(config: dict, config_dir: Path) -> list[dict]:
    conditions = []
    if config.get("include_baseline", True):
        conditions.append({"id": "baseline", "name": "baseline", "items": []})

    for idx, raw in enumerate(config.get("loras", []), 1):
        condition_id = sanitize_id(raw.get("id") or raw.get("name"), f"lora_{idx:02d}")
        items = raw.get("items")
        if items is None:
            items = [raw]
        if not isinstance(items, list) or len(items) == 0:
            raise ValueError(f"LoRA condition '{condition_id}' requires a non-empty items list")
        force_lbw_module = any(item.get("lbw") is not None for item in items)
        if force_lbw_module and any(item.get("lbw") is None for item in items):
            raise ValueError(f"condition {condition_id!r} must specify LBW for every item in its stack")
        normalized_items = [
            normalize_lora_item(item, config_dir, force_lbw_module, condition_id, item_index)
            for item_index, item in enumerate(items, 1)
        ]
        conditions.append(
            {
                "id": condition_id,
                "name": raw.get("name") or condition_id,
                "items": normalized_items,
            }
        )

    folded = [condition["id"].casefold() for condition in conditions]
    if len(set(folded)) != len(folded):
        raise ValueError("Condition IDs collide on Windows")
    if not conditions:
        raise ValueError("No LoRA conditions found. Add loras or set include_baseline=true.")
    return conditions


def order_jobs_for_generation(jobs: list[dict], conditions: list[dict]) -> list[dict]:
    condition_order = {condition["id"]: index for index, condition in enumerate(conditions)}
    original_order = {id(job): index for index, job in enumerate(jobs)}
    return sorted(
        jobs,
        key=lambda job: (
            condition_order.get(job["condition_id"], len(condition_order)),
            original_order[id(job)],
        ),
    )


def make_image_name(prompt: dict, prompt_index: int, seed: int) -> str:
    return f"p{prompt_index:04d}_{prompt['id']}_seed{seed}.png"


def target_path_key(path: Path) -> str:
    return os.path.normcase(str(path))


def normalize_lora_slot_item(item: dict) -> dict:
    normalized = dict(item)
    module = normalized.get("module")
    lbw = normalized.get("lbw")
    if module in (None, "networks.lora", REPORT_LORA_MODULE):
        normalized["module"] = REPORT_LORA_MODULE
        normalized["lbw"] = lbw or DEFAULT_LBW
    return normalized


def lora_slot_key(item: dict) -> tuple[str, str, tuple]:
    normalized = normalize_lora_slot_item(item)
    weights = parse_sdxl_lbw_weights(normalized.get("lbw")) or ()
    return normalized["module"], os.path.normcase(normalized["path"]), tuple(weights)


def build_lora_slots(conditions: list[dict]) -> list[dict]:
    slots = []
    slot_by_key = {}
    for condition in conditions:
        previous = -1
        for item in condition["items"]:
            key = lora_slot_key(item)
            if key not in slot_by_key:
                slot_by_key[key] = len(slots)
                slots.append(normalize_lora_slot_item(item))
            index = slot_by_key[key]
            if index <= previous:
                raise ValueError(f"condition {condition.get('id', '?')}: repeated slot or order conflicts with previous conditions")
            previous = index
    return slots


def build_lora_slots_from_jobs(jobs: list[dict]) -> list[dict]:
    return build_lora_slots([{"items": job.get("condition_items", [])} for job in jobs])

def freeze_comparison(config, config_dir):
    """Capture prompt text and sampled seeds before submitting a queue item or launching its worker."""
    if "_comparison_plan" in config:
        return config
    config = copy.deepcopy(config)
    for k in ("output_root","prompt_file"):
        if k in config:
            config[k] = str(resolve_path(config[k],config_dir))
    config.setdefault("output_root", str(resolve_path("lora_reports",config_dir)))
    gen=config.setdefault("sdxl_gen_img",{})
    for key in ("ckpt","vae","tokenizer_cache_dir"):
        if gen.get(key):
            gen[key]=str(resolve_path(gen[key],config_dir))
    args=list(gen.get("common_args",[]))
    for i,arg in enumerate(args):
        option,sep,inline=str(arg).partition("=")
        if option in {"--tokenizer_cache_dir","--embeddings","--vae","--ckpt"}:
            if sep:
                args[i]=option+"="+str(resolve_path(inline,config_dir))
            elif i+1<len(args):
                args[i+1]=str(resolve_path(args[i+1],config_dir))
    gen["common_args"]=args
    for condition in config.get("loras",[]):
        for item in condition.get("items",[condition]):
            if item.get("path"):
                item["path"]=str(resolve_path(item["path"],config_dir))
    if not config.get("prompt_file"):
        raise ValueError("prompt_file is required")
    source=Path(config["prompt_file"])
    prompts=parse_prompt_file(source)
    if not prompts:
        raise ValueError(f"{source}: no prompts")
    inputs=source.read_bytes()
    seeds=build_seeds(config.get("seeds",{}))
    conditions=build_lora_conditions(config,config_dir)
    slots=build_lora_slots(conditions)
    by_key={lora_slot_key(item):i for i,item in enumerate(slots,1)}
    for condition in conditions:
        for item in condition["items"]:
            slot=normalize_lora_slot_item(item)
            strength=component_strengths_from_spec(item["strength"])
            item.update(slot=by_key[lora_slot_key(item)], requested_module=item["module"], effective_module=slot["module"],
                        effective_lbw=parse_sdxl_lbw_weights(slot.get("lbw")), strength_width=strength.value_count,
                        effective_strength={"te1":strength.te1,"te2":strength.te2,"unet":strength.unet})
    cid=str(uuid.uuid4())
    config["_comparison_plan"]={
        "comparison_id":cid,"prompts":prompts,"prompt_file_base64":base64.b64encode(inputs).decode("ascii"),
        "seeds":seeds,"conditions":conditions,"cases":[],
    }
    cases=config["_comparison_plan"]["cases"]
    for i,prompt in enumerate(prompts,1):
        width=prompt["width"] if prompt.get("width") is not None else gen.get("width",1024)
        height=prompt["height"] if prompt.get("height") is not None else gen.get("height",1024)
        validate_image_size(width,height,f"{source}:{prompt['line_no']}")
        for seed in seeds:
            rng=random.Random(seed)
            parts=[]
            for key in ["prompt","negative"]:
                txt=prompt.get(key) or ""
                if re.search(r"(?:^|\s)--\w+",txt) or "\n" in txt or "\r" in txt:
                    raise ValueError(f"{source}:{prompt['line_no']}: embedded options and line breaks are not allowed in comparisons")
                expanded=handle_dynamic_prompt_variants(txt,1,rng=rng)
                if len(expanded)!=1 or RE_DYNAMIC_PROMPT.search(expanded[0]):
                    raise ValueError(f"{source}:{prompt['line_no']}: a comparison prompt must yield exactly one image")
                parts.append(expanded[0])
            case_id=str(uuid.uuid5(uuid.UUID(cid),json.dumps([i,*parts,seed,width,height],ensure_ascii=False)))
            cases.append(dict(case_id=case_id,prompt_index=i,prompt_id=prompt["id"],prompt=parts[0],negative=parts[1],
                              seed=seed,width=width,height=height))
    return config


def make_attempt(config,config_dir,attempt_id):
    import datetime as dt
    config=freeze_comparison(config,config_dir)
    frozen=config["_comparison_plan"]
    uid=uuid.UUID(frozen["comparison_id"])
    attempt=uuid.UUID(attempt_id)
    name=sanitize_id(config.get("run_name"),"lora_report")
    timestamp=dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    directory=Path(config["output_root"])/(timestamp+"_"+name+"_"+attempt.hex[:8])
    conditions=frozen["conditions"]
    slots=build_lora_slots(conditions)
    jobs=[]
    for case in frozen["cases"]:
        prompt=frozen["prompts"][case["prompt_index"]-1]
        for condition in conditions:
            image=f"images/{condition['id']}/{make_image_name(prompt,case['prompt_index'],case['seed'])}"
            job_id=str(uuid.uuid5(uid,case["case_id"]+"/"+condition["id"]))
            jobs.append(dict(**case,job_id=job_id,condition_id=condition["id"],condition_name=condition["name"],
                             condition_items=condition["items"],image=image,target_path=str(directory/image),
                             png_parameters=png_parameters(condition["items"],slots),display_order=len(jobs)+1,status="pending",returncode=None))
    for line,job in enumerate(order_jobs_for_generation(jobs,conditions),1):
        job["execution_order"]=line
    return directory,config,jobs


def slot_values(items,slots):
    mapping={lora_slot_key(s):idx for idx,s in enumerate(slots)}
    strengths=[(0.0,)]*len(slots)
    for item in items:
        strengths[mapping[lora_slot_key(item)]]=normalize_strength_spec(item["strength"])
    return flatten_strength_specs(strengths)


def png_parameters(items,slots):
    if not slots:
        return {}
    values=resolve_flat_strengths(slot_values(items,slots),len(slots))
    return {
        "network-strength-mode":[v.value_count for v in values],
        "network-te1-strength":[v.te1 for v in values], "network-te2-strength":[v.te2 for v in values],
        "network-unet-strength":[v.unet for v in values],
        "network-dynamic-slots":[{"slot":i,"lbw":s.get("lbw")} for i,s in enumerate(slots,1)],
    }
