import copy
import hashlib
import json
import uuid

from PIL import Image, PngImagePlugin
import pytest

from tools.lora_report_result import comparison_jobs, read_comparison


def comparison(folder):
    folder.mkdir(exist_ok=True, parents=True)
    blob = b'{ "run_name": "fixture" }\n'
    (folder / "config.json").write_bytes(blob)
    conditions = [{"id": "a", "items": []}, {"id": "b", "items": []}]
    jobs = []
    for name in ("a", "b"):
        image = f"images/{name}/sample.png"
        path = folder / image
        path.parent.mkdir(parents=True)
        tags = PngImagePlugin.PngInfo()
        tags.add_text("seed","0")
        tags.add_text("prompt","same")
        Image.new("RGB", (64,64)).save(path, pnginfo=tags)
        jobs.append(dict(job_id=f"job-{name}", case_id="case-1", condition_id=name, prompt_id="prompt-1",
                         prompt_index=1, prompt="same", negative="", seed=0, width=64, height=64, image=image,
                         status="done", returncode=0))
    return dict(kind="lora_image_comparison", schema_version=1, comparison_id=str(uuid.uuid4()),
                attempt_id=str(uuid.uuid4()), status="done", returncode=0, config_sha256=hashlib.sha256(blob).hexdigest(),
                prompts=[{"id":"prompt-1","line_no":1,"prompt":"same"}], seeds=[0], conditions=conditions, jobs=jobs)


def test_read_exact_metadata_bytes_and_config_integrity(tmp_path):
    m=comparison(tmp_path)
    data = (json.dumps(m, indent=3)+"\n").encode()
    (tmp_path/"metadata.json").write_bytes(data)
    raw, sha, jobs = read_comparison(tmp_path)
    assert raw == m and sha == hashlib.sha256(data).hexdigest() and jobs == m["jobs"]
    (tmp_path/"config.json").write_text("{}",encoding="utf-8")
    with pytest.raises(ValueError,match="digest"):
        read_comparison(tmp_path)


def test_only_identified_success_with_live_verified_png(tmp_path):
    m=comparison(tmp_path)
    job=m["jobs"][1]
    job.update(status="failed",returncode=2)
    m.update(status="partial",returncode=2)
    assert comparison_jobs(m,tmp_path)==m["jobs"][:1]
    path=tmp_path/m["jobs"][0]["image"]
    path.write_bytes(b"broken")
    assert comparison_jobs(m,tmp_path)==[]
    path.unlink()
    assert comparison_jobs(m,tmp_path)==[]


@pytest.mark.parametrize("change", ["duplicate_id","case_conflict","traversal","wrong_condition","job_missing","exitcode","unknown_schema"])
def test_invalid_graph_is_rejected(tmp_path, change):
    m=comparison(tmp_path)
    b=m["jobs"][1]
    if change=="duplicate_id": b["job_id"]=m["jobs"][0]["job_id"]
    if change=="case_conflict": b["prompt"]="different"
    if change=="traversal": b["image"]="images/b/../../external.png"
    if change=="wrong_condition": b["image"]="images/a/b.png"
    if change=="job_missing": m["jobs"].pop()
    if change=="exitcode": b["returncode"]=None
    if change=="unknown_schema": m["schema_version"]=2
    with pytest.raises(ValueError):
        comparison_jobs(m,tmp_path)


def test_dry_run_and_actual_dimensions_and_seed(tmp_path):
    m=comparison(tmp_path)
    p=tmp_path/m["jobs"][0]["image"]
    Image.new("RGB",(32,64)).save(p)
    assert comparison_jobs(m,tmp_path)==m["jobs"][1:]
    tags=PngImagePlugin.PngInfo()
    tags.add_text("seed","99")
    Image.new("RGB",(64,64)).save(p,pnginfo=tags)
    assert comparison_jobs(m,tmp_path)==m["jobs"][1:]
    m.update(status="dry_run",returncode=None)
    for j in m["jobs"]: j.update(status="dry_run",returncode=None)
    assert comparison_jobs(m,tmp_path)==[]


def test_saved_png_strength_and_lbw_match_the_generators_complete_slot_set(tmp_path):
    m = comparison(tmp_path)
    job = m["jobs"][0]
    expected = {"network-strength-mode": [3], "network-te1-strength": [0],
                "network-te2-strength": [.5], "network-unet-strength": [1],
                "network-dynamic-slots": [{"slot": 0, "lbw": [1] * 12}]}
    job["png_parameters"] = expected
    path = tmp_path / job["image"]
    tags = PngImagePlugin.PngInfo()
    for key, value in expected.items():
        tags.add_text(key, json.dumps(value))
    Image.new("RGB", (64, 64)).save(path, pnginfo=tags)
    assert comparison_jobs(m, tmp_path) == m["jobs"]
    job["png_parameters"] = {**expected, "network-te2-strength": [.25]}
    assert comparison_jobs(m, tmp_path) == m["jobs"][1:]
    job.pop("png_parameters")
    assert comparison_jobs(m, tmp_path) == m["jobs"][1:]
