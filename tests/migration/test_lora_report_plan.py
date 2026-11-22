import copy
import json
import subprocess
import uuid
from pathlib import Path
from types import SimpleNamespace

from PIL import Image,PngImagePlugin
import pytest

import sdxl_lora_report_cui as cui
import sdxl_lora_report_worker as worker
from tools import lora_report_plan as plan
from tools.lora_report_result import read_comparison


def inputs(base):
    (base/"prompts.txt").write_text("first {red|blue} bird\nfirst {red|blue} bird\n",encoding="utf-8")
    return {"prompt_file":"prompts.txt","output_root":"results","run_name":"study",
            "seeds":{"values":[5],"random_count":1,"random_source_seed":9},"loras":[
                {"id":"A","path":"model.safetensors","strength":[0.2,1e-13,0.5]}],
            "sdxl_gen_img":{"width":128,"height":192,"common_args":["--sdpa","--tokenizer_cache_dir","folder with spaces"]}}


def test_freeze_and_retarget_the_same_cases(tmp_path):
    cfg=inputs(tmp_path)
    fixed=plan.freeze_comparison(cfg,tmp_path)
    assert "_comparison_plan" not in cfg
    snapshots=copy.deepcopy(fixed)
    (tmp_path/"prompts.txt").unlink()
    path1,_,first=plan.make_attempt(fixed,tmp_path,str(uuid.uuid4()))
    path2,_,second=plan.make_attempt(fixed,tmp_path,str(uuid.uuid4()))
    assert first and path1!=path2 and snapshots==fixed
    assert [j["job_id"] for j in first]==[j["job_id"] for j in second]
    assert len({j["case_id"] for j in first})==4
    ordered=plan.order_jobs_for_generation(first,fixed["_comparison_plan"]["conditions"])
    assert [j["display_order"] for j in ordered]==[1,3,5,7,2,4,6,8]
    assert len({(j["prompt"],j["negative"],j["seed"]) for j in first})==2
    assert fixed["sdxl_gen_img"]["common_args"][-1]==str(tmp_path/"folder with spaces")


@pytest.mark.parametrize("bad", ["duplicate","windows","mixed_lbw","stack_order","stack_dup","compound","multi","size","embedded"])
def test_invalid_job_graph_fails_before_worker(tmp_path,bad):
    cfg=inputs(tmp_path)
    a={"path":"a.safetensors"}
    b={"path":"b.safetensors"}
    if bad=="duplicate":
        cfg["seeds"]={"values":[2,2]}
    if bad=="windows":
        cfg["loras"]=[{"id":"Ab","items":[a]},{"id":"ab","items":[b]}]
    if bad=="mixed_lbw":
        cfg["loras"]=[{"id":"s","items":[dict(a,lbw="ALL"),b]}]
    if bad=="stack_order":
        cfg["loras"]=[{"id":"s","items":[a,b]},{"id":"t","items":[b,a]}]
    if bad=="stack_dup":
        cfg["loras"]=[{"id":"s","items":[a,a]}]
    if bad=="compound":
        cfg["loras"][0]["strength"]=[0.3,0.3,0.3,0.3]
    if bad=="multi":
        (tmp_path/"prompts.txt").write_text("{e$$red|blue} bird")
    if bad=="size":
        (tmp_path/"prompts.txt").write_text("id|bird||0|128")
    if bad=="embedded":
        (tmp_path/"prompts.txt").write_text("bird --am 1")
    with pytest.raises(ValueError):
        plan.freeze_comparison(cfg,tmp_path)


def test_prompt_input_axis_overrides_and_dynamic_variants(tmp_path):
    cfg=inputs(tmp_path)
    (tmp_path/"prompts.txt").write_text("id|{red|blue} bird|cloud|64|\n",encoding="utf-8")
    frozen=plan.freeze_comparison(cfg,tmp_path)
    assert all((c["width"],c["height"],c["negative"])==(64,192,"cloud") for c in frozen["_comparison_plan"]["cases"])


def plan_attempt(tmp_path):
    cfg=inputs(tmp_path)
    cfg["seeds"]={"values":[39]}
    _,fixed,jobs=plan.make_attempt(cfg,tmp_path,str(uuid.uuid4()))
    for i,j in enumerate(jobs):
        j["job_index"]=i
    ordered=plan.order_jobs_for_generation(jobs,fixed["_comparison_plan"]["conditions"])
    slots=plan.build_lora_slots(fixed["_comparison_plan"]["conditions"])
    return cfg,ordered,slots


def make_png(path,job,*,wrong=False):
    tags=PngImagePlugin.PngInfo()
    tags.add_text("seed",str(job["seed"]+int(wrong)))
    tags.add_text("prompt",job["prompt"])
    for key,value in job["png_parameters"].items():
        tags.add_text(key,json.dumps(value))
    Image.new("RGB",(job["width"],job["height"])).save(path,pnginfo=tags)


@pytest.mark.parametrize("scenario",["normal","highres","missing","extra","wrong_header"])
def test_images_require_exact_job_numbers_and_headers(tmp_path,scenario):
    _,jobs,slots=plan_attempt(tmp_path)
    out=tmp_path/"worker"
    out.mkdir()
    for i,job in enumerate(jobs):
        index=job["execution_order"]
        if scenario in ("missing","extra") and i==0:
            continue
        highres=scenario=="highres"
        if highres:
            make_png(out/f"im_0{index:06}.png",job)
        make_png(out/f"im_{'1' if highres else ''}{index:06}.png",job,wrong=scenario=="wrong_header" and i==1)
    if scenario=="extra":
        make_png(out/"im_000090.png",jobs[0])
    rows=worker.move_outputs(out,jobs,slots,highres=scenario=="highres",generator_code=0)
    expected=len(jobs) if scenario in ("normal","highres") else 0 if scenario=="extra" else len(jobs)-1
    assert sum(j["status"]=="done" for j in rows)==expected
    assert len({j["job_id"] for j in rows})==len(jobs)


@pytest.mark.parametrize("flag",["--network_merge","--network_mul=0.5","--sequential_file_name","--outd","--use_original_file_name"])
def test_worker_owns_output_and_runtime_options(tmp_path,flag):
    with pytest.raises(ValueError):
        worker.build_command(tmp_path,{"sdxl_gen_img":{"common_args":[flag]}},tmp_path/"p",tmp_path/"o",[])


def test_real_cui_worker_dry_run_excludes_all_images(tmp_path):
    cfg=inputs(tmp_path)
    p=tmp_path/"config.json"
    p.write_text(json.dumps(cfg),encoding="utf-8")
    assert cui.run(SimpleNamespace(config=str(p),dry_run=True,attempt_id=None))==0
    folder=next((tmp_path/"results").iterdir())
    meta,digest,eligible=read_comparison(folder)
    assert meta["status"]=="dry_run" and meta["returncode"] is None and not eligible
    assert all(j["returncode"] is None and j["status"]=="dry_run" for j in meta["jobs"])
    assert not list(folder.rglob("*.png"))
    assert "1e-13" in (folder/"worker/generation-prompts.txt").read_text(encoding="utf-8")


@pytest.mark.parametrize("failure",["missing","duplicate","foreign"])
def test_cui_refuses_uncorrelated_worker_outcomes(tmp_path,monkeypatch,failure):
    _,fixed,jobs=plan.make_attempt(inputs(tmp_path),tmp_path,str(uuid.uuid4()))
    def fake_run(argv,cwd):
        jobfile=Path(argv[argv.index("--job-json")+1])
        resultfile=Path(argv[argv.index("--result-json")+1])
        p=json.loads(jobfile.read_text(encoding="utf-8"))
        if failure!="missing":
            bad=[dict(job_id=j["job_id"],status="done",returncode=0) for j in jobs]
            if failure=="duplicate":
                bad[1]=bad[0]
            if failure=="foreign":
                bad[1]["job_id"]="alien"
            resultfile.write_text(json.dumps(dict(attempt_id=p["attempt_id"],status="done",returncode=0,results=bad)),encoding="utf-8")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(cui.subprocess,"run",fake_run)
    response=cui.worker_result(tmp_path,fixed,jobs,str(uuid.uuid4()),False)
    assert response["status"]=="failed" and all(j["status"]=="failed" for j in jobs)
