"""Acceptance of the actual HTML in Qt WebEngine (Chromium), including a browser CSV download."""
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

import pytest
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu")
pytest.importorskip("PySide6.QtWebEngineWidgets")
from PySide6.QtCore import QUrl
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QApplication

from tools.lora_report_plan import make_attempt
from tools.lora_report_views import render_comparison


def build_report(root):
    prompt = root / "prompts.txt"
    prompt.write_text("same prompt\nsame prompt\nsame prompt\n", encoding="utf-8")
    config = {"prompt_file": str(prompt), "output_root": str(root), "seeds":{"values":[39]},
              "sdxl_gen_img":{"width":64, "height":64}, "include_baseline":False,
              "loras":[{"id":"a","name":"=A,\"private\"","path":"private-a.safetensors"},
                       {"id":"b","name":"B private","path":"private-b.safetensors"}]}
    directory, frozen, jobs = make_attempt(config,root,str(uuid.uuid4()))
    directory.mkdir()
    for job in jobs:
        path = Path(job.pop("target_path"))
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB",(64,64),"white").save(path)
        job.update(status="done",returncode=0)
    plan = frozen["_comparison_plan"]
    raw_config = json.dumps(config).encode()
    (directory/"config.json").write_bytes(raw_config)
    meta = {k:plan[k] for k in ("comparison_id","conditions","seeds","prompts")}
    meta.update(kind="lora_image_comparison", schema_version=1,attempt_id=str(uuid.uuid4()),status="done",returncode=0,
                jobs=jobs,config_sha256=hashlib.sha256(raw_config).hexdigest())
    raw = json.dumps(meta,ensure_ascii=False).encode()
    render_comparison(directory,raw)
    (directory/"metadata.json").write_bytes(raw)
    return directory, jobs


def wait(app, predicate):
    end = time.monotonic()+15
    while time.monotonic() < end:
        app.processEvents()
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError("Browser operation timed out")


class Browser:
    def __init__(self, directory):
        self.app = QApplication.instance() or QApplication([])
        self.view = QWebEngineView()
        self.profile = QWebEngineProfile(self.view)
        self.page = QWebEnginePage(self.profile,self.view)
        self.view.setPage(self.page)
        self.view.resize(1100,850)
        self.view.show()
        self.downloaded = []
        def accept(request):
            request.setDownloadDirectory(str(directory))
            request.setDownloadFileName("scores.csv")
            self.downloaded.append(request)
            request.accept()
        self.profile.downloadRequested.connect(accept)

    def open(self,path):
        completed = []
        cb = lambda ok: completed.append(ok)
        self.page.loadFinished.connect(cb)
        self.page.setUrl(QUrl.fromLocalFile(str(path)))
        wait(self.app,lambda:bool(completed))
        self.page.loadFinished.disconnect(cb)
        assert completed == [True]

    def js(self, code):
        received = []
        self.page.runJavaScript("JSON.stringify((()=>{"+code+"})())",0,received.append)
        wait(self.app,lambda:bool(received))
        assert received[0], code
        return json.loads(received[0])

    def close(self):
        self.view.close()
        self.page.deleteLater()
        self.app.processEvents()


def test_visible_results_and_blind_download(tmp_path):
    directory, jobs = build_report(tmp_path)
    b = Browser(tmp_path)
    b.open(directory/"report.html")
    assert b.js("return document.querySelectorAll('.card img').length") == 6
    assert b.js("byId('axis').value='case'; byId('axis').dispatchEvent(new Event('change')); return document.querySelectorAll('.card img').length") == 6
    assert b.js("document.querySelector('.card img').click(); return byId('viewer').open")
    b.open(directory/"blind_report.html")
    assert not b.js("return document.body.innerText.includes('private')")
    assert b.js("document.querySelector('.choice img').click(); return byId('viewer').open && !byId('viewerTitle').innerText.includes('private')")
    assert b.js("byId('viewer').close(); const r=document.querySelectorAll('.group:first-child input');r[0].click();r[1].click();byId('reveal').click();return [...document.querySelectorAll('#results tbody tr')].map(x=>+x.cells[1].textContent).sort()") == [0,1]
    b.open(directory/"blind_report_detailed.html")
    assert not b.js("return document.body.innerText.includes('private') || [...document.images].some(x=>x.alt.includes('private'))")
    assert b.js("return byId('export').disabled && summarize().every(r=>r.average===null)")
    assert b.js("document.querySelector('.choice img').click();return byId('viewer').open && !byId('viewerTitle').innerText.includes('private')")
    b.js("""byId('viewer').close();
    const card=i=>document.querySelector('article[data-job-index="'+i+'"]');
    [3,2,1,2,null,null].forEach((v,i)=>{if(v)card(i).querySelector('input[value="'+v+'"]').click()});
    [0,4].forEach(i=>card(i).querySelector('[data-flag="faithful"]').click());
    byId('reveal').click(); return true;""")
    expected = {"a":[3,2,4,2,2,1],"b":[3,2,4,2,0,1]}
    summary = "return Object.fromEntries(summarize().map(x=>[x.id,[x.images,x.rated,x.total,x.average,x.faithful,x.rank]]))"
    assert b.js(summary) == expected
    b.open(directory/"blind_report_detailed.html")
    assert b.js(summary) == expected
    image = directory/jobs[0]["image"]
    image.unlink()
    b.js("""const img=document.querySelector('.choice img[data-job-index="0"]'); img.loading='eager'; img.src+='?removed=1';return true""")
    wait(b.app,lambda:b.js("return failedImages.size") == 1)
    assert b.js(summary)["a"] == [2,1,1,1,1,2]
    # For an already-open document Chromium applies profile storage settings only on navigation.
    # Reproduce a browser refusing a write (private policy, quota) without discarding current ratings.
    b.js("Storage.prototype.setItem=function(){throw new DOMException('Storage unavailable','QuotaExceededError')};return true")
    b.js("""const cb=document.querySelector('article[data-job-index="4"] [data-flag="flexible"]');cb.click();byId('reveal').click();byId('export').click();return true;""")
    dest = tmp_path/"scores.csv"
    wait(b.app,lambda:bool(b.downloaded) and dest.is_file() and b.downloaded[-1].isFinished())
    assert dest.read_bytes().startswith(b"\xef\xbb\xbf")
    with dest.open(encoding="utf-8-sig",newline="") as stream:
        header,*rows = csv.reader(stream)
    assert len(header)==12 and all(len(row)==12 for row in rows)
    assert any(row[1] == "'=A,\"private\"" and row[2:6]==["1","2","1","1.00"] for row in rows)
    assert b.js("return byId('storageStatus').innerText.includes('保存できません')")
    other,_ = build_report(tmp_path)
    b.open(other/"blind_report_detailed.html")
    assert b.js("return summarize().every(r=>r.rated===0 && r.average===null)")
    b.close()
