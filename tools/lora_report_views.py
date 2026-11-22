"""Render normal and blind comparisons from the single published result."""
import hashlib
import html
import json
from pathlib import Path

def json_for_script(data: dict) -> str:
    return (
        json.dumps(data, ensure_ascii=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def write_report(output_dir: Path, metadata: dict):
    data_json = json_for_script(metadata)
    report = f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SDXL LoRA Report</title>
<style>
:root {{ color-scheme: light dark; --image-width: 320px; --card-width: calc(var(--image-width) + 18px); }}
body {{ margin: 0; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; background: #f5f5f2; color: #202124; }}
header {{ position: sticky; top: 0; z-index: 3; background: #ffffffee; border-bottom: 1px solid #d8d8d0; padding: 12px 18px; backdrop-filter: blur(8px); }}
h1 {{ font-size: 18px; margin: 0 0 10px; }}
.toolbar {{ display: flex; flex-wrap: wrap; gap: 14px; align-items: center; font-size: 13px; }}
.panel {{ padding: 12px 18px; border-bottom: 1px solid #ddd; background: #fbfbf8; }}
.filter-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
.filter-box {{ max-height: 150px; overflow: auto; border: 1px solid #d6d6ce; padding: 8px; background: white; }}
.filter-box label {{ display: block; margin: 3px 0; white-space: nowrap; }}
main {{ padding: 18px; }}
.group {{ margin: 0 0 18px; background: white; border: 1px solid #d8d8d0; box-shadow: 0 1px 3px #0001; }}
.group h2 {{ margin: 0; padding: 10px 12px; font-size: 13px; background: #ecece6; border-bottom: 1px solid #d8d8d0; }}
.cards {{ display: flex; flex-wrap: wrap; align-items: flex-start; gap: 14px; padding: 14px; }}
.card {{ width: var(--card-width); border: 1px solid #d8d8d0; background: #fbfbf8; padding: 8px; box-sizing: border-box; }}
.card-title {{ width: var(--image-width); margin: 0 0 6px; font-size: 12px; font-weight: 650; overflow-wrap: anywhere; }}
.card img {{ width: var(--image-width); max-width: none; height: auto; display: block; cursor: zoom-in; border: 1px solid #ccc; background: #eee; box-sizing: border-box; }}
.missing {{ width: var(--image-width); min-height: 120px; display: grid; place-items: center; border: 1px dashed #aaa; color: #777; font-size: 12px; box-sizing: border-box; }}
.meta {{ margin-top: 6px; font-size: 11px; color: #555; line-height: 1.35; }}
dialog {{ max-width: 96vw; max-height: 96vh; border: 0; padding: 0; background: transparent; }}
.viewer-card {{ margin: 0; color: #eee; }}
#viewerTitle {{ padding: 8px 10px; font-size: 13px; font-weight: 650; background: #111; overflow-wrap: anywhere; }}
dialog img {{ max-width: 96vw; max-height: 88vh; display: block; background: #111; }}
#viewerMeta {{ padding: 7px 10px; font-size: 12px; background: #111; color: #ccc; }}
dialog::backdrop {{ background: rgba(0,0,0,.78); }}
button, select, input {{ font: inherit; }}
@media (prefers-color-scheme: dark) {{
  body {{ background: #1f211f; color: #eee; }}
  header, .panel, .group, .card, .filter-box {{ background: #282b28; }}
  .group h2 {{ background: #343832; }}
  header, .panel, .filter-box, .group, .group h2, .card {{ border-color: #474b43; }}
  .meta {{ color: #bbb; }}
}}
</style>
</head>
<body>
<header>
  <h1>SDXL LoRA Report</h1>
  <div class="toolbar">
    <label>Axis <select id="axis"><option value="condition">X: LoRA / Y: Prompt+Seed</option><option value="case">X: Prompt+Seed / Y: LoRA</option></select></label>
    <label>Image size <input id="size" type="range" min="160" max="768" step="16" value="320"> <span id="sizeLabel">320px</span></label>
    <button id="showAll">Show all</button>
    <a href="blind_report.html">ブラインド評価</a>
    <a href="blind_report_detailed.html">詳細ブラインド評価</a>
  </div>
</header>
<section class="panel">
  <div class="filter-grid">
    <div><strong>LoRA</strong><div id="conditionFilters" class="filter-box"></div></div>
    <div><strong>Prompt</strong><div id="promptFilters" class="filter-box"></div></div>
    <div><strong>Seed</strong><div id="seedFilters" class="filter-box"></div></div>
  </div>
</section>
<main id="report"></main>
<dialog id="viewer"><figure class="viewer-card"><figcaption id="viewerTitle"></figcaption><img id="viewerImage" alt=""><div id="viewerMeta"></div></figure></dialog>
<script>
const reportData = {data_json};
const state = {{ axis: "condition", conditions: new Set(), prompts: new Set(), seeds: new Set(), size: 320 }};
let renderedJobs = [];
let renderedJobMap = new Map();
let viewerJobs = [];
let viewerIndex = -1;
const byId = id => document.getElementById(id);
const esc = value => String(value ?? "").replace(/[&<>"']/g, ch => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[ch]));
const strengthLabel = value => Array.isArray(value) ? value.join(" / ") : value;
const promptPosition = job => `p${{String(job.prompt_index ?? 0).padStart(4, "0")}}`;
const caseId = job => `${{promptPosition(job)}} / ${{job.prompt_id}} / seed ${{job.seed}}`;
const jobKey = job => job.job_id;
const conditionOrder = new Map(reportData.conditions.map((condition, index) => [condition.id, index]));
function initState() {{
  reportData.conditions.forEach(c => state.conditions.add(c.id));
  reportData.prompts.forEach(p => state.prompts.add(p.id));
  reportData.seeds.forEach(s => state.seeds.add(String(s)));
}}
function checkboxList(root, items, selected, labelFn) {{
  root.innerHTML = items.map(item => {{
    const id = String(item.id ?? item);
    return `<label><input type="checkbox" data-id="${{esc(id)}}" checked> ${{esc(labelFn(item))}}</label>`;
  }}).join("");
  root.querySelectorAll("input").forEach(input => {{
    input.addEventListener("change", () => {{
      if (input.checked) selected.add(input.dataset.id); else selected.delete(input.dataset.id);
      render();
    }});
  }});
}}
function selectedJobs() {{
  return reportData.jobs.filter(j => state.conditions.has(j.condition_id) && state.prompts.has(j.prompt_id) && state.seeds.has(String(j.seed)));
}}
function card(job, title) {{
  if (!job) return `<article class="card"><div class="card-title">${{esc(title || "missing")}}</div><div class="missing">no job</div></article>`;
  const image = job.status === "done"
    ? `<img src="${{esc(job.image)}}" alt="" loading="lazy" data-job-key="${{esc(jobKey(job))}}">`
    : `<div class="missing">${{esc(job.status)}}</div>`;
  const items = (job.condition_items || []).map(item => `${{esc(item.name || item.path)}} x${{esc(strengthLabel(item.strength))}} lbw=${{esc(item.lbw ?? "")}}`).join("<br>");
  return `<article class="card"><div class="card-title">${{esc(title || job.condition_name)}}</div>${{image}}<div class="meta">${{items}}<br>${{esc(caseId(job))}}<br>${{esc(job.width)}}x${{esc(job.height)}}</div></article>`;
}}
function setImageSize(value) {{
  state.size = Number(value);
  document.documentElement.style.setProperty("--image-width", `${{state.size}}px`);
  byId("sizeLabel").textContent = `${{state.size}}px`;
}}
function openViewer(key) {{
  const job = renderedJobMap.get(key);
  if (!job) return;
  viewerJobs = renderedJobs
    .filter(item => item.status === "done" && item.case_id === job.case_id)
    .sort((a, b) => (conditionOrder.get(a.condition_id) ?? 9999) - (conditionOrder.get(b.condition_id) ?? 9999));
  viewerIndex = Math.max(0, viewerJobs.findIndex(item => jobKey(item) === key));
  updateViewer();
  byId("viewer").showModal();
}}
function updateViewer() {{
  const job = viewerJobs[viewerIndex];
  if (!job) return;
  byId("viewerImage").src = job.image;
  byId("viewerTitle").textContent = job.condition_name || "";
  byId("viewerMeta").textContent = `${{caseId(job)}} / ${{job.width}}x${{job.height}}`;
}}
function moveViewer(delta) {{
  if (!byId("viewer").open || viewerJobs.length === 0) return;
  viewerIndex = (viewerIndex + delta + viewerJobs.length) % viewerJobs.length;
  updateViewer();
}}
function render() {{
  setImageSize(state.size);
  const jobs = selectedJobs();
  renderedJobs = jobs;
  renderedJobMap = new Map(jobs.map(j => [jobKey(j), j]));
  const report = byId("report");
  const axis = state.axis;
  const jobMap = new Map();
  for (const job of jobs) {{
    if (!jobMap.has(job.case_id)) jobMap.set(job.case_id, new Map());
    jobMap.get(job.case_id).set(job.condition_id, job);
  }}
  const cases = [...jobMap.keys()];
  const conditions = reportData.conditions.filter(c => state.conditions.has(c.id));
  let html = "";
  if (axis === "condition") {{
    for (const caze of cases) {{
      html += `<section class="group"><h2>${{esc(caseId(jobMap.get(caze).values().next().value))}}</h2><div class="cards">`;
      for (const condition of conditions) html += card(jobMap.get(caze).get(condition.id), condition.name);
      html += "</div></section>";
    }}
  }} else {{
    for (const condition of conditions) {{
      html += `<section class="group"><h2>${{esc(condition.name)}}</h2><div class="cards">`;
      for (const caze of cases) html += card(jobMap.get(caze).get(condition.id), condition.name);
      html += "</div></section>";
    }}
  }}
  report.innerHTML = html;
  report.querySelectorAll("img").forEach(img => img.addEventListener("error", () => {{
    renderedJobMap.get(img.dataset.jobKey).status = "missing";
    img.replaceWith(document.createTextNode("画像なし"));
  }}));
  report.querySelectorAll("img").forEach(img => img.addEventListener("click", () => {{
    openViewer(img.dataset.jobKey);
  }}));
}}
initState();
checkboxList(byId("conditionFilters"), reportData.conditions, state.conditions, item => item.name);
checkboxList(byId("promptFilters"), reportData.prompts, state.prompts, item => item.id);
checkboxList(byId("seedFilters"), reportData.seeds, state.seeds, item => item);
byId("axis").addEventListener("change", event => {{ state.axis = event.target.value; render(); }});
byId("size").addEventListener("input", event => {{ setImageSize(event.target.value); render(); }});
byId("showAll").addEventListener("click", () => location.reload());
byId("viewer").addEventListener("click", () => byId("viewer").close());
document.addEventListener("keydown", event => {{
  if (!byId("viewer").open) return;
  if (event.key === "ArrowLeft") {{ event.preventDefault(); moveViewer(-1); }}
  if (event.key === "ArrowRight") {{ event.preventDefault(); moveViewer(1); }}
}});
render();
</script>
</body>
</html>
"""
    with (output_dir / "report.html").open("w", encoding="utf-8") as f:
        f.write(report)


def write_blind_report(output_dir: Path, metadata: dict):
    data_json = json_for_script(metadata)
    report = """<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SDXL LoRA Blind Report</title>
<style>
:root { color-scheme: light dark; --image-width: 320px; --card-width: calc(var(--image-width) + 22px); }
body { margin: 0; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; background: #f5f5f2; color: #202124; }
header { position: sticky; top: 0; z-index: 3; background: #ffffffee; border-bottom: 1px solid #d8d8d0; padding: 12px 18px; backdrop-filter: blur(8px); }
h1 { font-size: 18px; margin: 0 0 10px; }
.toolbar { display: flex; flex-wrap: wrap; gap: 14px; align-items: center; font-size: 13px; }
main { padding: 18px; }
.group { margin: 0 0 22px; background: white; border: 1px solid #d8d8d0; box-shadow: 0 1px 3px #0001; }
.group h2 { margin: 0; padding: 10px 12px; font-size: 13px; background: #ecece6; border-bottom: 1px solid #d8d8d0; }
.choices { display: flex; flex-wrap: wrap; align-items: flex-start; gap: 14px; padding: 14px; }
.choice { width: var(--card-width); border: 1px solid #d8d8d0; background: #fbfbf8; padding: 10px; box-sizing: border-box; }
.choice img { width: var(--image-width); max-width: none; height: auto; display: block; cursor: zoom-in; border: 1px solid #ccc; background: #eee; box-sizing: border-box; }
.choice label { display: inline-flex; gap: 7px; align-items: center; margin-top: 8px; font-weight: 650; }
.missing { width: var(--image-width); min-height: 140px; display: grid; place-items: center; border: 1px dashed #aaa; color: #777; font-size: 12px; box-sizing: border-box; }
.answer { display: none; margin-top: 7px; font-size: 12px; color: #555; line-height: 1.35; }
body.revealed .answer { display: block; }
#results { margin: 0 18px 22px; padding: 12px; border: 1px solid #d8d8d0; background: white; display: none; }
#results table { border-collapse: collapse; }
#results th, #results td { border: 1px solid #d8d8d0; padding: 6px 10px; text-align: left; }
dialog { max-width: 96vw; max-height: 96vh; border: 0; padding: 0; background: transparent; }
.viewer-card { margin: 0; color: #eee; }
#viewerTitle { padding: 8px 10px; font-size: 13px; font-weight: 650; background: #111; overflow-wrap: anywhere; }
dialog img { max-width: 96vw; max-height: 88vh; display: block; background: #111; }
dialog::backdrop { background: rgba(0,0,0,.78); }
button, input { font: inherit; }
@media (prefers-color-scheme: dark) {
  body { background: #1f211f; color: #eee; }
  header, .group, .choice, #results { background: #282b28; }
  .group h2 { background: #343832; }
  header, .group, .group h2, .choice, #results, #results th, #results td { border-color: #474b43; }
  .answer { color: #bbb; }
}
</style>
</head>
<body>
<header>
  <h1>SDXL LoRA Blind Report</h1>
  <div class="toolbar">
    <label>Image size <input id="size" type="range" min="160" max="768" step="16" value="320"> <span id="sizeLabel">320px</span></label>
    <button id="reveal">Reveal / 答え合わせ</button>
  </div>
</header>
<section id="results"></section>
<main id="report"></main>
<dialog id="viewer"><figure class="viewer-card"><figcaption id="viewerTitle"></figcaption><img id="viewerImage" alt=""></figure></dialog>
<script>
const reportData = __REPORT_DATA__;
let renderedGroups = [];
let viewerJobs = [];
let viewerIndex = -1;
const byId = id => document.getElementById(id);
const esc = value => String(value ?? "").replace(/[&<>"']/g, ch => {
  if (ch === "&") return "&amp;";
  if (ch === "<") return "&lt;";
  if (ch === ">") return "&gt;";
  if (ch === '"') return "&quot;";
  return "&#39;";
});
const strengthLabel = value => Array.isArray(value) ? value.join(" / ") : value;
const promptPosition = job => `p${String(job.prompt_index ?? 0).padStart(4, "0")}`;
const caseId = job => `${promptPosition(job)} / ${job.prompt_id} / seed ${job.seed}`;
function shuffle(items) {
  const result = [...items];
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}
function conditionLabel(job) {
  const items = (job.condition_items || []).map(item => `${esc(item.name || item.path)} x${esc(strengthLabel(item.strength))} lbw=${esc(item.lbw ?? "")}`).join("<br>");
  return `${esc(job.condition_name)}${items ? "<br>" + items : ""}`;
}
function buildGroups() {
  const groups = new Map();
  for (const job of reportData.jobs) {
    if (!groups.has(job.case_id)) groups.set(job.case_id, []);
    groups.get(job.case_id).push(job);
  }
  return [...groups.entries()].map(([label, jobs], index) => ({ label: caseId(jobs[0]), index, jobs: shuffle(jobs) }));
}
function choice(job, groupIndex, choiceIndex) {
  const image = job.status === "done"
    ? `<img src="${esc(job.image)}" alt="" loading="lazy" data-group-index="${groupIndex}" data-choice-index="${choiceIndex}">`
    : `<div class="missing">${esc(job.status)}</div>`;
  return `<article class="choice" data-condition="${esc(job.condition_id)}">
    ${image}
    <label><input type="radio" name="best_${groupIndex}" value="${esc(job.condition_id)}" ${job.status==="done"?"":"disabled"}> Best</label>
    <div class="answer">#${choiceIndex + 1}<br>${conditionLabel(job)}</div>
  </article>`;
}
function setImageSize(value) {
  document.documentElement.style.setProperty("--image-width", `${value}px`);
  byId("sizeLabel").textContent = `${value}px`;
}
function openViewer(groupIndex, choiceIndex) {
  const group = renderedGroups[groupIndex];
  if (!group) return;
  viewerJobs = group.jobs.filter(job => job.status === "done");
  const clickedJob = group.jobs[choiceIndex];
  viewerIndex = Math.max(0, viewerJobs.findIndex(job => job === clickedJob));
  updateViewer();
  byId("viewer").showModal();
}
function updateViewer() {
  const job = viewerJobs[viewerIndex];
  if (!job) return;
  byId("viewerImage").src = job.image;
  byId("viewerTitle").textContent = caseId(job);
}
function moveViewer(delta) {
  if (!byId("viewer").open || viewerJobs.length === 0) return;
  viewerIndex = (viewerIndex + delta + viewerJobs.length) % viewerJobs.length;
  updateViewer();
}
function render() {
  setImageSize(byId("size").value);
  renderedGroups = buildGroups();
  byId("report").innerHTML = renderedGroups.map(group => `
    <section class="group">
      <h2>${esc(group.label)}</h2>
      <div class="choices">${group.jobs.map((job, index) => choice(job, group.index, index)).join("")}</div>
    </section>
  `).join("");
  byId("report").querySelectorAll("img").forEach(img => img.addEventListener("error", () => {
    const radio = img.closest(".choice").querySelector("input");
    const job = renderedGroups[Number(img.dataset.groupIndex)].jobs[Number(img.dataset.choiceIndex)];
    job.status = "missing";
    radio.checked = false;
    radio.disabled = true;
    img.replaceWith(document.createTextNode("画像なし"));
    if (document.body.classList.contains("revealed")) reveal();
  }));
  byId("report").querySelectorAll("img").forEach(img => img.addEventListener("click", () => {
    openViewer(Number(img.dataset.groupIndex), Number(img.dataset.choiceIndex));
  }));
}
function reveal() {
  document.body.classList.add("revealed");
  const votes = new Map(reportData.conditions.map(condition => [condition.id, { name: condition.name, count: 0 }]));
  document.querySelectorAll('input[type="radio"]:checked:not(:disabled)').forEach(input => {
    if (votes.has(input.value)) votes.get(input.value).count += 1;
  });
  const rows = [...votes.values()].sort((a, b) => b.count - a.count || a.name.localeCompare(b.name));
  byId("results").style.display = "block";
  byId("results").innerHTML = `<strong>Vote results</strong>
    <table><thead><tr><th>LoRA</th><th>Votes</th></tr></thead><tbody>
    ${rows.map(row => `<tr><td>${esc(row.name)}</td><td>${row.count}</td></tr>`).join("")}
    </tbody></table>`;
}
byId("size").addEventListener("input", event => {
  setImageSize(event.target.value);
});
byId("reveal").addEventListener("click", reveal);
byId("viewer").addEventListener("click", () => byId("viewer").close());
document.addEventListener("keydown", event => {
  if (!byId("viewer").open) return;
  if (event.key === "ArrowLeft") { event.preventDefault(); moveViewer(-1); }
  if (event.key === "ArrowRight") { event.preventDefault(); moveViewer(1); }
});
render();
</script>
</body>
</html>
""".replace("__REPORT_DATA__", data_json)
    with (output_dir / "blind_report.html").open("w", encoding="utf-8") as f:
        f.write(report)


def write_detailed_blind_report(output_dir: Path, metadata: dict, digest: str):
    data_json = json_for_script(metadata)
    template_path = Path(__file__).resolve().parent.parent / "lora_report_templates" / "blind_report_detailed.html"
    report = template_path.read_text(encoding="utf-8")
    # Keep saved ratings separate for every report and metadata revision.
    storage_key = "sdxl-lora-detailed-v1:" + metadata["comparison_id"] + ":" + digest
    report = report.replace("__STORAGE_KEY__", json.dumps(storage_key)).replace("__REPORT_DATA__", data_json)
    (output_dir / "blind_report_detailed.html").write_text(report, encoding="utf-8")




def render_comparison(folder, metadata_bytes):
    from tools.lora_report_result import comparison_jobs

    metadata = json.loads(metadata_bytes)
    accepted = {job["job_id"] for job in comparison_jobs(metadata, folder)}
    for job in metadata["jobs"]:
        if job["status"] == "done" and job["job_id"] not in accepted:
            job["status"] = "missing"
    write_report(folder, metadata)
    write_blind_report(folder, metadata)
    write_detailed_blind_report(folder, metadata, hashlib.sha256(metadata_bytes).hexdigest())
