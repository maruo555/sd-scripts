// Browser integration for offline comparison. Playwright and Edge/Chromium are test-only.
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const {chromium} = require(process.env.LORA_VIEWER_PLAYWRIGHT_MODULE || 'playwright');

const output = process.env.LORA_VIEWER_ARTIFACTS || fs.mkdtempSync(path.join(os.tmpdir(),'diagnostic-viewer-'));
fs.mkdirSync(output, {recursive:true});
const viewer = path.resolve(__dirname, '../../tools/lora_comparison.html');
const makeChart = (id, condition={}, xLabel='Attempt step') => ({
  id, title:id, x_label:xLabel, x:[1,2,3,4], comparison_condition:condition,
  series:[{name:'測定', y:[0,null,3,4]}]
});
function report(runId, generation) {
  return {
    report_schema_version:1, base_name:'同名 </script> 日本語', generated_at:'2026-11-22',
    provenance:{run_id:runId, session_id:'42', generation_id:generation},
    training_settings:{status:'resolved',requested:{seed:0, omitted:null},resolved:{
      dq:{rounding_mode:'stoch',stat:'rms',granularity:'channel',auto:{enabled:true,preset:'clip_rate_high'}},
      args:{seed:0, enabled:false}}},
    charts:{grad:[makeChart('loss_ma',{},'Batch observation')],
      dq:[makeChart('dq_clip_unet',{scope:'unet',definition_version:1,target:'delta',source:'observed'})],
      promote:[makeChart('promote_proxy_loss',{},'Epoch')]},
    rank:{final_heatmaps:{rank_sat_wmean:{Epoch:3, TrainStep:5,Scope:'unet',
      rows:[{label:'down',cells:[{role:'mlp',value:0},{role:'to_q',value:null}]}]}}}
  };
}
const upload = (name,data) => ({name, mimeType:'application/json',buffer:Buffer.from(JSON.stringify(data))});
(async ()=>{
  const browser = await chromium.launch({headless:true,
    ...(process.env.LORA_VIEWER_BROWSER?{executablePath:process.env.LORA_VIEWER_BROWSER}:{})});
  try {
    const context = await browser.newContext({acceptDownloads:true,offline:true});
    const page = await context.newPage();
    const errors=[];
    page.on('pageerror',e=>errors.push(e.message));
    async function add(files) {
      await page.setInputFiles('#file-input',files);
      await page.waitForFunction(()=>!document.querySelector('#add-files').disabled);
    }
    await page.goto(pathToFileURL(viewer).href);
    const first=report(crypto.randomUUID(),'gen1'), other=report(crypto.randomUUID(),'gen2');
    other.training_settings.requested.seed=4;
    other.charts.dq[0].comparison_condition.definition_version=2;
    await add([upload('01.json',first),upload('02.json',other)]);
    assert.equal(await page.locator('.run-card .run-name').count(),2);
    assert((await page.locator('#run-list').innerText()).includes(first.provenance.run_id));
    assert((await page.locator('#catalog').innerText()).includes('Promote'));
    assert.equal(await page.locator('#plots path[data-series]').count(),2);
    if (!await page.locator('#settings-differences').isVisible()) await page.locator('#training-settings > summary').click();
    await page.locator('#settings-differences').uncheck();
    assert((await page.locator('#settings-table').textContent()).includes('実効値 / args / enabledfalsefalse'));
    assert((await page.locator('#settings-table').textContent()).includes('要求値 / omittedNone'));
    await page.locator('[data-chart-id="dq_clip_unet"]').click();
    assert((await page.locator('#notices').textContent()).includes('測定条件'));
    await page.selectOption('#mode','difference');
    assert((await page.locator('#notices').textContent()).includes('測定条件'));
    await page.locator('[data-chart-id="heat_rank_sat_wmean"]').click();
    assert(await page.locator('#plots').textContent());
    await add([upload('duplicate.json',first)]);
    assert.equal(await page.locator('.run-card .run-name').count(),2);
    assert((await page.locator('#errors').textContent()).includes('同じ内容'));
    const next={...first,provenance:{...first.provenance,generation_id:'gen3'},charts:{},rank:{}};
    page.once('dialog',dialog=>dialog.dismiss());
    await add([upload('replacement.json',next)]);
    assert((await page.locator('#run-list').textContent()).includes('gen1'));
    page.once('dialog',dialog=>dialog.accept());
    await add([upload('replacement.json',next)]);
    assert.equal(await page.locator('.run-card .run-name').count(),2);
    assert((await page.locator('#run-list').textContent()).includes('gen3'));
    await page.locator('#run-list input').nth(1).uncheck();
    assert((await page.locator('#plots').textContent()).includes('比較できる系列なし'));
    await add([upload('unknown.json',{...first,report_schema_version:2})]);
    assert((await page.locator('#errors').textContent()).includes('report_schema_version=1'));
    const downloadEvent=page.waitForEvent('download');
    await page.click('#save');
    const download=await downloadEvent;
    const saved=path.join(output,'saved-fixture.html');
    await download.saveAs(saved);
    const offline=await context.newPage();
    offline.on('pageerror',e=>errors.push(e.message));
    await offline.goto(pathToFileURL(saved).href);
    assert.equal(await offline.locator('.run-card .run-name').count(),2);
    assert((await offline.locator('#run-list').textContent()).includes('gen3'));
    assert((await offline.locator('#plots').textContent()).includes('比較できる系列なし'));
    if (process.env.LORA_VIEWER_ACTUAL_JSON) {
      await page.goto(pathToFileURL(viewer).href);
      await add([process.env.LORA_VIEWER_ACTUAL_JSON]);
      assert.equal(await page.locator('.run-card .run-name').count(),1);
      assert(await page.locator('#catalog button').count()>10);
      assert(await page.locator('[data-chart-id="promote_proxy_loss"]').count());
      assert(await page.locator('#plots path[data-series]').count());
      console.log('Actual indexed diagnostic charts:',await page.locator('#catalog button').count());
    }
    assert.deepEqual(errors,[]);
    console.log('Passed offline compare, conditions, regeneration, settings, empty series and saved HTML');
  } finally { await browser.close(); }
})().catch(e=>{console.error(e);process.exitCode=1;});
