/* Offline browser checks. Usage: node tools/validate_dq_dataset_report.cjs REPORT */
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');
const { pathToFileURL } = require('url');
(async () => {
  const report=path.resolve(process.argv[2]);
  const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
  try {
    const page=await browser.newPage({viewport:{width:1440,height:1050}}),errors=[],checks=[];
    page.on('pageerror',e=>errors.push(e.message));
    await page.route(/^https?:/,route=>route.abort());
    await page.goto(pathToFileURL(report).href);
    const check=(name,passed)=>{checks.push({name,passed});if(!passed)throw Error(name)};
    check('no JavaScript errors',errors.length===0);
    check('52 observed images',await page.locator('#scatter circle').count()===52);
    check('selected folder foreground',await page.locator('#scatter circle[opacity="1"]').count()===13);
    check('background context',await page.locator('#scatter circle[opacity="0.27"]').count()===39);
    check('learning and quantization both visible',await page.locator('#scatter-learning').isVisible()&&await page.locator('#scatter').isVisible());
    check('52 paired learning images',await page.locator('#scatter-learning circle').count()===52);
    const learningPoints=()=>page.locator('#scatter-learning circle').evaluateAll(cs=>cs.map(c=>[c.dataset.imageId,c.getAttribute('cx'),c.getAttribute('cy')]));
    const learningBefore=await learningPoints();
    const quantBefore=await page.locator('#scatter circle').evaluateAll(cs=>cs.map(c=>c.getAttribute('cy')));
    const left=await page.locator('#scatter-learning').boundingBox(),right=await page.locator('#scatter').boundingBox();
    check('desktop charts side by side',Math.abs(left.y-right.y)<5&&left.x+left.width<=right.x);
    const axes=await page.locator('#scatter text').allTextContents();
    await page.selectOption('#mul','4');
    check('fixed axes across mul',JSON.stringify(axes)===JSON.stringify(await page.locator('#scatter text').allTextContents()));
    check('mul leaves learning values unchanged',JSON.stringify(learningBefore)===JSON.stringify(await learningPoints()));
    check('mul changes quantization values',JSON.stringify(quantBefore)!==JSON.stringify(await page.locator('#scatter circle').evaluateAll(cs=>cs.map(c=>c.getAttribute('cy')))));
    const learningPoint=page.locator('#scatter-learning circle').first();
    const imageId=await learningPoint.getAttribute('data-image-id');
    await learningPoint.focus();await page.keyboard.press('Enter');
    check('learning point links same image in quantization chart',await page.locator('#scatter circle[data-image-id="'+imageId+'"]').getAttribute('stroke-width')==='2.5');
    await page.uncheck('#background');
    check('background toggle',await page.locator('#scatter circle').count()===13);
    check('background toggle shared by learning chart',await page.locator('#scatter-learning circle').count()===13);
    await page.check('#background');
    await page.check('#single');
    check('exact one selected character tag',await page.locator('#scatter circle').count()===34);
    check('tag filter shared by learning chart',await page.locator('#scatter-learning circle').count()===34);
    await page.uncheck('#single');
    await page.locator('#compare-tabs [data-kind="character"]').click();
    check('two tag series',await page.locator('#comparison polyline').count()===2);
    const cpu=await page.evaluate(()=>{const es=entities('character');return es.map(e=>aggregate(e.samples,true).quant.map(q=>q.d))});
    const payload=JSON.parse(fs.readFileSync(path.join(path.dirname(report),'dataset_summary.json'),'utf8'));
    check('finite real summary curves',cpu.every(a=>a.length===5&&a.every(Number.isFinite))&&payload.samples.length===52);
    await page.locator('#comparison circle').nth(3).scrollIntoViewIfNeeded();
    const pointBox=await page.locator('#comparison circle').nth(3).boundingBox();
    await page.mouse.click(pointBox.x+pointBox.width/2,pointBox.y+pointBox.height/2);
    check('point click synchronizes mul',await page.locator('#mul').inputValue()==='3');
    check('point click synchronizes kind',await page.locator('#view-tabs [data-kind="character"]').getAttribute('aria-pressed')==='true');
    await page.locator('#compare-tabs [data-kind="image"]').click();
    await page.locator('details').filter({has:page.locator('#compare-options')}).locator('summary').click();
    const choices=page.locator('#compare-options input');
    for(let i=3;i<6;i++)await choices.nth(i).check();
    await choices.nth(6).click();
    check('comparison maximum six',await page.locator('#compare-options input:checked').count()===6);
    check('limit has explanation',(await page.locator('#compare-status').textContent()).includes('最大6'));
    await page.locator('#open-tags').click();
    await page.fill('#tag-search','2girls');
    check('original tag choices',await page.locator('#tag-options input[type=checkbox]').count()===1);
    await page.locator('#close-tags').click();
    check('no runtime errors after interaction',errors.length===0);
    await page.locator('#image-reactions').screenshot({path:path.join(path.dirname(report),'reactions.png')});
    await page.screenshot({path:path.join(path.dirname(report),'desktop.png'),fullPage:true});
    await page.setViewportSize({width:390,height:844});
    const upper=await page.locator('#scatter-learning').boundingBox(),lower=await page.locator('#scatter').boundingBox();
    check('mobile charts stack vertically',upper.y+upper.height<lower.y);
    check('responsive page width',await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth));
    await page.screenshot({path:path.join(path.dirname(report),'mobile.png'),fullPage:true});
    const localData=structuredClone(payload);localData.manifest.mode='local';
    for(const sample of localData.samples){sample.improvement_by_noise=[null,null,null];for(const b of sample.bins){b.loss_pre=null;b.improvement_rel=null;b.improvement_abs=null;}}
    const local=await browser.newPage();local.on('pageerror',e=>errors.push(e.message));
    await local.setContent(fs.readFileSync(path.join(__dirname,'../dq_profile/dataset_report.html'),'utf8').replace('__DATASET_PAYLOAD__',JSON.stringify(localData).replace(/</g,'\\u003c')));
    check('local explains unavailable learning comparison',(await local.locator('#learning-note').textContent()).includes('未算出'));
    check('local learning chart remains visible without fabricated points',await local.locator('#scatter-learning').isVisible()&&await local.locator('#scatter-learning circle').count()===0);
    check('local quantization chart remains populated',await local.locator('#scatter circle').count()===52);
    check('all report modes without runtime errors',errors.length===0);await local.close();
    // One physical image can have multiple caption/subset contexts. Do not
    // reuse its measured context when the selected context has no observation.
    for(const scenario of ['unmeasured','valid','invalid_quant','missing_pre']){
      const data=structuredClone(payload), original=data.samples[1], other=structuredClone(original);
      original.caption='context-a';original.tags=['context-a'];
      other.sample_id+=':context-b';other.subset_index=999;other.caption='context-b';other.tags=['context-b'];
      other.folder_id='context-b-folder';other.folder_path='D:/fixture/context-b';other.folder_name='Context B';
      other.measured=scenario!=='unmeasured';
      for(const b of other.bins){
        b.loss_pre=['unmeasured','missing_pre'].includes(scenario)?null:.9;
        b.loss_post=scenario==='unmeasured'?null:.6;
        b.improvement_rel=b.loss_pre===null?null:1/3;
        for(const q of b.quant){q.d=scenario==='unmeasured'?null:4;q.delta=scenario==='unmeasured'?null:.01;}
      }
      if(scenario==='invalid_quant')other.bins[0].quant[0].d=null;
      data.samples.push(other);data.manifest.group_map.groups=[{id:'context-b',kind:'character',tags_any:['context-b']}];
      const contextPage=await browser.newPage();contextPage.on('pageerror',e=>errors.push(e.message));
      await contextPage.setContent(fs.readFileSync(path.join(__dirname,'../dq_profile/dataset_report.html'),'utf8').replace('__DATASET_PAYLOAD__',JSON.stringify(data).replace(/</g,'\\u003c')));
      await contextPage.evaluate(()=>selectEntity('character','context-b'));
      const observed=await contextPage.evaluate(id=>['scatter-learning','scatter'].map(chart=>{const p=[...document.getElementById(chart).querySelectorAll('circle')].find(p=>p.dataset.imageId===id);return p?{opacity:Number(p.getAttribute('opacity')),x:Number(p.dataset.valueX),y:Number(p.dataset.valueY)}:null}),original.image_id);
      check(scenario+' learning context validity',observed[0].opacity===(['unmeasured','missing_pre'].includes(scenario)?.27:1));
      check(scenario+' quantization context validity',observed[1].opacity===(['unmeasured','invalid_quant'].includes(scenario)?.27:1));
      if(scenario==='valid'){
        check('foreground uses selected sample values',Math.abs(observed[0].x-.9)<1e-10&&Math.abs(observed[0].y-1/3)<1e-10&&Math.abs(observed[1].x-.6)<1e-10&&observed[1].y===4);
        const before=await contextPage.locator('#scatter text').allTextContents();
        await contextPage.evaluate(()=>selectEntity('image',entities('image')[1].id));
        check('context selection leaves axes fixed',JSON.stringify(before)===JSON.stringify(await contextPage.locator('#scatter text').allTextContents()));
      }
      if(scenario==='unmeasured'){
        await contextPage.uncheck('#background');
        check('unmeasured context has no foreground points',await contextPage.locator('#scatter circle,#scatter-learning circle').count()===0);
        await contextPage.evaluate(()=>selectEntity('folder','context-b-folder'));
        check('folder context also excludes borrowed measurements',await contextPage.locator('#scatter circle,#scatter-learning circle').count()===0);
      }
      if(scenario==='invalid_quant'){
        await contextPage.selectOption('#bin','1');
        check('valid selected bin restores foreground',await contextPage.locator('#scatter circle[data-image-id="'+original.image_id+'"]').getAttribute('opacity')==='1');
      }
      await contextPage.close();
    }
    check('context regressions without runtime errors',errors.length===0);
    fs.writeFileSync(path.join(path.dirname(report),'browser-validation.json'),JSON.stringify({checks,errors},null,2));
    console.log(JSON.stringify({passed:checks.length,errors,report}));
  }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
