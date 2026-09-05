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
    const sorting=await browser.newPage();sorting.on('pageerror',e=>errors.push(e.message));
    await sorting.setContent(fs.readFileSync(path.join(__dirname,'../dq_profile/dataset_report.html'),'utf8').replace('__DATASET_PAYLOAD__',JSON.stringify(payload).replace(/</g,'\\u003c')));
    await sorting.click('#view-tabs [data-kind="image"]');
    const selected=await sorting.locator('#detail h2').first().textContent();
    await sorting.selectOption('#sort-by','post');await sorting.selectOption('#sort-direction','desc');
    check('highest MSE image sorts first',await sorting.locator('#rows button').first().getAttribute('data-id')==='i51');
    check('sorting preserves selected detail',await sorting.locator('#detail h2').first().textContent()===selected);
    await sorting.click('#next-page');check('sorting spans pagination',await sorting.locator('#rows button').first().getAttribute('data-id')==='i27');
    await sorting.evaluate(()=>selectEntity('image','i0'));check('selection locates sorted page',(await sorting.locator('#page-label').textContent()).startsWith('49'));
    for(const kind of ['image','folder','character']){
      await sorting.click('#view-tabs [data-kind="'+kind+'"]');
      for(const [metric,col] of [['pre',2],['post',3],['rel',4],['d',5]])for(const direction of ['asc','desc']){
        await sorting.selectOption('#sort-by',metric);await sorting.selectOption('#sort-direction',direction);
        const values=await sorting.locator('#rows tr').evaluateAll((rs,col)=>rs.map(r=>parseFloat(r.children[col].textContent)),col);
        check(kind+' '+metric+' '+direction+' ordered',values.every((v,i)=>!i||!Number.isFinite(v)||Math.abs(v-values[i-1])<1e-8||(direction==='asc'?v>=values[i-1]:v<=values[i-1])));
      }
    }
    await sorting.evaluate(()=>{for(const b of S[0].bins)b.loss_pre=null;render()});
    await sorting.click('#view-tabs [data-kind="image"]');await sorting.selectOption('#sort-by','pre');
    for(const direction of ['asc','desc']){
      await sorting.selectOption('#sort-direction',direction);await sorting.click('#next-page');await sorting.click('#next-page');
      check('missing values last in '+direction,await sorting.locator('#rows button').last().getAttribute('data-id')==='i0');
    }
    await sorting.evaluate(()=>{for(const b of S[0].bins)b.quant[4].d=1000});
    await sorting.selectOption('#sort-by','d');await sorting.selectOption('#sort-direction','desc');await sorting.selectOption('#mul','4');
    check('sort responds to selected mul',await sorting.locator('#rows button').first().getAttribute('data-id')==='i0');
    await sorting.close();check('sorting without runtime errors',errors.length===0);
    // Changing only aliases can change membership without changing canonical group rules.
    const aliasData=structuredClone(payload);
    for(const g of aliasData.manifest.group_map.groups)g.tags_all||=[];
    const aliasGroup=aliasData.manifest.group_map.groups[0];
    for(const s of aliasData.samples)if(!s.tags.includes(aliasGroup.tags_any[0]))s.tags.push('alias_extra','é');
    const aliasPage=await browser.newPage();aliasPage.on('pageerror',e=>errors.push(e.message));
    await aliasPage.setContent(fs.readFileSync(path.join(__dirname,'../dq_profile/dataset_report.html'),'utf8').replace('__DATASET_PAYLOAD__',JSON.stringify(aliasData).replace(/</g,'\\u003c')));
    await aliasPage.evaluate(id=>selectEntity('character',id),aliasGroup.id);
    const originalInterval=await aliasPage.locator('#intervals').textContent();
    const originalCount=await aliasPage.evaluate(()=>aggregate(entities('character')[0].samples).total);
    check('original tag interval is available',originalInterval.includes('2000回'));
    const importAliases=async (aliases,expected=aliases)=>{
      await aliasPage.locator('#import-map').setInputFiles({name:'groups.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify({...aliasData.manifest.group_map,aliases}))});
      await aliasPage.waitForFunction(expected=>JSON.stringify(state.aliases)===JSON.stringify(expected),expected);
    };
    await importAliases({alias_extra:aliasGroup.tags_any[0]});
    check('alias import changes membership without changing group rules',await aliasPage.evaluate(count=>aggregate(entities('character')[0].samples).total>count&&JSON.stringify(state.groups)===JSON.stringify(M.group_map.groups),originalCount));
    check('changed aliases hide stale interval and request rebuild',(await aliasPage.locator('#intervals').textContent()).includes('CPU再集計'));
    await importAliases(aliasData.manifest.group_map.aliases||{});
    check('restoring aliases restores original interval',await aliasPage.locator('#intervals').textContent()===originalInterval);
    await importAliases({' e\u0301 ':' '+aliasGroup.tags_any[0]+' '},{'é':aliasGroup.tags_any[0]});
    check('normalized alias key and value include matching images',await aliasPage.evaluate(count=>aggregate(entities('character')[0].samples).total>count,originalCount));
    const validAliasState=await aliasPage.evaluate(()=>JSON.stringify({aliases:state.aliases,groups:state.groups}));
    for(const [label,aliases] of [
      ['empty key',{' ':'a'}],['empty value',{'a':' '}],
      ['conflict',{'é':'a','e\u0301':'b'}],
      ['chain',{'x':' e\u0301 ','é':'y'}],['cycle',{'é':' e\u0301 '}],
    ]){
      await aliasPage.locator('#map-status').evaluate(e=>e.textContent='');
      await aliasPage.locator('#import-map').setInputFiles({name:'invalid.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify({...aliasData.manifest.group_map,aliases}))});
      await aliasPage.waitForFunction(()=>document.getElementById('map-status').textContent.startsWith('読込失敗'));
      check('reject normalized alias '+label+' without replacing active settings',await aliasPage.evaluate(()=>JSON.stringify({aliases:state.aliases,groups:state.groups}))===validAliasState);
    }
    await importAliases({'é':aliasGroup.tags_any[0],' e\u0301 ':aliasGroup.tags_any[0]},{'é':aliasGroup.tags_any[0]});
    check('equivalent duplicate aliases merge safely',await aliasPage.evaluate(()=>Object.keys(state.aliases).length===1));
    const unicodeMap={aliases:{[aliasGroup.tags_any[0]]:' e\u0301 '},groups:[{id:'unicode',tags_any:['é']}]};
    await aliasPage.locator('#import-map').setInputFiles({name:'unicode.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(unicodeMap))});
    await aliasPage.waitForFunction(()=>state.groups[0].id==='unicode');
    check('normalized Unicode alias value matches canonical group tag',await aliasPage.evaluate(()=>state.aliases[M.group_map.groups[0].tags_any[0]]==='é'&&aggregate(entities('character')[0].samples).total===S.length));
    await aliasPage.close();check('alias interval checks without runtime errors',errors.length===0);
    const pathPage=await browser.newPage();pathPage.on('pageerror',e=>errors.push(e.message));
    await pathPage.setContent(fs.readFileSync(path.join(__dirname,'../dq_profile/dataset_report.html'),'utf8').replace('__DATASET_PAYLOAD__',JSON.stringify(payload).replace(/</g,String.fromCharCode(92)+'u003c')));
    const originalPathSettings=await pathPage.evaluate(()=>JSON.stringify({aliases:state.aliases,groups:state.groups}));
    const backslash=String.fromCharCode(92);
    for(const imagePath of ['image.png','./image.png','../image.png','D:image.png',backslash+'image.png']){
      await pathPage.locator('#map-status').evaluate(e=>e.textContent='');
      await pathPage.locator('#import-map').setInputFiles({name:'relative.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify({aliases:{new_alias:'canonical'},groups:[{id:'relative',image_paths:[imagePath]}]}))});
      await pathPage.waitForFunction(()=>document.getElementById('map-status').textContent.length>0);
      const message=await pathPage.locator('#map-status').textContent();
      check('relative image path rejected with CPU guidance: '+imagePath,message.startsWith('読込失敗')&&message.includes('Python')&&message.includes('再集計'));
      check('relative image path preserves current settings: '+imagePath,await pathPage.evaluate(()=>JSON.stringify({aliases:state.aliases,groups:state.groups}))===originalPathSettings);
    }
    for(const imagePath of ['D:/fixture/image.png',['D:','fixture','image.png'].join(backslash),backslash.repeat(2)+['server','share','image.png'].join(backslash),'/fixture/image.png']){
      await pathPage.evaluate(p=>S[0].path=p,imagePath);
      await pathPage.locator('#map-status').evaluate(e=>e.textContent='');
      await pathPage.locator('#import-map').setInputFiles({name:'absolute.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify({groups:[{id:'absolute',image_paths:[imagePath]}]}))});
      await pathPage.waitForFunction(()=>document.getElementById('map-status').textContent.length>0);
      check('absolute image path still matches: '+imagePath,await pathPage.evaluate(()=>document.getElementById('map-status').textContent==='設定を読み込みました。'&&entities('character')[0].samples.length===1));
    }
    await pathPage.close();check('path import checks without runtime errors',errors.length===0);
    // Exercise native initial focus: selectOption alone does not reproduce focus scrolling.
    for(const width of [1440,390]){
      const sticky=await browser.newPage({viewport:{width,height:1000}});
      sticky.on('pageerror',e=>errors.push(e.message));
      await sticky.goto(pathToFileURL(report).href);
      await sticky.locator('#comparison').evaluate(e=>e.scrollIntoView({block:'start'}));
      await sticky.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
      const origin=await sticky.evaluate(()=>scrollY);
      const clickVisible=async selector=>{
        const r=await sticky.locator(selector).boundingBox();
        await sticky.mouse.click(r.x+r.width/2,r.y+r.height/2);
      };
      for(const id of ['mul','bin']){
        await clickVisible('#'+id);await sticky.keyboard.press('Escape');
        check(width+' '+id+' initial mouse focus preserves scroll',Math.abs(await sticky.evaluate(()=>scrollY)-origin)<2);
        await sticky.keyboard.press('ArrowDown');
        check(width+' '+id+' value change preserves scroll',Math.abs(await sticky.evaluate(()=>scrollY)-origin)<2);
      }
      // Start Tab navigation immediately before the toolbar, without moving the viewport.
      await sticky.evaluate(()=>{const b=document.createElement('button');b.id='focus-start';document.getElementById('report-controls').before(b);b.focus({preventScroll:true});b.style.position='absolute'});
      for(const id of ['mul','bin','open-tags']){
        await sticky.keyboard.press('Tab');
        check(width+' '+id+' keyboard focus preserves scroll',await sticky.evaluate(({id,origin})=>document.activeElement.id===id&&Math.abs(scrollY-origin)<2,{id,origin}));
      }
      await sticky.locator('#focus-start').evaluate(e=>e.remove());
      await clickVisible('#open-tags');
      const bar=await sticky.locator('#report-controls').boundingBox(),tags=await sticky.locator('#tag-panel').boundingBox();
      check(width+' tag panel clears sticky toolbar',tags.y>=bar.y+bar.height);
      await clickVisible('#close-tags');
      check(width+' closing tags restores position',Math.abs(await sticky.evaluate(()=>scrollY)-origin)<2);
      await sticky.locator('#image-detail').evaluate(e=>e.scrollIntoView({block:'start'}));
      check(width+' detail anchor clears toolbar',(await sticky.locator('#image-detail').boundingBox()).y>=bar.height);
      await sticky.close();
    }
    check('sticky controls without runtime errors',errors.length===0);
    fs.writeFileSync(path.join(path.dirname(report),'browser-validation.json'),JSON.stringify({checks,errors},null,2));
    console.log(JSON.stringify({passed:checks.length,errors,report}));
  }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
