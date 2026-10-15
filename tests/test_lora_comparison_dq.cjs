// CPU-only synthetic regressions for the standalone viewer's DQ import semantics.
// No browser, GPU, real reports or filesystem paths from training data are used.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const html=fs.readFileSync(path.join(__dirname,'../tools/lora_comparison.html'),'utf8');
const script=html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
class Element{
 constructor(){this.children=[];this.style={setProperty(){}};this.textContent='';this.hidden=false}
 setAttribute(){} addEventListener(){}
 append(...nodes){this.children.push(...nodes)}
 replaceChildren(...nodes){this.children=nodes}
}
const nodes=new Map();
const document={documentElement:{outerHTML:''},getElementById(id){if(!nodes.has(id))nodes.set(id,new Element());return nodes.get(id)},createElement(){return new Element()}};
const context=vm.createContext({document,getComputedStyle(){return {getPropertyValue(){return '#123456'}}}});
vm.runInContext(script.slice(0,script.indexOf("$('version').textContent="))+"globalThis.api={normalizeReport,state,catalogue,renderDQContext,sampleAt};})();",context);
const {normalizeReport:normalize,state,catalogue,renderDQContext,sampleAt}=context.api;
const plain=v=>JSON.parse(JSON.stringify(v));
const chart=(id,scope)=>({id,title:'Recorded title',dq_scope:scope,x_label:'TrainStep',x:[100,150,200],series:[{name:'ClipRateRaw',y:[.01,null,.02]}]});
const report=(dq,charts,settings)=>({base_name:'synthetic',dq,charts,training_settings:settings});
const norm=(r,id='run')=>normalize(r,id+'.json',id);
const get=(r,id)=>r.charts.find(c=>c.id===id);
const one=norm(report({rows:[{Scope:'unet'}]},{dq:[chart('dq_clip','combined')]}),'one');
const both=norm(report({scope_rows:{unet:[],te:[]},rows:[{Scope:'unet'},{Scope:'te'}]},
 {dq:[chart('dq_clip','combined'),chart('dq_clip_unet','unet'),chart('dq_clip_te','te')]}),'both');
assert.equal(one.charts[0].id,'dq_clip_unet','single-scope aggregate aligns with that scope in mixed logs');
assert.equal(get(both,'dq_clip_unet').scope,'unet','chart scope overrides the containing multi-scope section');
assert.equal(get(both,'dq_clip_te').scope,'te');
assert.equal(get(both,'dq_clip_both').scope,'both','combined remains separate from UNet');
assert.match(get(both,'dq_clip_te').title,/TE/);
state.runs=[one,both];state.selected=['one','both'];
assert.equal(catalogue().find(c=>c.id==='dq_clip_unet').count,2);
assert.deepEqual(plain(get(both,'dq_clip_unet').series[0].points),[[100,.01],[150,null],[200,.02]],'gaps and raw numbers unchanged');
const old=norm(report({rows:[{TrainStep:100}]},{dq:[chart('dq_clip')]},{values:{dq_delta_scope:'unet'}}));
assert.equal(old.charts[0].id,'dq_clip_unknown','training scope is not proof of logging scope');
const oldKnown=norm(report({rows:[{Scope:'unet'}]},{dq:[chart('dq_clip')]}));
assert.equal(oldKnown.charts[0].id,'dq_clip_unet','legacy Scope column is supported');
const auto=norm(report({rows:[{Scope:'unet'},{Scope:'te'}],auto_monitor:{mode:'recorded',scopes:['unet']}},
 {dq_auto:[chart('dq_auto_clip_unet'),chart('dq_auto_mul_unet'),chart('dq_auto_bad_unet')]}));
assert.equal(get(auto,'dq_auto_clip_unet').scope,'unet','controller chart does not inherit mixed logging scope');
assert.equal(get(auto,'dq_auto_mul_unet').discrete,true);
assert.equal(get(auto,'dq_auto_bad_unet').discrete,true);
assert.equal(sampleAt([[0,2.7],[10,3]],5,get(auto,'dq_auto_mul_unet').discrete),2.7,'mul changes use preceding value, not interpolation');
assert.equal(sampleAt([[0,.1],[10,.2]],5,get(auto,'dq_auto_clip_unet').discrete),.15000000000000002);
const rows=[{label:'Other UNet',mul:2.7,enabled:true},{label:'unet.attn2',mul:3.75,enabled:true},{label:'TE1',mul:3.75,enabled:false}];
const spatial=norm(report({scope_rows:{unet:[]},mul_settings:{rows,source:'saved policy',note:'fixed allocation'},auto_enabled:false,auto_monitor:{mode:'reference',notes:['auto is OFF']}},
 {dq:[{...chart('dq_range_mul'),series:[{name:'RangeMul',y:[2.7,2.7,2.7]}]}]}),'spatial');
assert.equal(spatial.charts[0].scope,'','common mul column is not a component scope');
assert.match(spatial.charts[0].title,/共通設定値/);
assert.equal(spatial.charts[0].series[0].name,'共通設定値（RangeMul列）');
assert.deepEqual(plain(spatial.dq_context.mulRows),rows);
assert.equal(spatial.dq_context.autoEnabled,false);
assert.equal(norm(report({}, {dq:[chart('dq_bits')]})).dq_context.autoEnabled,null,'missing auto setting is not OFF');
const missing=norm(report({}, {dq:[chart('dq_range_mul')]},{values:{dq_delta_policy_file:'policy.json'}}));
assert.match(missing.dq_context.mulNote,/再生成/,'do not reopen a policy path or present its base as uniform');
assert.deepEqual(plain(JSON.parse(JSON.stringify(spatial)).dq_context),plain(spatial.dq_context),'context survives saved comparison serialization');
function content(n){return n.textContent+n.children.map(c=>typeof c==='string'?c:content(c)).join('\n')}
state.chart='dq_range_mul';renderDQContext([{run:spatial,chart:spatial.charts[0]}]);
const rendered=content(nodes.get('dq-context-body'));
assert.match(rendered,/unet.attn2：3.75/);assert.match(rendered,/TE1：量子化OFF/);assert.match(rendered,/自動mul調整：OFF/);
assert.match(content(nodes.get('notices')),/同じ線でも部位別mulの配分が同じとは限りません/);
renderDQContext([{run:one,chart:{section:'grad'}}]);assert.equal(nodes.get('dq-context').hidden,true);
state.chart='dq_clip_unknown';renderDQContext([{run:old,chart:old.charts[0]}]);
assert.match(content(nodes.get('notices')),/旧JSON/);
console.log('PASS: DQ scopes, legacy/current alignment, combined isolation, missing scope, controller scope, discrete history, gaps, fixed mul context, unknown auto, serialization and context rendering.');
