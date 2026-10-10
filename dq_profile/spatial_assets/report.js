/* Offline report. Values are computed from matched diagnostic observations. */
(() => {
  'use strict';
  const DATA = window.DQ_DATA;
  const PAGE = document.body.dataset.page;
  const el = id => document.getElementById(id);
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const finite = x => typeof x === 'number' && Number.isFinite(x);
  const fmt = (x, n=3) => finite(x) ? x.toFixed(n) : '—';
  const signed = (x, n=4) => finite(x) ? `${x > 0 ? '+' : ''}${x.toFixed(n)}` : '—';
  const mean = values => values.length && values.every(finite) ? values.reduce((a,b)=>a+b,0)/values.length : null;
  const pct = (a,b) => finite(a) && finite(b) && b !== 0 ? 100*(a-b)/b : null;
  const familyColor = {uniform:'#2563ad', adaptive:'#168474', reference:'#8656b3'};
  const metricDefs = [ ['p50','P50','#2563ad'], ['body','Body · P95','#17836c'], ['tail','Tail','#cc7427'] ];
  const parallelDefs = [ ['parallel_p50','元方向 · P50','#6550a6'], ['parallel_p05','元方向 · P05','#338d9c'] ];
  const sourceColors = ['#3577b3','#1a987d','#c48735','#9263b2','#cd657b','#5d8c76'];
  const query = new URLSearchParams(location.search);
  const state = {dataset:query.get('dataset')||DATA.datasets[0].id, condition:query.get('condition')||'', part:query.get('part')||'all', regime:query.get('regime')||'off', showA:true, fixed:false, entityKind:'image', entityId:null, bin:'all', sort:'id', compareMetric:'d'};
  let version = 0;
  let busy = Promise.resolve();
  let plots = [];
  let datasetPlotSnapshot = null;
  const dataset = () => DATA.datasets.find(d=>d.id===state.dataset) || DATA.datasets[0];
  const activeRegime = () => state.regime==='on' ? 'on' : 'off';
  const shownRegimes = () => state.regime==='both' ? ['off','on'] : [activeRegime()];
  const candidate = () => dataset().candidates.find(c=>c.id===state.condition);
  const candidates = () => dataset().candidates.filter(c=>state.showA || c.kind==='uniform');
  const score = (cid, regime=activeRegime(), part=state.part) => dataset().regimes[regime]?.scores[part]?.[cid] || {};
  const plain = node => { if (node?.data) Plotly.purge(node); if(node) node.innerHTML=''; };
  const empty = (id, title, message, link='') => {
    const node=el(id); plain(node);
    node.innerHTML=`<div class="empty"><strong>${esc(title)}</strong><small>${esc(message)}</small>${link}</div>`;
  };
  function layout(extra={}) {
    return {margin:{l:64,r:24,t:30,b:66}, font:{family:'Yu Gothic UI, Meiryo, sans-serif',size:12,color:'#425970'}, paper_bgcolor:'#fff',plot_bgcolor:'#fff', hovermode:'closest', hoverlabel:{bgcolor:'#fff',font:{size:12}}, legend:{orientation:'h',x:0,y:1.12,font:{size:12}}, xaxis:{gridcolor:'#e9eef4',zeroline:false,automargin:true}, yaxis:{gridcolor:'#e9eef4',zerolinecolor:'#c2cfdb',automargin:true}, ...extra};
  }
  function plot(id, traces, options, onclick) {
    const node=el(id); if (!node) return;
    // Empty placeholders and old SVGs must be discarded before a new measured view.
    plain(node);
    const task=Plotly.newPlot(node,traces,layout(options),{displayModeBar:false,responsive:true,scrollZoom:false}).then(()=>{
      node.on('plotly_click', event=>{ const point=event.points?.[0]; if(point&&onclick) onclick(point); });
    });
    plots.push(task);
  }
  function selectCondition(id) {
    if(!dataset().candidates.some(c=>c.id===id)) return;
    state.condition=id;
    if(candidate().kind!=='uniform') state.showA=true;
    refresh();
  }
  function selectSample(id) {
    state.entityKind='image';state.entityId=id;refresh();
  }
  function optionList(items,selected) { return items.map(([v,l])=>`<option value="${esc(v)}"${v===selected?' selected':''}>${esc(l)}</option>`).join(''); }
  function normalize() {
    const d=dataset();state.dataset=d.id;
    if(!d.views[state.part]) state.part='all';
    if(!d.regimes.on || !['off','on','both'].includes(state.regime)) state.regime='off';
    if(PAGE==='dataset'&&state.regime==='both') state.regime='off';
    if(!d.candidates.some(c=>c.id===state.condition)) state.condition=d.candidates.find(c=>c.kind==='reference')?.id||d.candidates.find(c=>c.mul[0]===d.body_mul)?.id||d.candidates[0].id;
    if(!state.showA&&candidate().kind!=='uniform') state.condition=d.candidates.find(c=>c.kind==='uniform'&&c.mul[0]===candidate().mul[0])?.id||d.candidates[0].id;
    if(state.entityKind==='tag'&&!d.samples.some(s=>s.tags.length)) state.entityKind='image';
  }
  function shared() {
    const d=dataset(),c=candidate();
    el('dataset').innerHTML=optionList(DATA.datasets.map(v=>[v.id,v.label]),d.id);
    el('part').innerHTML=optionList(Object.entries(d.views),state.part);
    el('part').disabled=Object.keys(d.views).length===1;
    const regimeOptions=PAGE==='beginner'?[['off','OFFのみ（基本）'],['both','OFFとONを並べる'],['on','ONのみ']]:[['off','OFF'],['on','ON']];
    el('regime').innerHTML=optionList(regimeOptions.filter(([v])=>v==='off'||d.regimes.on),state.regime);
    el('regime').disabled=!d.regimes.on;
    el('condition').innerHTML=optionList((PAGE==='beginner'?candidates():d.candidates).map(v=>[v.id,v.label]),state.condition);
    el('selection-name').textContent=c.label;
    el('training-cli').textContent=c.training_cli;
    el('condition-notes').textContent=[...(c.aliases||[]).map(a=>a.label+' と同じ配分のため統合'),...(d.skipped||[]).map(x=>(x.label?x.label+': ':'')+({body_representative_unavailable:'Body代表なし：Body基準の追加配分は未測定',low_exceeds_body:'低側がBodyより高いため除外',spatial_comparison_disabled:'追加配分は指定により省略'}[x.reason]||x.reason))].join(' / ');
    el('allocation').innerHTML=['その他UNet','attn2','TE1','TE2'].map((s,i)=>`<span>${s} <b>${c.mul[i]===null?'量子化OFF':fmt(c.mul[i],2)}</b></span>`).join('');
    el('metadata').innerHTML=[`<span class="tag te">量子化対象 · ${esc(d.quant_scope)}</span>`,`<span class="tag">${d.meta.images}入力 / ${d.meta.groups}/${d.meta.total_groups}元画像グループ</span>`,`<span class="tag">${d.meta.timestep_bins}時刻帯 × ${d.meta.noise_replicas}ノイズ × ${d.meta.quant_repeats}量子化反復</span>`, finite(d.meta.step)?`<span class="tag">共通warmup ${d.meta.step} step</span>`:'',`<span class="tag experimental">${d.candidates.length}条件 · 保存済み実測</span>`].join('');
    el('scope-note').textContent=d.scope_note+' '+d.loss_note+' 元画像グループを均等に扱う集計と、各画像を均等に扱う一覧の数値は区別しています。';
    const params=new URLSearchParams({dataset:d.id,condition:state.condition,part:state.part,regime:state.regime});
    history.replaceState(null,'',`${location.pathname}?${params}`);
    el('nav-beginner').href=`beginner_report.html?${params}`;
    if(state.regime==='both') params.set('regime','off');
    el('nav-dataset').href=`dataset_report.html?${params}`;
    const legacy=el('legacy-detail');
    if(legacy){legacy.hidden=!d.legacy_detail_url;if(d.legacy_detail_url)legacy.href=d.legacy_detail_url;}
  }
  function perSample(sample,cid,regime=activeRegime(),part=state.part) {
    const rows=(sample.bins[regime]||[]).filter(b=>state.bin==='all'||String(b.bin)===state.bin);
    const warm=(sample.bins.off||[]).filter(b=>state.bin==='all'||String(b.bin)===state.bin);
    const qrows=rows.map(b=>b.q[cid]);
    const values=key=>mean(qrows.map(q=>q?.[key]));
    const pre=mean(warm.map(r=>r.pre)),post=mean(warm.map(r=>r.post));
    return {d:mean(qrows.map(q=>q?.d?.[part])),parallel:mean(qrows.map(q=>q?.parallel_by_part?.[part])),reference_loss:values('reference_loss'),quantized_loss:values('quantized_loss'),delta:values('delta'),pre,post,improvement:finite(pre)&&pre>0&&finite(post)?100*(pre-post)/pre:null,observations:qrows.length&&qrows.every(q=>q)?qrows.reduce((a,q)=>a+q.observations,0):0};
  }
  function aggregate(samples,cid,regime=activeRegime(),part=state.part) {
    const rows=samples.map(s=>perSample(s,cid,regime,part)),out={};
    ['d','parallel','reference_loss','quantized_loss','delta','pre','post'].forEach(k=>out[k]=mean(rows.map(r=>r[k])));
    out.improvement=finite(out.pre)&&out.pre>0&&finite(out.post)?100*(out.pre-out.post)/out.pre:null;
    out.count=samples.length;out.observations=rows.reduce((a,r)=>a+r.observations,0);
    return out;
  }
  function curveTraces(defs,regime,accessor) {
    const all=candidates(),traces=[];
    defs.forEach(([key,label,color])=>{
      ['uniform','adaptive','reference'].forEach(kind=>{
        const group=all.filter(c=>c.kind===kind);if(!group.length)return;
        const values=group.map(c=>accessor?accessor(c,key,regime):score(c.id,regime)[key]);
        if(!values.some(finite))return;
        traces.push({type:'scatter',mode:kind==='uniform'?'lines+markers':'markers',name:label,legendgroup:key,showlegend:kind==='uniform',x:group.map(c=>c.mul[0]),y:values,line:{color,width:2},marker:{color,symbol:group.map(c=>c.symbol),size:group.map(c=>c.kind==='uniform'?(c.id===state.condition?11:7):(c.id===state.condition?17:13)),line:{color:'#fff',width:1}},customdata:group.map(c=>[c.id,c.label,c.assignment_label,label]),hovertemplate:'<b>%{customdata[1]}</b><br>%{customdata[2]}<br>%{customdata[3]} = %{y:.4f}<extra></extra>'});
      });
    });
    return traces;
  }
  function panelCharts(id,prefix) {
    const regimes=shownRegimes(),parent=el(id);
    parent.querySelectorAll('.js-plotly-plot').forEach(n=>Plotly.purge(n));
    parent.classList.toggle('single',regimes.length===1);
    parent.innerHTML=regimes.map(r=>`<div class="chart-cell"><div class="regime-label">dropout ${r.toUpperCase()} · ${esc(dataset().views[state.part])}</div><div class="chart" id="${prefix}-${r}"></div></div>`).join('');
    return regimes;
  }
  function mainCharts() {
    const regimes=panelCharts('main-charts','distance');
    const values=regimes.flatMap(r=>candidates().flatMap(c=>metricDefs.map(([k])=>score(c.id,r)[k]))).filter(finite);
    const max=state.fixed?4:Math.max(0.1,...values)*1.15;
    const xr={title:{text:'その他UNetのmul'},tickvals:[2.7,3.15,3.45,3.75,4.05],tickformat:'.2f',range:[2.60,4.14],gridcolor:'#edf1f5',zeroline:false};
    regimes.forEach(r=>plot(`distance-${r}`,curveTraces(metricDefs,r),{xaxis:xr,yaxis:{title:{text:'相対勾配差 d'},range:[0,max],gridcolor:'#e7edf4'},shapes:[{type:'line',xref:'paper',x0:0,x1:1,y0:1,y1:1,line:{color:'#a6b4c5',width:1,dash:'dot'}}]},p=>selectCondition(p.customdata[0])));
    el('clip-note').textContent=values.some(v=>v>max)?'共通目盛の上限4を超えた点があります。「共通目盛」のチェックを外すと全範囲を表示します。表には実際の値を残しています。':'';
    const direction=panelCharts('direction-charts','parallel');
    el('direction-scope').textContent=dataset().views[state.part]+'の実測';
    if(!direction.some(r=>candidates().some(c=>finite(score(c.id,r).parallel_p50)))) {
      direction.forEach(r=>empty(`parallel-${r}`,'部位別の元方向成分は未記録です','保存されている部位別の二乗ノルムだけからは復元できません。全体へ切り替えると実測値を表示します。'));
    } else {
      const ys=direction.flatMap(r=>candidates().flatMap(c=>parallelDefs.map(([k])=>score(c.id,r)[k]))).filter(finite);
      const yRange=[Math.min(0,...ys)*1.12,Math.max(1,...ys)*1.15];
      direction.forEach(r=>plot(`parallel-${r}`,curveTraces(parallelDefs,r),{xaxis:xr,yaxis:{title:{text:'元方向への平行成分'},range:yRange,gridcolor:'#e7edf4'},shapes:[{type:'line',xref:'paper',x0:0,x1:1,y0:1,y1:1,line:{color:'#a6b4c5',width:1,dash:'dot'}}]},p=>selectCondition(p.customdata[0])));
    }
  }
  function metricsTable() {
    const c=candidate(),s=score(c.id),reg=activeRegime(),d=dataset();
    el('comparison-guide').textContent='その他UNetと同じ低mulの一律条件、高側と同じ一律条件を対照にします。数値の違いと画像評価は分けて判断します。';
    el('table-regime').textContent=`${d.views[state.part]} · dropout ${reg.toUpperCase()}${state.regime==='both'?'（表はOFF）':''}`;
    const contrasts=[...new Set([c.mul[0],c.mul[1]])].map(m=>d.candidates.find(v=>v.kind==='uniform'&&v.mul[0]===m)).filter(x=>x&&x.id!==c.id);
    el('quick-compare').innerHTML=contrasts.length?contrasts.map(v=>{
      const other=score(v.id),diff=pct(s.body,other.body);
      return `<span>${esc(v.label)}に対して、Body ${fmt(other.body)} → <b>${fmt(s.body)}</b>（${signed(diff,1)}%）</span>`;
    }).join('<br>'):'一律条件の曲線を基準に、attn2・TE高mul型の同じ位置の点と比較してください。';
    el('metrics-table').innerHTML='<thead><tr><th>診断条件 / 配分</th><th>P50</th><th>Body</th><th>Tail</th><th>元方向 P50</th><th>元方向 P05</th></tr></thead><tbody>'+candidates().map(v=>{
      const t=score(v.id);return `<tr data-candidate="${esc(v.id)}" class="${v.id===c.id?'selected':''}"><td class="label"><button class="rowlink" data-candidate="${esc(v.id)}">${esc(v.label)}</button><span class="sub">${esc(v.assignment_label)}</span></td>${['p50','body','tail','parallel_p50','parallel_p05'].map(k=>`<td>${fmt(t[k])}</td>`).join('')}</tr>`;
    }).join('')+'</tbody>';
    el('loss-note').textContent='この表は元画像グループを等重みで集計した分位値です。部位別のdは、その部位の元勾配を分母にしています。部位間の大小は全体への寄与率とは異なります。';
  }
  function globalLoss() {
    const d=dataset(),r=activeRegime(),c=candidate(),m=aggregate(d.samples,c.id,r,'all'),s=score(c.id);
    el('warmup-value').textContent=d.warmup.available?`${fmt(d.warmup.reduction_pct,1)}% 減少`:'対応する記録なし';
    el('warmup-value').classList.toggle('compact',!d.warmup.available);
    el('warmup-note').textContent=d.warmup.available?`raw MSE: ${fmt(d.warmup.before,4)} → ${fmt(d.warmup.after,4)}。量子化しない共通warmup。`:d.warmup.reason;
    el('change-title').textContent=`量子化による勾配変化 · ${d.views[state.part]}`;
    el('change-value').textContent=`Body ${fmt(s.body)}`;
    el('change-note').textContent=`P50 ${fmt(s.p50)} / Tail ${fmt(s.tail)}。${c.label} · ${r.toUpperCase()}`;
    el('direction-value').textContent=finite(s.parallel_p05)?fmt(s.parallel_p05):'部位別は未記録';
    el('direction-value').classList.toggle('compact',!finite(s.parallel_p05));
    el('direction-note').textContent=finite(s.parallel_p05)?'元の方向へ押す強さの弱い側。長期学習での定着や絵の固さを直接示す値ではありません。':'全体のグラフで確認できます。勾配差の大きさだけからは計算できません。';
    if(d.warmup.available) plot('warmup-chart',[{type:'bar',x:['学習前','warmup後'],y:[d.warmup.before,d.warmup.after],marker:{color:['#97b9d8','#387caf']},text:[fmt(d.warmup.before,4),fmt(d.warmup.after,4)],textposition:'outside',cliponaxis:false,hovertemplate:'%{x}<br>%{y:.6f}<extra></extra>'}],{showlegend:false,yaxis:{title:{text:'raw MSE'},rangemode:'tozero',range:[0,Math.max(d.warmup.before,d.warmup.after)*1.3],gridcolor:'#edf1f6'},margin:{l:72,r:25,t:25,b:50}});
    else empty('warmup-chart','この測定では、学習前の値が未記録です','対応する学習前後の記録がある場合だけ表示します。別測定の値では補いません。');
    const traces=[];
    shownRegimes().forEach((reg,idx)=>{
      const color=reg==='off'?'#3476ac':'#b16b36';
      const one=curveTraces([['delta',`C − B · ${reg.toUpperCase()}`,color]],reg,(v,key,rg)=>aggregate(d.samples,v.id,rg,'all')[key]);
      one.forEach(t=>{t.legendgroup=reg;t.line.dash=idx?'dot':'solid';});traces.push(...one);
    });
    plot('loss-chart',traces,{margin:{l:90,r:20,t:38,b:66},xaxis:{title:{text:'その他UNetのmul'},tickvals:[2.7,3.15,3.45,3.75,4.05],tickformat:'.2f',range:[2.6,4.14]},yaxis:{title:{text:'予測誤差の差 · C − B'},tickformat:'.6f',gridcolor:'#edf1f6',zeroline:true,zerolinewidth:1.5,zerolinecolor:'#8799ae'}},p=>selectCondition(p.customdata[0]));
    el('global-loss-note').textContent=`${d.loss_note} 選択中: B ${fmt(m.reference_loss,6)} → C ${fmt(m.quantized_loss,6)}、差 ${signed(m.delta,6)}（${r.toUpperCase()}・画像等重み）。負なら同じ入力で誤差が減少、正なら増加。追加学習の結果ではありません。`;
  }
  function details() {
    if(!el('detail-section')?.open)return;
    const d=dataset(),reg=activeRegime();
    const regimes=panelCharts('distribution-charts','distribution');
    const max=Math.max(.1,...regimes.flatMap(r=>candidates().map(c=>score(c.id,r).p90)).filter(finite))*1.1;
    regimes.forEach(r=>{
      const traces=[];
      candidates().forEach(c=>{
        const s=score(c.id,r),col=familyColor[c.kind];
        [[s.p10,s.p90,3,.35],[s.p25,s.p75,9,.8]].forEach(([a,b,w,op])=>traces.push({type:'scatter',mode:'lines',x:[a,b],y:[c.short,c.short],line:{color:col,width:w},opacity:op,showlegend:false,hoverinfo:'skip'}));
        traces.push({type:'scatter',mode:'markers',x:[s.p50],y:[c.short],marker:{color:col,size:10,symbol:c.symbol,line:{color:'#fff',width:1}},showlegend:false,customdata:[[c.id,c.label,fmt(s.p10),fmt(s.p25),fmt(s.p75),fmt(s.p90)]],hovertemplate:'%{customdata[1]}<br>P10 %{customdata[2]} / P25 %{customdata[3]}<br>P50 %{x:.4f}<br>P75 %{customdata[4]} / P90 %{customdata[5]}<extra></extra>'});
      });
      plot(`distribution-${r}`,traces,{margin:{l:130,r:20,t:15,b:52},xaxis:{title:{text:'相対勾配差 d'},range:[0,max],gridcolor:'#e7edf4'},yaxis:{autorange:'reversed',categoryorder:'array',categoryarray:candidates().map(c=>c.short)}},p=>selectCondition(p.customdata[0]));
    });
    el('interval-table').innerHTML='<thead><tr><th>条件</th><th>Body / 参考95%区間</th><th>Tail / 参考95%区間</th></tr></thead><tbody>'+candidates().map(c=>{
      const s=score(c.id),ci=s.overall_ci||{};
      return `<tr><td>${esc(c.label)}</td>${['body','tail'].map(k=>`<td>${fmt(s[k])} / ${ci[k]?ci[k].map(v=>fmt(v)).join(' 〜 '):'未算出'}</td>`).join('')}</tr>`;
    }).join('')+'</tbody>';
    const components=d.regimes[reg].contributions;
    if(!components || !candidates().every(c=>components[c.id]&&['te1','te2','attn2','unet_rest'].every(k=>finite(components[c.id][k])))) {
      empty('contributions','部位ごとの寄与は未記録です','この既存診断では全体の勾配差を保存しています。');
      empty('shares','構成比は表示できません','未記録の値を0%として表示しません。');return;
    }
    const labels=['TE1','TE2','attn2','その他UNet'],colors=['#577fb2','#91b8db','#8c72bc','#9bc6aa'];
    const values=candidates().map(c=>{const x=components[c.id];return [x.te1,x.te2,x.attn2,x.unet_rest];});
    [false,true].forEach(share=>{
      const traces=labels.map((name,i)=>({type:'bar',name,x:candidates().map(c=>c.short),y:values.map(v=>share?(v.reduce((a,b)=>a+b,0)>0?100*v[i]/v.reduce((a,b)=>a+b,0):null):v[i]),marker:{color:colors[i]},customdata:candidates().map(c=>[c.id,c.assignment_label]),hovertemplate:`${name}<br>%{x}<br>%{y:.4f}${share?'%':''}<br>%{customdata[1]}<extra></extra>`}));
      plot(share?'shares':'contributions',traces,{barmode:'stack',margin:{l:65,r:10,t:37,b:100},xaxis:{tickangle:-35},yaxis:{title:{text:share?'割合（%）':'平均d² · 共通の分母'},rangemode:'tozero',...(share?{range:[0,100]}:{}),gridcolor:'#edf1f6'},legend:{orientation:'h',x:0,y:1.16,font:{size:11}}},p=>selectCondition(p.customdata[0]));
    });
    el('contributions').previousElementSibling.textContent=`総量と内訳 · 平均d² · ${reg.toUpperCase()}`;
    el('shares').previousElementSibling.textContent=`内訳の割合 · 100% · ${reg.toUpperCase()}`;
  }
  function beginner() {
    const d=dataset();el('show-a').checked=state.showA;el('show-a').disabled=!d.candidates.some(c=>c.kind!=='uniform');
    el('fixed-scale').checked=state.fixed;
    document.querySelector('.family-legend').hidden=!d.candidates.some(c=>c.kind!=='uniform');
    el('body-badge').textContent=`BODY代表 mul ${fmt(d.body_mul,2)}`;
    el('body-reason').textContent=d.body_note+' 追加する低側2.70・3.15に対し、attn2・TEをこの値に固定した比較です。OFFで選んだ配分をONにも共通で用いています。';
    mainCharts();globalLoss();metricsTable();details();
  }
  function entities() {
    const d=dataset();
    if(state.entityKind==='image') return d.samples.map(s=>({id:s.id,label:`${s.id} · ${s.name}`,samples:[s]}));
    const groups=new Map();
    d.samples.forEach(s=>{
      const keys=state.entityKind==='tag'?s.tags:[s[state.entityKind]];
      keys.forEach(key=>{if(!groups.has(key))groups.set(key,[]);groups.get(key).push(s);});
    });
    return [...groups.entries()].map(([id,samples])=>({id,label:id,samples}));
  }
  function datasetScatters() {
    const d=dataset(),cid=state.condition;
    const all=d.samples.map(s=>({sample:s,m:perSample(s,cid)}));
    const sources=[...new Set(d.samples.map(s=>s.source))];
    const traces=kind=>sources.map((src,i)=>{
      const rows=all.filter(x=>x.sample.source===src);
      const xkey=kind==='learning'?'pre':'reference_loss',ykey=kind==='learning'?'improvement':'d';
      return {type:'scatter',mode:'markers',name:src,x:rows.map(x=>x.m[xkey]),y:rows.map(x=>x.m[ykey]),marker:{size:rows.map(x=>x.sample.id===state.entityId?13:9),color:sourceColors[i%sourceColors.length],opacity:.85,line:{color:'#fff',width:1}},customdata:rows.map(x=>[x.sample.id,x.sample.name,x.sample.source]),hovertemplate:`<b>%{customdata[0]} · %{customdata[1]}</b><br>%{customdata[2]}<br>${kind==='learning'?'学習前':'Bの誤差'} %{x:.5f}<br>${kind==='learning'?'改善率':'平均d'} %{y:.4f}${kind==='learning'?'%':''}<extra></extra>`};
    });
    if(d.warmup.available) plot('learning-chart',traces('learning'),{margin:{l:73,r:25,t:34,b:68},xaxis:{title:{text:'学習前のraw MSE'},rangemode:'tozero',gridcolor:'#e7edf4'},yaxis:{title:{text:'Warmupでの改善率（%）'},gridcolor:'#e7edf4',zeroline:true,zerolinecolor:'#9daec1'}},p=>selectSample(p.customdata[0]));
    else empty('learning-chart','学習開始前の記録がありません','この研究ログにはBとCの比較値を保存しています。A→Bを別実験の値で補わず、未記録として表示します。');
    const q=traces('quant');
    const scale=d.samples.flatMap(s=>d.candidates.flatMap(c=>Object.keys(d.regimes).map(r=>perSample(s,c.id,r))));
    const xmax=Math.max(.001,...scale.map(v=>v.reference_loss).filter(finite))*1.12;
    const ymax=Math.max(.001,...scale.map(v=>v.d).filter(finite))*1.12;
    plot('quant-chart',q,{margin:{l:67,r:25,t:34,b:68},xaxis:{title:{text:`Bの${d.loss_label}`},range:[0,xmax],gridcolor:'#e7edf4'},yaxis:{title:{text:`${d.views[state.part]}の平均d`},range:[0,ymax],gridcolor:'#e7edf4'}},p=>selectSample(p.customdata[0]));
    el('quant-axes').textContent=`量子化OFFの誤差 × ${d.views[state.part]}の平均勾配変化 · dropout ${activeRegime().toUpperCase()}`;
    datasetPlotSnapshot={learning:traces('learning').map(t=>({x:t.x,y:t.y})),quant:q.map(t=>({x:t.x,y:t.y}))};
  }
  function entityTable(list) {
    const metrics=list.map(e=>({...e,m:aggregate(e.samples,state.condition)}));
    if(state.sort!=='id')metrics.sort((a,b)=> (finite(b.m[state.sort])?b.m[state.sort]:-Infinity)-(finite(a.m[state.sort])?a.m[state.sort]:-Infinity));
    el('entity-count').textContent=`${metrics.length}件 · ${dataset().views[state.part]} / dropout ${activeRegime().toUpperCase()}`;
    el('entity-table').innerHTML='<thead><tr><th>対象</th><th>入力数</th><th>学習前</th><th>Warmup後</th><th>改善率</th><th>平均d</th><th>予測誤差の差 C−B</th></tr></thead><tbody>'+metrics.map(e=>`<tr data-sample="${esc(e.id)}" class="${e.id===state.entityId?'selected':''}"><td class="sample-column"><button class="rowlink" data-sample="${esc(e.id)}">${esc(e.label)}</button>${state.entityKind==='image'?`<span class="sub">${esc(e.samples[0].source)} · ${esc(e.samples[0].folder)}</span>`:''}</td><td>${e.samples.length}</td><td>${fmt(e.m.pre,4)}</td><td>${fmt(e.m.post,4)}</td><td>${finite(e.m.improvement)?fmt(e.m.improvement,1)+'%':'—'}</td><td>${fmt(e.m.d)}</td><td>${signed(e.m.delta,5)}</td></tr>`).join('')+'</tbody>';
  }
  function entityComparison(entity) {
    const d=dataset(),c=candidate(),rows=d.candidates.map(v=>({c:v,m:aggregate(entity.samples,v.id)}));
    const metric=state.compareMetric;
    const traces=['uniform','adaptive','reference'].map(kind=>{
      const rr=rows.filter(v=>v.c.kind===kind);
      return {type:'scatter',mode:kind==='uniform'?'lines+markers':'markers',name:{uniform:'一律',adaptive:'高mul型 · Body基準',reference:'高mul型 · 固定基準'}[kind],x:rr.map(v=>v.c.short),y:rr.map(v=>v.m[metric]),line:{width:2,color:familyColor[kind]},marker:{color:familyColor[kind],symbol:rr.map(v=>v.c.symbol),size:rr.map(v=>v.c.id===c.id?15:10),line:{color:'#fff',width:1}},customdata:rr.map(v=>[v.c.id,v.c.assignment_label,v.m.observations]),hovertemplate:'%{x}<br>%{customdata[1]}<br>%{y:.6f}<br>観測数 %{customdata[2]}<extra></extra>'};
    }).filter(t=>t.x.length);
    plot('entity-comparison',traces,{margin:{l:metric==='d'?72:90,r:20,t:37,b:115},xaxis:{type:'category',categoryorder:'array',categoryarray:d.candidates.map(v=>v.short),tickangle:-30},yaxis:{title:{text:metric==='d'?`${d.views[state.part]}の平均d`:'予測誤差の差 C − B'},...(metric==='d'?{rangemode:'tozero'}:{tickformat:'.6f'}),gridcolor:'#e7edf4',zeroline:true,zerolinecolor:'#98abbe'}},p=>selectCondition(p.customdata[0]));
    el('entity-title').textContent=state.entityKind==='image'?`選んだ画像の条件別比較 · ${entity.id}`:`選んだグループの条件別比較 · ${entity.samples.length}入力`;
    el('sample-name').textContent=entity.label;
    imageContext(state.entityKind==='image'?entity.samples[0]:null);
    const selected=rows.find(x=>x.c.id===c.id).m;
    el('sample-info').innerHTML=[['入力数',String(entity.samples.length)],['観測する部位',d.views[state.part]],['選択中の平均d',fmt(selected.d)],['量子化OFFの誤差',fmt(selected.reference_loss,6)],['量子化ONの誤差',fmt(selected.quantized_loss,6)],['予測誤差の差',signed(selected.delta,6)]].map(([k,v])=>`<div class="keyline"><span>${esc(k)}</span><b>${esc(v)}</b></div>`).join('');
    el('sample-loss-note').textContent=d.loss_note+' 各条件は同じ対象を使い、各入力を等重みで比較しています。';
    el('condition-table').innerHTML='<thead><tr><th>診断条件 / 配分</th><th>平均d</th><th>Bの予測誤差</th><th>Cの予測誤差</th><th>差 C−B</th><th>観測数</th></tr></thead><tbody>'+rows.map(({c:v,m})=>`<tr data-candidate="${esc(v.id)}" class="${v.id===c.id?'selected':''}"><td class="label"><button class="rowlink" data-candidate="${esc(v.id)}">${esc(v.label)}</button><span class="sub">${esc(v.assignment_label)}</span></td><td>${fmt(m.d)}</td><td>${fmt(m.reference_loss,6)}</td><td>${fmt(m.quantized_loss,6)}</td><td>${signed(m.delta,6)}</td><td>${m.observations}</td></tr>`).join('')+'</tbody>';
  }
  function imageContext(sample) {
    const preview=el('sample-preview'),note=el('sample-preview-note'),panel=el('sample-context-panel'),content=el('sample-contexts');
    preview.hidden=!sample?.image_uri;preview.onerror=null;note.textContent='';content.replaceChildren();
    panel.hidden=!sample;
    if(!sample){preview.removeAttribute('src');return;}
    if(!preview.hidden){
      preview.alt=sample.name;
      preview.onerror=()=>{preview.hidden=true;note.textContent='保存済みサムネイルと元画像を表示できません。測定値は引き続き参照できます。';};
      preview.src=sample.image_uri;
      note.textContent=(sample.preview_kind==='cached_thumbnail'?'保存済みサムネイル':'元画像を参照')+'。学習時のcrop・変換後の入力とは異なる場合があります。';
    }
    const contexts=sample.contexts||[];
    const warning=document.createElement('p');warning.className='small';
    warning.textContent=sample.inventory_match==='multiple_contexts'?'同じ画像に複数の学習設定があります。どれか1つのcaptionやwarmup反応を、この測定の条件として決めつけません。':contexts.length?'診断時に保存した学習設定です。':'この測定にはcaption・学習設定を保存していません。';
    content.append(warning);
    for(const [index,context] of contexts.entries()){
      const section=document.createElement('div');section.className='sample-context';
      const title=document.createElement('h4');title.textContent=`学習設定 ${index+1}`;section.append(title);
      const resolution=value=>Array.isArray(value)?value.join(' × '):value??'未記録';
      const fields=[['caption',context.caption??'未記録'],['class tokens',context.class_tokens??'未記録'],['dataset / subset',`${context.dataset_index??'—'} / ${context.subset_index??'—'}`],['subset group',context.subset_group??'未記録'],['解像度 / bucket',`${resolution(context.resolution)} / ${resolution(context.bucket_resolution)}`],['repeat / 正則化画像',`${context.num_repeats??'—'} / ${context.is_reg==null?'未記録':context.is_reg?'あり':'なし'}`],['提示 / 更新 / skip回数',`${context.presented_count??'—'} / ${context.updated_count??'—'} / ${context.skipped_count??'—'}`]];
      for(const [label,value] of fields){const line=document.createElement('p'),b=document.createElement('b'),text=document.createElement('span');b.textContent=label;text.textContent=String(value);line.append(b,text);section.append(line);}
      content.append(section);
    }
  }
  function datasetPage() {
    el('bin').value=state.bin;el('sort').value=state.sort;el('compare-metric').value=state.compareMetric;
    el('tag-button').disabled=!dataset().samples.some(s=>s.tags.length);
    el('tag-button').title=el('tag-button').disabled?'この測定にはタグを保存していません':'';
    el('entity-tabs').querySelectorAll('button').forEach(n=>n.classList.toggle('active',n.dataset.entity===state.entityKind));
    const list=entities();
    if(!list.some(e=>e.id===state.entityId))state.entityId=list[0]?.id;
    datasetScatters();entityTable(list);
    const selected=list.find(e=>e.id===state.entityId);if(selected)entityComparison(selected);
  }
  async function render() {
    normalize();shared();plots=[];
    if(PAGE==='beginner')beginner();else datasetPage();
    await Promise.all(plots);
    document.documentElement.dataset.ready='true';
    window.previewState={...state};window.previewPlotSnapshot=datasetPlotSnapshot;
  }
  function refresh() {
    const current=++version;document.documentElement.dataset.ready='false';
    busy=busy.then(()=>{if(current===version)return render();}).catch(error=>{
      document.documentElement.dataset.ready='error';
      console.error(error);
      let box=el('preview-error');if(!box){box=document.createElement('div');box.id='preview-error';box.className='note amber';document.querySelector('main').prepend(box);}
      box.textContent=`表示エラー: ${error.message}`;
      throw error;
    });
    window.previewReady=busy;return busy;
  }
  el('dataset').addEventListener('change',e=>{state.dataset=e.target.value;state.condition='';state.part='all';state.entityId=null;state.entityKind='image';refresh();});
  el('condition').addEventListener('change',e=>selectCondition(e.target.value));
  el('part').addEventListener('change',e=>{state.part=e.target.value;refresh();});
  el('regime').addEventListener('change',e=>{state.regime=e.target.value;refresh();});
  if(PAGE==='beginner') {
    el('show-a').addEventListener('change',e=>{state.showA=e.target.checked;refresh();});
    el('fixed-scale').addEventListener('change',e=>{state.fixed=e.target.checked;refresh();});
    el('detail-section').addEventListener('toggle',()=>{if(el('detail-section').open)refresh();});
  } else {
    el('bin').addEventListener('change',e=>{state.bin=e.target.value;refresh();});
    el('sort').addEventListener('change',e=>{state.sort=e.target.value;refresh();});
    el('compare-metric').addEventListener('change',e=>{state.compareMetric=e.target.value;refresh();});
    el('entity-tabs').addEventListener('click',e=>{const b=e.target.closest('[data-entity]');if(!b||b.disabled)return;state.entityKind=b.dataset.entity;state.entityId=null;refresh();});
  }
  document.addEventListener('click',e=>{
    const cand=e.target.closest('[data-candidate]');if(cand){selectCondition(cand.dataset.candidate);return;}
    const sample=e.target.closest('[data-sample]');if(sample){state.entityId=sample.dataset.sample;refresh();}
  });
  refresh();
})();
