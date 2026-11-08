'use strict';
const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const number = value => Number(value || 0).toLocaleString('zh-CN');
const labels = {pending:'待审核',keep:'配对可用',reject:'已排除',unsure:'稍后决定'};
const categories = {sword:'剑',pickaxe:'镐',axe:'斧',hoe:'锄',shovel:'锹',helmet:'头盔',chestplate:'胸甲',leggings:'护腿',boots:'靴子',shield:'盾牌',bow:'弓',crossbow:'弩',gem:'宝石',crystal:'晶体',ingot:'锭',nugget:'粒',dust:'粉末',gear:'齿轮',potion:'药水',bottle:'瓶子',apple:'苹果',door:'门',other:'其他'};
let preferences = {};
try { preferences = JSON.parse(localStorage.getItem('mc-pair-review') || '{}'); } catch (_) {}
const state = {split:preferences.split || 'train',filter:preferences.filter || 'pending',selected:preferences.selected || 'all',category:preferences.category || '',search:preferences.search || '',page:preferences.page || 1,view:preferences.view || 'focus',active:null,current:null,items:[],total:0,limit:24,revision:0,summary:null,request:0,busy:false,bulk:new Set(),lastTarget:preferences.target || null};
let saveQueue = Promise.resolve(), toastTimer, searchTimer;
const noteTimers = new Map();

function remember() {
  try { localStorage.setItem('mc-pair-review',JSON.stringify({split:state.split,filter:state.filter,selected:state.selected,category:state.category,search:state.search,page:state.page,view:state.view,target:state.active,autoNext:$('auto-next').checked})); } catch (_) {}
}
function toast(text,error=false) {
  clearTimeout(toastTimer);$('toast').textContent=text;$('toast').className=error?'error':'';$('toast').hidden=false;
  toastTimer=setTimeout(()=>$('toast').hidden=true,error?6500:2800);
}
function saving(busy,error=false) {
  state.busy=busy;const target=$('save-status');target.className='save-status'+(error?' error':busy?' busy':'');
  target.replaceChildren(Object.assign(document.createElement('i'),{}),document.createTextNode(error?'保存失败':busy?'正在保存…':'已保存到本机'));
  document.querySelectorAll('.selection-switch').forEach(button=>button.disabled=busy || state.current?.split!=='train');
}
async function api(path,options={}) {
  const response=await fetch(path,{...options,headers:{'Content-Type':'application/json',...(options.headers||{})}});
  const data=await response.json().catch(()=>({detail:'服务暂时不可用'}));
  if(!response.ok) throw new Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail));
  return data;
}
function countsForSplit(summary) {
  const sources=state.split==='all'?Object.values(summary.counts):[summary.counts[state.split]];
  return sources.reduce((result,c)=>{for(const k of Object.keys(labels))result[k]+=c[k]||0;return result;},{pending:0,keep:0,reject:0,unsure:0});
}
async function loadSummary() {
  const data=await api('/api/summary');state.summary=data;state.revision=data.revision;
  const counts=countsForSplit(data),total=Object.values(counts).reduce((a,b)=>a+b,0),reviewed=total-counts.pending;
  for(const key of Object.keys(labels))$('count-'+key).textContent=number(counts[key]);
  $('count-all').textContent=number(total);$('stat-pending').textContent=number(counts.pending);$('stat-keep').textContent=number(counts.keep);
  $('stat-selected').innerHTML=number(data.train_selected_pairs)+'<em>对</em>';
  $('stat-selected-targets').textContent='涉及 '+number(data.train_selected_targets)+' 个 MC 物品';
  const percentage=total?Math.round(reviewed/total*100):0;$('stat-progress').innerHTML=percentage+'<em>%</em>';$('progress-fill').style.width=percentage+'%';
  $('selected-sidebar-count').textContent=number(data.train_selected_pairs);$('export-count').textContent=number(data.train_selected_pairs);
  $('dataset-size').textContent=number(data.targets)+' 个物品 · '+number(data.pairs)+' 个配对';
  $('export-pairs').textContent=number(data.train_selected_pairs);$('export-targets').textContent='涉及 '+number(data.train_selected_targets)+' 个不同 MC 物品';
  $('create-export').disabled=!data.train_selected_pairs;
  if($('category').options.length===1) for(const name of data.categories)$('category').add(new Option(categories[name] || name,name));
  $('category').value=state.category;
  return data;
}
function updateFilters() {
  $('split').value=state.split;$('category').value=state.category;$('search').value=state.search;$('selection-filter').value=state.selected;
  document.querySelectorAll('[data-state]').forEach(button=>button.classList.toggle('active',button.dataset.state===state.filter));
  $('selected-filter').classList.toggle('active',state.selected==='yes');
  document.querySelectorAll('[data-view]').forEach(button=>button.classList.toggle('active',button.dataset.view===state.view));
}
function queueItem(item) {
  const chosen=item.pairs.filter(p=>p.train_selected).length,reviewed=item.pairs.filter(p=>p.state!=='pending').length;
  return `<button class="queue-item ${item.target_hash===state.active?'active':''}" data-item="${item.target_hash}" title="${esc(item.prompt)}"><img src="${item.target_url}" alt="" loading="lazy"><span class="queue-text"><strong>${esc(item.prompt)}</strong><small>${reviewed}/${item.pairs.length} 已审核${chosen?' · 已选 '+chosen+' 对':''}</small></span>${chosen?'<span class="selected-mini">✓</span>':''}</button>`;
}
function renderQueue() {
  $('queue-list').innerHTML=state.items.map(queueItem).join('');$('queue-count').textContent=number(state.total);
  $('queue-page').textContent=state.page+' / '+Math.max(1,Math.ceil(state.total/state.limit));
  $('page-prev').disabled=state.page<=1;$('page-next').disabled=state.page*state.limit>=state.total;
}
async function loadList({preserve=true,target=null}={}) {
  const ticket=++state.request;
  const params=new URLSearchParams({split:state.split,state:state.filter,search:state.search,category:state.category,selected:state.selected,page:state.page,limit:state.limit});
  const data=await api('/api/items?'+params);
  if(ticket!==state.request)return;
  state.items=data.items;state.total=data.total;state.revision=Math.max(state.revision,data.revision);state.bulk.clear();
  if(!data.items.length && data.total && state.page>1){state.page=Math.max(1,Math.ceil(data.total/state.limit));return loadList({preserve:false});}
  renderQueue();renderGrid();$('loading').hidden=true;
  const wanted=target || (preserve?state.active:null);
  if(wanted){
    const listed=state.items.find(i=>i.target_hash===wanted);
    if(listed)showItem(listed);
    else if(preserve && state.current)showItem(await api('/api/items/'+wanted));
    else if(state.items[0])showItem(state.items[0]);
    else{state.current=null;state.active=null;renderVisibility();}
  }else if(state.items[0])showItem(state.items[0]);
  else{state.current=null;state.active=null;renderVisibility();}
  remember();
}
function renderVisibility() {
  const available=Boolean(state.current || state.items.length);
  $('empty').hidden=available;$('focus-view').hidden=!(state.view==='focus'&&state.current);$('grid-view').hidden=!(state.view==='grid'&&state.items.length);
}
function referenceCard(pair,index) {
  const generator=pair.generator==='flux'?'FLUX':pair.generator==='qwen'?'Qwen':pair.generator;
  const chosen=pair.train_selected,training=state.current.split==='train';
  const subtitle=!training?'验证配对，不加入训练':chosen?'已加入本次训练清单':pair.state==='keep'?'可用，但尚未选入训练':'未加入训练清单';
  const reasons=['配对不一致','组件或图集','形状不符','配色不符','背景或遮挡','细节丢失','其他'];
  return `<article class="reference-card ${chosen?'selected':''} ${pair.state==='reject'?'rejected':''}" data-pair-card="${pair.pair_id}">
    <div class="card-heading"><span class="generator ${pair.generator==='qwen'?'qwen':''}"><i class="generator-dot"></i>${esc(generator)} 参考图</span><span class="header-selection"><span>用于训练</span><button class="switch selection-switch" role="switch" aria-label="${esc(generator)} 用于本次训练" aria-checked="${chosen}" data-select-pair="${pair.pair_id}" ${!training||state.busy?'disabled':''}></button></span></div>
    <button class="image-stage" data-zoom-pair="${pair.pair_id}" title="放大 ${esc(generator)} 参考图"><img src="${pair.reference_url}" alt="${esc(generator)} 高清参考图"><span class="zoom-hint">↗ 放大</span></button>
    <div class="selection-row"><div><strong>${training?(chosen?'已选入本次训练':'未选入本次训练'):'验证集配对'}</strong><small>${subtitle}</small></div><span class="review-badge ${pair.state}">${labels[pair.state]}</span></div>
    <div class="quality-controls">${['keep','reject','unsure'].map(k=>`<button class="quality-button ${k} ${pair.state===k?'active':''}" data-quality="${k}" data-pair="${pair.pair_id}">${{keep:'可用',reject:'排除',unsure:'待定'}[k]}</button>`).join('')}</div>
    <div class="card-shortcuts"><kbd>${index*2+1}</kbd> 选入训练 <span>·</span><kbd>${index*2+2}</kbd> 排除</div>
    <details class="card-notes"><summary>${pair.note?'已添加备注':'备注与排除原因'}</summary><select data-reason="${pair.pair_id}" aria-label="${esc(generator)} 原因"><option value="">原因（可选）</option>${reasons.map(reason=>`<option ${pair.reason===reason?'selected':''}>${reason}</option>`).join('')}</select><textarea maxlength="2000" data-note="${pair.pair_id}" placeholder="记录这张参考图的问题，或保留的设计…" aria-label="${esc(generator)} 备注">${esc(pair.note)}</textarea></details>
  </article>`;
}
function showItem(item) {
  state.current=item;state.active=item.target_hash;
  $('item-prompt').textContent=item.prompt;$('item-filename').textContent=item.filename || '原始文件名未提供';
  $('item-split').textContent=item.split==='train'?'训练配对':'验证配对';$('item-split').className='badge'+(item.split==='val'?' val':'');
  $('native-size').textContent=item.native_size+' × '+item.native_size;$('target-image').src=item.target_url;
  $('pixel-grid').style.backgroundSize=(100/item.native_size)+'% '+(100/item.native_size)+'%';
  $('reference-cards').innerHTML=item.pairs.map(referenceCard).join('');
  const position=state.items.findIndex(i=>i.target_hash===item.target_hash);
  $('item-position').textContent=position>=0?number((state.page-1)*state.limit+position+1)+' / '+number(state.total):'已处理';
  $('choose-both').innerHTML=(item.split==='train'?'✓ 选入这'+(item.pairs.length===2?'两':item.pairs.length)+'对':'✓ 标记配对可用')+' <kbd>A</kbd>';
  $('reject-both').innerHTML='排除这'+(item.pairs.length===2?'两':item.pairs.length)+'对 <kbd>X</kbd>';
  renderQueue();renderVisibility();remember();
}
function gridCard(item) {
  const chosen=item.pairs.filter(p=>p.train_selected).length;
  return `<article class="grid-card ${state.bulk.has(item.target_hash)?'bulk-selected':''}"><div class="grid-card-heading"><input class="grid-select" type="checkbox" data-bulk-item="${item.target_hash}" aria-label="批量选择 ${esc(item.prompt)}" ${state.bulk.has(item.target_hash)?'checked':''}><strong>${esc(item.prompt)}</strong><button class="icon-button" data-open-item="${item.target_hash}" aria-label="逐项查看 ${esc(item.prompt)}">↗</button></div><div class="grid-pictures"><figure><img loading="lazy" src="${item.target_url}" alt="MC原图"><figcaption>MC 原图</figcaption></figure>${item.pairs.map(pair=>`<figure><img loading="lazy" src="${pair.reference_url}?size=320" alt="${esc(pair.generator)}参考图"><figcaption>${esc(pair.generator==='flux'?'FLUX':'Qwen')}<label title="用于本次训练"><input type="checkbox" data-grid-train="${pair.pair_id}" aria-label="${esc(item.prompt)} ${esc(pair.generator)} 用于训练" ${pair.train_selected?'checked':''} ${item.split!=='train'?'disabled':''}>训练</label></figcaption></figure>`).join('')}</div><div class="grid-item-footer"><span>${chosen?'✓ 已选 '+chosen+' 对训练配对':'尚未用于训练'}</span><button data-open-item="${item.target_hash}">查看详情 →</button></div></article>`;
}
function renderGrid() {
  $('grid-title').textContent='当前页 · '+state.items.length+' 个物品';$('grid-items').innerHTML=state.items.map(gridCard).join('');
  $('bulk-count').textContent=state.bulk.size?'已勾选 '+state.bulk.size+' 个物品':'';
  $('bulk-choose').disabled=!state.bulk.size;$('bulk-reject').disabled=!state.bulk.size;
  $('bulk-choose').textContent=state.split==='val'?'标记可用':'选入训练';
}
function pairById(id){return state.current?.pairs.find(p=>p.pair_id===id) || state.items.flatMap(i=>i.pairs).find(p=>p.pair_id===id);}
function mutate(ids,payload,{advance=false,message='',refreshDetail=true}={}) {
  const target=state.active;
  saveQueue=saveQueue.then(async()=>{
    saving(true);
    const response=await api('/api/decisions',{method:'POST',body:JSON.stringify({pair_ids:ids,...payload,expected_revision:state.revision})});
    state.revision=response.revision;
    await loadSummary();
    const item=target?await api('/api/items/'+target):null;
    const shouldAdvance=advance && $('auto-next').checked && item?.pairs.every(p=>p.state!=='pending');
    const oldIndex=state.items.findIndex(i=>i.target_hash===target);
    const next=state.items[oldIndex+1]?.target_hash;
    if(refreshDetail)await loadList({preserve:!shouldAdvance,target:shouldAdvance?next:target});
    else{
      const params=new URLSearchParams({split:state.split,state:state.filter,search:state.search,category:state.category,selected:state.selected,page:state.page,limit:state.limit});
      const listed=await api('/api/items?'+params);state.items=listed.items;state.total=listed.total;renderQueue();
      if(state.current?.target_hash===target && item)state.current=item;
    }
    saving(false);if(message)toast(message);
  }).catch(async error=>{saving(false,true);toast(error.message,true);try{await loadSummary();}catch(_){}});
  return saveQueue;
}
function selectPair(id) {
  if(state.busy)return;
  const pair=pairById(id);if(!pair)return;
  const chosen=!pair.train_selected;
  return mutate([id],chosen?{state:'keep',train_selected:true}:{train_selected:false},{advance:chosen,message:chosen?'已选入本次训练':'已取消训练选用；可用标记保留'});
}
function quality(id,value) {
  if(state.busy)return;
  return mutate([id],{state:value},{advance:true,message:value==='keep'?'已标记可用，训练选用可单独勾选':value==='reject'?'已排除，不会用于训练':'已暂缓，不会用于训练'});
}
function chooseBoth(){if(!state.current||state.busy)return;return mutate(state.current.pair_ids,state.current.split==='train'?{state:'keep',train_selected:true}:{state:'keep'},{advance:true,message:state.current.split==='train'?'本项配对已选入训练':'已标记验证配对可用'});}
function rejectBoth(){if(!state.current||state.busy)return;return mutate(state.current.pair_ids,{state:'reject'},{advance:true,message:'本项配对已全部排除'});}
async function navigate(direction) {
  if(!state.items.length)return;
  const i=state.items.findIndex(item=>item.target_hash===state.active),next=i+direction;
  if(next>=0&&next<state.items.length)showItem(state.items[next]);
  else if(direction>0&&state.page*state.limit<state.total){state.page++;await loadList({preserve:false});}
  else if(direction<0&&state.page>1){state.page--;await loadList({preserve:false});if(state.items.length)showItem(state.items.at(-1));}
  else if(i===-1 && direction>0)showItem(state.items[0]);
}
async function changeFilter() {
  state.page=1;state.bulk.clear();updateFilters();remember();
  try{await loadSummary();await loadList({preserve:false});}catch(error){toast(error.message,true);}
}
async function undo(){
  if(state.busy)return;
  saveQueue=saveQueue.then(async()=>{
    saving(true);const response=await api('/api/undo',{method:'POST',body:JSON.stringify({expected_revision:state.revision})});
    state.revision=response.revision;await loadSummary();await loadList({preserve:true});saving(false);toast('已撤销上次操作');
  }).catch(error=>{saving(false,true);toast(error.message,true);});
  return saveQueue;
}
function zoom(url,title,pixelated=false){$('zoom-title').textContent=title;$('zoom-image').src=url;$('zoom-image').className=pixelated?'pixelated':'';$('zoom-dialog').showModal();}
function queueNote(id,note){
  clearTimeout(noteTimers.get(id)?.timer);
  const job={id,note};job.timer=setTimeout(()=>{noteTimers.delete(id);mutate([id],{note},{refreshDetail:false});},650);
  noteTimers.set(id,job);
}
function flushNotes(){for(const job of noteTimers.values()){clearTimeout(job.timer);mutate([job.id],{note:job.note},{refreshDetail:false});}noteTimers.clear();}

document.addEventListener('click',event=>{
  const stateButton=event.target.closest('[data-state]');if(stateButton){state.filter=stateButton.dataset.state;state.selected='all';changeFilter();return;}
  const item=event.target.closest('[data-item]');if(item){const found=state.items.find(i=>i.target_hash===item.dataset.item);if(found)showItem(found);return;}
  const open=event.target.closest('[data-open-item]');if(open){const found=state.items.find(i=>i.target_hash===open.dataset.openItem);state.view='focus';updateFilters();if(found)showItem(found);return;}
  const select=event.target.closest('[data-select-pair]');if(select){selectPair(select.dataset.selectPair);return;}
  const qualityButton=event.target.closest('[data-quality]');if(qualityButton){quality(qualityButton.dataset.pair,qualityButton.dataset.quality);return;}
  const zoomButton=event.target.closest('[data-zoom-pair]');if(zoomButton){const pair=pairById(zoomButton.dataset.zoomPair);if(pair)zoom(pair.reference_url+'?size=2048',pair.generator.toUpperCase()+' · '+state.current.prompt);return;}
  const close=event.target.closest('[data-close]');if(close){$(close.dataset.close).close();return;}
  const view=event.target.closest('[data-view]');if(view){state.view=view.dataset.view;updateFilters();renderGrid();renderVisibility();remember();return;}
  const background=event.target.closest('[data-background]');if(background){document.querySelectorAll('[data-background]').forEach(b=>b.classList.toggle('active',b===background));const stage=$('target-zoom');stage.classList.toggle('checker',background.dataset.background==='checker');stage.style.backgroundColor={checker:'',white:'#fff',dark:'#27382c'}[background.dataset.background];return;}
});
document.addEventListener('change',event=>{
  const bulk=event.target.closest('[data-bulk-item]');if(bulk){bulk.checked?state.bulk.add(bulk.dataset.bulkItem):state.bulk.delete(bulk.dataset.bulkItem);renderGrid();return;}
  const training=event.target.closest('[data-grid-train]');if(training){selectPair(training.dataset.gridTrain);return;}
  const reason=event.target.closest('[data-reason]');if(reason){mutate([reason.dataset.reason],{reason:reason.value},{refreshDetail:false});}
});
document.addEventListener('input',event=>{const field=event.target.closest('[data-note]');if(field)queueNote(field.dataset.note,field.value);});
$('target-zoom').onclick=()=>state.current&&zoom(state.current.target_url,'MC 原图 · '+state.current.prompt,true);
$('show-grid').onchange=event=>$('pixel-grid').hidden=!event.target.checked;
$('item-next').onclick=()=>navigate(1);$('item-prev').onclick=()=>navigate(-1);
$('page-next').onclick=async()=>{state.page++;await loadList({preserve:false});};$('page-prev').onclick=async()=>{if(state.page>1)state.page--;await loadList({preserve:false});};
$('choose-both').onclick=chooseBoth;$('reject-both').onclick=rejectBoth;$('undo').onclick=undo;
$('skip-item').onclick=()=>state.current&&!state.busy&&mutate(state.current.pair_ids,{state:'unsure'},{advance:true,message:'已放入稍后决定'});
$('auto-next').checked=preferences.autoNext!==false;$('auto-next').onchange=remember;
$('selected-filter').onclick=()=>{state.selected='yes';state.filter='all';state.split='train';changeFilter();};
$('show-all').onclick=()=>{state.filter='all';state.selected='all';state.search='';state.category='';changeFilter();};
$('split').onchange=event=>{state.split=event.target.value;changeFilter();};$('category').onchange=event=>{state.category=event.target.value;changeFilter();};$('selection-filter').onchange=event=>{state.selected=event.target.value;changeFilter();};
$('search').oninput=event=>{clearTimeout(searchTimer);state.search=event.target.value;searchTimer=setTimeout(changeFilter,230);};
$('shortcuts').onclick=()=>$('help-dialog').showModal();
for(const dialog of document.querySelectorAll('dialog'))dialog.addEventListener('click',event=>{if(event.target===dialog){const r=dialog.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)dialog.close();}});
$('bulk-choose').onclick=()=>{
  const items=state.items.filter(i=>state.bulk.has(i.target_hash));const train=items.every(i=>i.split==='train');
  if(!train&&items.some(i=>i.split==='train')){toast('请分别处理训练配对和验证配对。',true);return;}
  mutate(items.flatMap(i=>i.pair_ids),train?{state:'keep',train_selected:true}:{state:'keep'},{message:train?'勾选物品的全部参考版本已选入训练':'已标记所选验证配对可用'});
};
$('bulk-reject').onclick=()=>mutate(state.items.filter(i=>state.bulk.has(i.target_hash)).flatMap(i=>i.pair_ids),{state:'reject'},{message:'勾选物品的全部配对已排除'});
$('export-button').onclick=async()=>{flushNotes();await saveQueue;await loadSummary();$('export-result').hidden=true;$('export-progress').textContent='';$('export-dialog').showModal();};
$('create-export').onclick=async()=>{
  flushNotes();await saveQueue;$('create-export').disabled=true;$('export-progress').textContent='正在核验图片并生成清单…';
  try{
    const result=await api('/api/exports',{method:'POST',body:JSON.stringify({validation_policy:$('validation-policy').value})});
    $('export-result').hidden=false;$('export-summary').textContent=`训练 ${number(result.train_pairs)} 对 / ${number(result.train_targets)} 个物品；验证 ${number(result.val_pairs)} 对。`;
    $('download-selection').href=result.download_url;$('download-archive').href=result.archive_url;$('selection-path').textContent=result.selection_path;
    const quote=text=>"'"+text.replaceAll("'","'\\''")+"'";
    $('training-command').textContent='python scripts/prepare_reviewed_reference.py \\\n  --selection '+quote(result.selection_path)+' \\\n  --out outputs/reference_reviewed';
    $('export-progress').textContent='已核验并保存';
  }catch(error){$('export-progress').textContent='生成失败';toast(error.message,true);}finally{$('create-export').disabled=false;}
};
$('copy-path').onclick=async()=>{try{await navigator.clipboard.writeText($('selection-path').textContent);toast('已复制本机清单位置');}catch(_){toast('可直接选中上方清单位置复制。');}};
document.addEventListener('keydown',event=>{
  if(document.querySelector('dialog[open]')||event.ctrlKey||event.altKey||event.metaKey||event.repeat)return;
  if(event.target.closest('input,textarea,select,[contenteditable]'))return;
  const key=event.key.toLowerCase();
  if(key==='/'){event.preventDefault();$('search').focus();return;}
  if(key==='?'){event.preventDefault();$('help-dialog').showModal();return;}
  if(state.view!=='focus'||!state.current)return;
  if(key==='arrowright'){event.preventDefault();navigate(1);}else if(key==='arrowleft'){event.preventDefault();navigate(-1);}
  else if(key==='a'){event.preventDefault();chooseBoth();}else if(key==='x'){event.preventDefault();rejectBoth();}
  else if(key==='s'){event.preventDefault();$('skip-item').click();}else if(key==='u'){event.preventDefault();undo();}
  else if(['1','2','3','4'].includes(key)){
    const n=Number(key)-1,pair=state.current.pairs[Math.floor(n/2)];if(!pair)return;event.preventDefault();
    if(n%2)quality(pair.pair_id,'reject');else if(state.current.split==='train')selectPair(pair.pair_id);else quality(pair.pair_id,'keep');
  }
});
async function init(){
  updateFilters();
  try{await loadSummary();await loadList({preserve:false,target:state.lastTarget});saving(false);}
  catch(error){$('loading').hidden=true;$('empty').hidden=false;$('empty').querySelector('h2').textContent='暂时无法读取配对数据';$('empty').querySelector('p').textContent=error.message;saving(false,true);}
}
init();
