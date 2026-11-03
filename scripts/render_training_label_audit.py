"""Render exact sampled training captions without modifying their text."""
import json,html,textwrap,zipfile
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
ROOT=Path('outputs/training_label_audit_20261103')
def esc(x):return html.escape(str(x if x is not None else '—'))
def main():
 rows=json.load(open(ROOT/'sample.json'));audit=json.load(open(ROOT/'audit.json'))
 archives=json.load(open(ROOT/'archive_audit.json')) if (ROOT/'archive_audit.json').exists() else []
 byid={r['sample_id']:r for r in archives}
 groups={'dual':'双标签随机样本','filename_only':'仅文件名标签随机样本','diagnostic':'额外疑点样本（非随机）'}
 parts=['''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MCGen 训练标签与文件名核对</title><style>body{font:16px system-ui;max-width:1400px;margin:32px auto;padding:0 24px;background:#f6f5ef;color:#23382f}h1{margin-bottom:8px}p{line-height:1.65}.muted{color:#65776d}.toolbar{position:sticky;top:0;padding:12px;background:#f6f5ef;border-bottom:1px solid #ccd6cc;z-index:1}select,input{font:inherit;padding:8px;margin-right:8px}.card{display:grid;grid-template-columns:136px 1fr;gap:22px;padding:24px;margin:18px 0;border:1px solid #d8dfd5;border-radius:12px;background:white}.sprite{width:128px;height:128px;image-rendering:pixelated;background:#808080}.card h2{font-size:18px;margin:0 0 8px;overflow-wrap:anywhere}.labels{display:grid;grid-template-columns:140px 1fr;gap:12px}.label{font-size:14px;color:#65776d}.text{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.55}.auto{background:#edf4eb;padding:10px;border-radius:6px}.missing{background:#fff4df;padding:10px}.badge{font-size:12px;color:#65776d;margin:8px 0}details{margin-top:18px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}a{color:#236d50}@media(max-width:700px){.card{grid-template-columns:1fr}.labels{grid-template-columns:1fr}}</style><h1>训练图片 · 两种文本标签 · 文件名来源</h1><p>固定随机种子抽取 <b>48 张双标签图片 + 16 张仅文件名标签图片</b>；另附 8 张疑点图片。此处刻意多展示双标签样本，不代表全库比例。当前训练库 324,921 张，其中 10,000 张有双标签（约 3.08%）。所有文字按现有记录原样展示，没有重新标注或润色。</p><p class="muted">双标签图片训练时随机选择文件名标签或自动描述；仅文件名图片不使用自动描述。模型统一风格前缀未重复展示。灰底只用于观察透明图片。</p><p><b>核对结论：</b>本地训练文件名与下载数据集一致。原包抽查发现目录语义丢失：skull.png 来自 elixir/cap/；shield_…png 来自 tool/cleaver/；fiery_ingot.png 来自 sword/pommel/。文件名可以是真实的，但不能直接当作完整物品名称。sensing_spell_book_e.png 则由包内 emissive.properties 明确标记为发光层。<a href="source_evidence.json">查看证据</a></p><div class="toolbar"><select id="group"><option value="">全部 72 张</option>''']
 parts += [f'<option value="{g}">{v}</option>' for g,v in groups.items()]
 parts += ['</select><input id="search" placeholder="搜索文件名或标签"><a href="labels.csv">CSV</a> · <a href="sample.json">完整记录</a> · <a href="audit.json">本地链路核对</a> · <a href="archive_audit.json">原资源包核对</a></div>']
 for r in rows:
  n=r['sample_id'];auto=r['automatic_training_label'];raw=r['raw_automatic_candidate'];ar=byid.get(n)
  parts.append(f'<article class="card" data-group="{r["group"]}"><div><img class="sprite" src="{r["image"]}"><div class="badge">#{n:03d} · {groups[r["group"]]}</div></div><div><h2>{esc(r["source_filename"])}</h2><div class="badge">Mod：{esc(r["raw_metadata"]["mod_slug"])} · 原始行 {r["source_row"]}</div><div class="labels"><div class="label">文件名提取标签</div><div class="text">{esc(r["filename_label"])}</div><div class="label">实际训练自动描述</div><div class="text {"auto" if auto else "missing"}">{esc(auto) if auto else esc(r["automatic_status"])}</div></div>')
  if not auto and raw:parts.append(f'<details><summary>查看未用于训练的自动描述候选</summary><p>{esc(raw)}</p><p>质量标记：{esc(r["quality_flags"])}</p></details>')
  parts.append('<details><summary>查看来源证据与同图别名</summary>')
  parts.append(f'<p>下载的 Parquet 文件名：{esc(r["raw_parquet"]["filename"])}<br>原图像素与训练图：{"一致" if r["raw_parquet"]["pixels_match_training"] else "不一致"}<br>下载数据集附带的旧文本（不是当前自动标签）：{esc(r["raw_parquet"]["original_text"])}</p>')
  if ar:
   parts.append(f'<p>原包内同名路径：{esc(ar["basename_matches"])}<br>Alpha 及全部可见 RGB 像素一致的路径：{esc(ar.get("visible_rgba_matches",[]))}<br>完整 RGBA（包括透明区 RGB）也一致的路径：{esc(ar["pixel_matches"])}<br>下载错误：{esc(ar["download_error"])}</p>')
  else:parts.append('<p>本样本未下载原资源包；已核对到下载的 Parquet。</p>')
  parts.append(f'<pre>{esc(json.dumps({"source_aliases":r["source_aliases"],"raw_metadata":r["raw_metadata"],"raw_parquet":r["raw_parquet"],"target_hash":r["target_hash"]},ensure_ascii=False,indent=2))}</pre></details></div></article>')
 parts.append('''<script>function filter(){let g=document.querySelector('#group').value,q=document.querySelector('#search').value.toLowerCase();document.querySelectorAll('.card').forEach(e=>e.style.display=(!g||e.dataset.group===g)&&e.textContent.toLowerCase().includes(q)?'grid':'none')}document.querySelector('#group').onchange=filter;document.querySelector('#search').oninput=filter;</script></html>''')
 (ROOT/'index.html').write_text(''.join(parts))
 font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',17);title=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',21)
 def wrap(text,width):
  lines=[];line=''
  for word in str(text).split(' '):
   if font.getlength(line+' '+word)>width and line:lines.append(line);line=word
   else:line=(line+' '+word).strip()
  if line:lines.append(line)
  return lines
 for start in range(0,len(rows),8):
  batch=rows[start:start+8];sheet=Image.new('RGB',(1600,60+4*300),'#f6f5ef');d=ImageDraw.Draw(sheet)
  d.text((24,14),f'训练标签核对 · {start:03d}—{start+len(batch)-1:03d} · 文字原样展示',font=title,fill='#23382f')
  for j,r in enumerate(batch):
   x=(j%2)*800;y=60+(j//2)*300;d.rounded_rectangle((x+10,y+5,x+790,y+290),10,fill='white')
   im=Image.open(ROOT/r['image']).convert('RGBA').resize((112,112),Image.Resampling.NEAREST);bg=Image.new('RGBA',im.size,(128,128,128,255));bg.alpha_composite(im);sheet.paste(bg.convert('RGB'),(x+25,y+20))
   d.text((x+25,y+145),f'#{r["sample_id"]:03d}',font=title,fill='#23382f')
   labels=[('文件名：',r['source_filename']),('文件名标签：',r['filename_label']),('自动描述：',r['automatic_training_label'] or '无训练用自动描述；仅用文件名标签')]
   yy=y+15
   for label,text in labels:
    lines=wrap(label+text,620)
    for line in lines:d.text((x+155,yy),line,font=font,fill='#23382f');yy+=24
    yy+=9
   assert yy<y+290,(r['sample_id'],yy-y)
  sheet.save(ROOT/f'preview_{start//8:02d}.png')
 print('Rendered 72 cards and 9 preview sheets.')
if __name__=='__main__':main()
