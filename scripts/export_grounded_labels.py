"""Export revised labels and review page without carrying old labels into training."""
import csv,html,json
from collections import Counter
from pathlib import Path
from mcgen.catalog import atomic_json
ROOT=Path('outputs/grounded_label_revision_20261103')
def export():
 protocol=json.load(open(ROOT/'protocol.json'));labels={}
 for path in ROOT.glob('labels_*.jsonl'):
  for line in path.open():
   r=json.loads(line);assert r['signature']==protocol['signature'];assert r['target_hash'] not in labels,'Duplicate annotation';labels[r['target_hash']]=r
 stats=Counter(r['decision'] for r in labels.values());summary={'total':protocol['total'],'processed':len(labels),'pending':protocol['total']-len(labels),'decisions':dict(stats),'file_labels':sum(bool(r.get('file_label')) for r in labels.values()),'visual_labels':sum(bool(r.get('visual_label')) for r in labels.values()),'status':'complete' if len(labels)==protocol['total'] else 'in_progress','automatic_weak_labels':True}
 atomic_json(ROOT/'summary.json',summary)
 with (ROOT/'corrected_labels.jsonl').open('w') as out:
  for r in labels.values():out.write(json.dumps(r,ensure_ascii=False)+'\n')
 with (ROOT/'corrected_labels.csv').open('w',encoding='utf-8-sig',newline='') as out:
  fields=['target_hash','source_filename','old_filename_label','old_visual_label','file_label','visual_label','decision','quality_flags','resource_paths'];w=csv.DictWriter(out,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(labels.values())
 samples=json.load(open('outputs/training_label_audit_20261103/sample.json'));escape=lambda v:html.escape(str(v if v is not None else '—'))
 parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>标签修订对照</title><style>body{font:16px system-ui;margin:24px;background:#f5f4ef;color:#23382f}article{padding:24px;margin:16px 0;background:white;border-radius:12px;display:grid;grid-template-columns:128px 1fr;gap:24px}img{width:128px;height:128px;image-rendering:pixelated;background:#808080}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left;vertical-align:top}table{width:100%;table-layout:fixed}p,td{overflow-wrap:anywhere}</style><h1>文件语义与独立视觉标签修订</h1><p>视觉阶段只看图片；名称阶段核对原文件名、已恢复路径和独立观察。组件、图层和不确定对象隔离；旧描述不会回填。自动标签仍需抽查。此页展示原72张核验图的修订结果。</p><a href="summary.json">全量进度</a> · <a href="corrected_labels.csv">已修订标签CSV</a>']
 for r in samples:
  c=labels.get(r['target_hash']);parts.append(f'<article><img src="../training_label_audit_20261103/{r["image"]}"><div><h3>#{r["sample_id"]:03d} {escape(r["source_filename"])}</h3>')
  if not c:parts.append('<p>待处理</p></div></article>');continue
  parts.append(f'<p>结果：{escape(c["decision"])}；原因：{escape(c["quality_flags"])}</p><table><tr><th></th><th>原标签</th><th>修订后训练候选</th></tr><tr><td>文件标签</td><td>{escape(c["old_filename_label"])}</td><td>{escape(c["file_label"])}</td></tr><tr><td>视觉标签</td><td>{escape(c["old_visual_label"])}</td><td>{escape(c["visual_label"])}</td></tr></table><p>资源路径：{escape(c["resource_paths"])}</p><details><summary>查看独立观察和名称核验</summary><p>{escape(c.get("visual_observation"))}</p><p>{escape(c.get("semantic_review"))}</p></details></div></article>')
 parts.append('</html>');(ROOT/'review.html').write_text(''.join(parts));print(json.dumps(summary),flush=True)
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT);args=parser.parse_args();ROOT=args.root;export()
