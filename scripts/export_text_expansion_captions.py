"""Export paired filename/description supervision without pretending labels are gold."""
import base64,html,io,json
from pathlib import Path
from PIL import Image
from audit_item_captions import array
from mcgen.images import composite
ROOT=Path('outputs/text_expansion2k_20261102')
if __name__=='__main__':
 rows=json.loads((ROOT/'cases.json').read_text());selected=set(json.loads((ROOT/'fixed_review64.json').read_text())['indices'])
 elements=['<!doctype html><meta charset="utf-8"><title>Expanded caption supervision</title><style>body{font:16px system-ui;margin:30px;max-width:1300px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:12px;text-align:left}img{width:96px;height:96px;image-rendering:pixelated}small{color:#666}</style><h1>文件名与详细描述监督对照</h1><p>固定 64 张候选抽样；本页展示经过图像/格式筛选后保留的样本。自动弱标注，不等于人工认证。两种文本合格时各 50%；详细描述异常时仅使用文件名，图片抽样权重保持相同。</p><table><tr><th>图像</th><th>文件名文本</th><th>详细描述</th><th>来源和标记</th></tr>']
 exported=[]
 for r in rows:
  if r.get('caption_origin')!='expansion_automatic_weak_label' or r['review_index'] not in selected:continue
  buf=io.BytesIO();Image.fromarray(composite(array(r),128).round().astype('uint8')).save(buf,format='PNG');src=base64.b64encode(buf.getvalue()).decode()
  elements.append(f'<tr><td><img src="data:image/png;base64,{src}"></td><td>{html.escape(r["short"])}</td><td>{html.escape(r["long"] or "[已弃用详细描述，仅使用文件名标签]")}</td><td>{html.escape(r["split"])} / {r["review_index"]}<br><small>{html.escape(r["source_filename"])}<br>{html.escape(str(r.get("caption_quality_flags",[])))}</small></td></tr>');exported.append({k:r.get(k) for k in ['id','review_index','split','source_filename','short','long','caption_quality_flags','long_caption_eligible','long_caption_rejection','raw_descriptive_text']})
 elements.append('</table>');(ROOT/'caption_comparison.html').write_text('\n'.join(elements));(ROOT/'caption_comparison.json').write_text(json.dumps(exported,ensure_ascii=False,indent=2));print('exported',len(exported))
