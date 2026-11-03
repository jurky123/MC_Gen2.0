"""Prepare the larger scratch model; final batch and exposure budget are measured later."""
import json,time
from pathlib import Path
import yaml
from mcgen.catalog import atomic_json,file_hash
from mcgen.config import Experiment
ROOT=Path('outputs/qwen_vl_embedding_large_20261103')
DATA=Path('outputs/resource_text_training_20261103')
if __name__=='__main__':
 ROOT.mkdir(exist_ok=False)
 import requests
 reference=requests.get('https://raw.githubusercontent.com/QwenLM/Qwen3-VL-Embedding/main/src/models/qwen3_vl_embedding.py',timeout=30);reference.raise_for_status();(ROOT/'official_embedding_reference.py').write_text(reference.text)
 model=Path('/home/iflab/models/Qwen3-VL-Embedding-8B');revision=json.load(open(model/'download_manifest.json'))['revision']
 cfg=yaml.safe_load((DATA/'config.yaml').read_text());cfg['model'].update(width=512,depth=12,text_dim=4096,text_length=1,conditioning='text_only',ref_size=16)
 cfg['text'].update(backend='qwen3_vl_embedding',model=str(model),revision=revision,layers=[-1],instruction='',cache_mb=0)
 cfg['train'].update(lr=2e-4,steps=10000,batch_size=64,accumulation=1,warmup=1000,eval_every=500,seed=2026110310,schedule_start=0)
 (ROOT/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False));Experiment.load(ROOT/'config.yaml')
 (ROOT/'catalog').symlink_to((DATA/'catalog').resolve(),target_is_directory=True)
 for name in ['eval_cases.json','sample.json']:(ROOT/name).write_bytes((DATA/name).read_bytes())
 atomic_json(ROOT/'plan.json',{'status':'batch_and_steps_pending_measurement','generator':'512width12layers text_only scratch','encoder':'Qwen/Qwen3-VL-Embedding-8B official final-token normalized4096D embedding','revision':revision,'train_targets':311510,'conditional_probability':.9,'target_conditional_exposures':14,'optimizer':'fresh AdamW,LR2e-4','budget_formula':'ceil(target_exposure *311510 / (batch_size *accumulation *.9))','warmup_formula':'ceil(.5 *311510 / effective_batch)','eval_interval_formula':'ceil(311510 / (.9 *effective_batch))','batch_policy':'real image/embedding forward+backward+AdamW, memory peak and throughput; choose stable throughput-optimal batch with free-memory margin','data_manifest_sha256':file_hash(DATA/'catalog/samples.jsonl'),'visual_labels':False,'pretrained_generator':False,'official_protocol_source':'https://github.com/QwenLM/Qwen3-VL-Embedding/blob/main/src/models/qwen3_vl_embedding.py'})
 old=Path('outputs/resource_text_scratch_20261103');stopped=json.load(open(old/'training/stopped.json'));atomic_json(old/'queue_status.json',{'stage':'stopped','reason':'user_requested_larger_model_and_correct_embedding','saved_step':stopped['step'],'updated_unix':time.time()})
 print('Prepared',ROOT)
