"""Add target-balanced short/long supervision to an existing reviewed catalog."""
import argparse
from pathlib import Path
from mcgen.catalog import read_catalog, write_catalog, atomic_json
from mcgen.semantic_captions import filename_words


def build(source,output):
    records,meta=read_catalog(source)
    rows=[]
    for r in records:
        if r['split']!='train':
            rows.append(r);continue
        short=' '.join(filename_words(r['source_filename']))
        if not short.strip() or not r['prompt'].strip():raise ValueError('Missing caption variant')
        # Dataset/TaskSampler sample targets first, then variants; two captions
        # do not double this image's weight relative to other targets.
        for kind,prompt in [('long',r['prompt']),('short',short)]:
            rows.append({**r,'id':r['id']+':'+kind,'prompt':prompt,'caption_variant':kind,'caption_experiment_arm':'mixed'})
    write_catalog(output,rows,{'native_size':meta['native_size'],'parent_fingerprint':meta['fingerprint'],
        'policy':'Same images/splits as parent. Uniform target sampling, then uniform distinct short/long captions. Validation unchanged. Automatic weak labels; no production promotion.'})
    atomic_json(output/'variant_protocol.json',{'source':str(source),'parent_fingerprint':meta['fingerprint'],
        'train_targets':len({r['target_hash'] for r in rows if r['split']=='train'}),'train_rows':sum(r['split']=='train' for r in rows),
        'validation_unchanged':True,'desired_long_probability':.5,'identical_variants':'Deduplicated by existing text dataset; equivalent text retains target weight.'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();build(a.source,a.out)
