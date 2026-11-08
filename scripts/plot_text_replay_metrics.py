"""Standalone scientific figure; requires matplotlib, not torch or the encoder."""
import argparse
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(root,out):
    history=[json.loads(line) for line in (root/'training/validation.jsonl').read_text().splitlines()]
    train=[json.loads(line) for line in (root/'training/metrics.jsonl').read_text().splitlines()]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(15,4.4),layout='constrained')
    x=[r['step'] for r in history]
    axes[0].plot(x,[r['broad_text']['generation_mae'] for r in history],'o-',label='Broad T2I validation')
    axes[0].plot(x,[r['paired_text_generation_mae'] for r in history],'o-',label='Paired T2I validation')
    axes[0].plot(x,[r['reference_generation_mae'] for r in history],'o-',label='Reference validation')
    axes[0].set(title='Fixed validation generation',xlabel='New optimizer updates',ylabel='Alignment MAE (lower is better)')
    axes[0].legend(fontsize=8)
    axes[1].plot(x,[r['caption_response_l1']['joint'] for r in history],'o-',label='Joint: changed caption')
    axes[1].plot(x,[r['caption_response_l1']['text'] for r in history],'o-',label='T2I: changed caption')
    axes[1].set(title='Same reference / different captions',xlabel='New optimizer updates',ylabel='Raw output response L1')
    axes[1].legend(fontsize=8)
    chunks=[train[i:i+100] for i in range(0,len(train),100)]
    axes[2].plot([c[-1]['step'] for c in chunks],[statistics.median(r['gradient_norm'] for r in c) for c in chunks],label='100-update median')
    axes[2].axhline(1,color='#dc2626',linestyle='--',label='Gradient clipping threshold')
    axes[2].set(title='Finite gradients still fluctuate',xlabel='New optimizer updates',ylabel='Pre-clip gradient norm',yscale='log')
    axes[2].legend(fontsize=8)
    for a in axes:a.grid(alpha=.2)
    out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=150)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    plot(a.root,a.out)
