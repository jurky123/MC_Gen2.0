"""Run one caption-only training arm with a bounded share of GPU memory."""
import argparse
import time
import torch
from mcgen.config import Experiment
from mcgen.training import Trainer

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--catalog',required=True)
    p.add_argument('--out',required=True);p.add_argument('--initialize',required=True);p.add_argument('--device',default='cuda:1')
    a=p.parse_args();torch.set_num_threads(4)
    while torch.cuda.mem_get_info(a.device)[0]<23*1024**3:time.sleep(30)
    free,total=torch.cuda.mem_get_info(a.device)
    torch.cuda.set_per_process_memory_fraction(min(.30,(free-3*1024**3)/total),a.device)
    cfg=Experiment.load(a.config)
    Trainer(cfg,a.catalog,a.out,a.device,a.device,initialize=a.initialize).run()
