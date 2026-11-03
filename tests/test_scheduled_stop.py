import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from mcgen.catalog import atomic_json
from mcgen.smoke import fixture, tiny_config
from mcgen.training import Trainer


def test_requested_stop_saves_optimizer_and_resumes_exactly(tmp_path):
    torch.set_num_threads(1)
    catalog=fixture(tmp_path/'catalog');cfg=tiny_config()
    full=Trainer(cfg,catalog,tmp_path/'full');full.run()
    stopped=Trainer(cfg,catalog,tmp_path/'stopped');original=stopped.optimizer.step
    def step(*args,**kwargs):
        result=original(*args,**kwargs);stopped.stop_requested=True;return result
    stopped.optimizer.step=step;stopped.run()
    assert stopped.step==1
    assert json.load(open(tmp_path/'stopped/stopped.json'))['step']==1
    resumed=Trainer(cfg,catalog,tmp_path/'stopped',resume=tmp_path/'stopped/latest.pt');resumed.run()
    for k,v in full.model.state_dict().items():assert torch.equal(v,resumed.model.state_dict()[k])


def test_absolute_timer_stops_only_recorded_session(tmp_path):
    p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)','queue_fullscale_text.py'],start_new_session=True)
    try:
        fields=Path(f'/proc/{p.pid}/stat').read_text().rsplit(')',1)[1].split()
        atomic_json(tmp_path/'stop_schedule.json',{'deadline_unix':time.time()+.2,'deadline_local':'test',
                    'queue_pgid':p.pid,'queue_start_ticks':fields[19]})
        subprocess.run([sys.executable,'scripts/stop_fullscale_at.py','--root',str(tmp_path)],check=True,timeout=10)
        assert p.wait(timeout=3)==-15
        result=json.load(open(tmp_path/'stop_timer_status.json'))
        assert result['stage']=='stopped' and not result['forced_after_120_seconds']
    finally:
        if p.poll() is None:p.terminate();p.wait()
