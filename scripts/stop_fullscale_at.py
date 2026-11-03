"""Detached absolute-time stop for one recorded queue session; never touches other jobs."""
import argparse
import json
import os
from pathlib import Path
import signal
import time

from mcgen.catalog import atomic_json


def members(pgid):
    result=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            fields=(path/'stat').read_text().rsplit(')',1)[1].split()
            if int(fields[2])==pgid and int(fields[3])==pgid and fields[0]!='Z':
                result.append({'pid':int(path.name),'start_ticks':fields[19],
                               'command':(path/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')})
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    return result


def main(root):
    schedule=json.load(open(root/'stop_schedule.json'))
    deadline=schedule['deadline_unix'];pgid=schedule['queue_pgid']
    atomic_json(root/'stop_timer_status.json',{'stage':'armed','deadline':schedule['deadline_local'],'pid':os.getpid()})
    while time.time()<deadline:time.sleep(min(30,max(.01,deadline-time.time())))
    current=members(pgid)
    parent=next((p for p in current if p['pid']==pgid),None)
    if parent is not None and parent['start_ticks']!=schedule['queue_start_ticks']:
        raise RuntimeError('Queue PID reused; refusing to signal unrelated process group')
    if current and not any('queue_fullscale_text.py' in p['command'] or str(root) in p['command'] for p in current):
        raise RuntimeError('Queue session no longer matches expected commands')
    atomic_json(root/'STOP_REQUESTED.json',{'requested_unix':time.time(),'deadline':schedule['deadline_local'],'reason':'user_scheduled_stop'})
    if current:
        try:os.killpg(pgid,signal.SIGTERM)
        except ProcessLookupError:pass
    atomic_json(root/'stop_timer_status.json',{'stage':'stop_signalled','deadline':schedule['deadline_local'],'signalled_members':current})
    # Allow the current update and atomic checkpoint write to finish.
    end=time.time()+120
    while members(pgid) and time.time()<end:time.sleep(1)
    remaining=members(pgid)
    if remaining:
        try:os.killpg(pgid,signal.SIGKILL)
        except ProcessLookupError:pass
    result={'stage':'stopped','finished_unix':time.time(),'deadline':schedule['deadline_local'],
            'forced_after_120_seconds':bool(remaining),'checkpoint_exists':(root/'training/latest.pt').exists()}
    if (root/'training/stopped.json').exists():result['training']=json.load(open(root/'training/stopped.json'))
    atomic_json(root/'stop_timer_status.json',result)
    atomic_json(root/'queue_status.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
