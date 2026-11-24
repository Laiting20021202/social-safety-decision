#!/usr/bin/env python3
"""Isolated Gazebo Classic contact evaluator; no policy outputs.

Counts physics contact reports, not independent collisions. Only the fixed
pedestal/ground contact is excluded; moving arm contacts are all retained.
An empty or interrupted transport stream cannot pass as collision-free.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import signal
import subprocess
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seconds',type=float,default=230)
    p.add_argument('--world',default='openarm_sorting')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    process=subprocess.Popen(['gz','topic','-e',f'/gazebo/{a.world}/physics/contacts','-d',str(int(a.seconds))],stdout=subprocess.PIPE,text=True,bufsize=1)
    stopped = []
    def finish(*_):
        stopped.append(True)
        process.terminate()
    signal.signal(signal.SIGINT, finish)
    signal.signal(signal.SIGTERM, finish)
    pairs=Counter(); depths={}; first={}; all_contacts=0; pair=None; collision1=None
    start=time.monotonic(); start_epoch=time.time()
    for line in process.stdout:
        if 'collision1:' in line:
            collision1=re.search(r'"([^"]+)"',line).group(1)
        elif 'collision2:' in line:
            all_contacts+=1
            collision2=re.search(r'"([^"]+)"',line).group(1)
            values=sorted([collision1,collision2]); pair=None
            moving=any(v.startswith('openarm::') and 'openarm_body_link0::' not in v for v in values)
            if moving:
                pair=' | '.join(values); pairs[pair]+=1
                first.setdefault(pair,time.monotonic()-start)
        elif pair is not None and line.strip().startswith('depth:'):
            depth=float(line.split(':',1)[1]); depths[pair]=max(depth,depths.get(pair,0.))
    process.wait()
    duration = time.monotonic()-start
    valid = all_contacts > 0 and ((process.returncode == 0 and duration >= .98*a.seconds) or (bool(stopped) and duration > 5))
    result=dict(backend='Gazebo Classic physics contact stream; evaluator only',duration_sec=duration,start_epoch_sec=start_epoch,end_epoch_sec=time.time(),requested_duration_sec=a.seconds,stopped_by_trial_runner=bool(stopped),transport_exit=process.returncode,all_contact_reports=all_contacts,moving_arm_contact_reports=dict(pairs),maximum_penetration_m=depths,first_contact_wall_sec=first,passed=bool(valid and not pairs),limitation='Discrete physics contacts; does not measure positive clearance or prove continuous-time collision freedom. Geometry quality is the SDF collision geometry, not necessarily visual meshes.')
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
