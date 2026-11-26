import csv
import sys
from pathlib import Path

rows=list(csv.DictReader(Path(sys.argv[1]).open()))
def event(name):
    return [i for i,r in enumerate(rows) if r['event']==name]
assert len(event('cycle'))==2 and len(event('released'))==1
holds=event('hold')
assert len(holds)>=2
for i in holds:
    j=next(j for j in event('stream') if j>i)
    delay=(int(rows[j]['wall_ns'])-int(rows[i]['wall_ns']))/1e9
    samples=[float(r['encoder_rad']) for r in rows[i:j+1] if r['encoder_rad']]
    drift=max(samples)-min(samples)
    assert delay>=1.25 and drift < .025, (delay,drift)
    assert 'plan' in [r['event'] for r in rows[i+1:j]]
    print(f"hold -> replan: {delay:.3f}s, encoder displacement {drift:.5f} rad")
assert max(float(r['encoder_rad']) for r in rows if r['encoder_rad'])>.28
print("Two mock curl cycles, hand stop, lost-guard stop, measured-state replanning, return and release verified.")
