"""Audited JSON-comparison fix applied to the preserved training source snapshot.

The original plan and source manifest remain unchanged. The only behavioral
change is comparing JSON-normalized plans; optimizer and loss code are untouched.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

EXPECTED='011ef260a5aa25853fe1d746d3e3e9e740a402e50c75d7447d763fb5ccab6003'
OLD="assert json.loads((OUT/'plan.json').read_text())==plan,'Immutable plan mismatch'"
NEW="assert json.loads((OUT/'plan.json').read_text())==json.loads(json.dumps(plan)),'Immutable plan mismatch'"


def patched_source(path):
    raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==EXPECTED,'Unexpected original runner'
    source=raw.decode();assert source.count(OLD)==1
    result=source.replace(OLD,NEW)
    marker="    start=resume['next_step'] if resume else 0"
    assert result.count(marker)==1
    result=result.replace(marker,marker+"\n    print(f'GIM_COUNT67_RESUME_START {start}',flush=True)")
    return result


def main():
    assert os.environ.get('SLURM_JOB_ID'),'Use an Explorer SLURM allocation'
    root=Path(os.environ['PAIR_ROOT']);compare=root/'code/gim_count67_20260914/compare'
    original=compare/'gim_count67_experiment.py';source=patched_source(original)
    sys.path.insert(0,str(compare))
    from common import save_json,sha
    out=root/'output/gim_count67_20260914'
    record=dict(reason='JSON serializes tuple-valued configuration entries as lists; compare canonical JSON forms',
                original_source_sha256=EXPECTED,executed_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                wrapper_sha256=sha(__file__),old=OLD,new=NEW,job_id=os.environ['SLURM_JOB_ID'],wrapper_source=Path(__file__).read_text())
    save_json(record,out/f'resume_execution_{os.environ["SLURM_JOB_ID"]}.json')
    print('GIM_COUNT67_JSON_RESUME_FIX_APPLIED',flush=True)
    exec(compile(source,str(original),'exec'),dict(__name__='__main__',__file__=str(original)))


if __name__=='__main__':main()
