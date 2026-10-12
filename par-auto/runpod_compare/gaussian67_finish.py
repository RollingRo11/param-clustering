"""Archive the completed bounded trial and verify durable upload receipts."""
import json
import tarfile
from pathlib import Path
from common import ROOT, HERE, save_json, upload, sha, credentials


def main():
    credentials(); out=ROOT/'output/gaussian67_20260914'
    for arm in ['gaussian','boundary001']:
        assert json.loads((out/arm/'latest_checkpoint.json').read_text())['step']==5000
        assert (out/arm/'eval_fisher_k8/klkeep.json').exists()
    assert (out/'comparison.json').exists()
    with tarfile.open(out/'source.tar.gz','w:gz') as archive:
        for p in sorted(HERE.parent.rglob('*')):
            if p.is_file() and p.suffix in ['.py','.yaml','.json','.sha256'] and '__pycache__' not in p.parts:
                archive.add(p,arcname=str(p.relative_to(HERE.parent)),recursive=False)
    receipts=[]
    for p in sorted(out.rglob('*')):
        if not p.is_file() or 'wandb' in p.parts:continue
        # Keep all model snapshots, plus only the final optimizer states for resume.
        if p.name=='checkpoint.pt' and p.parent.name!='005000':continue
        if p.suffix in ['.json','.md','.pth','.pt','.gz']:
            receipts.append(upload(p,'experiments/gaussian67_20260914/'+str(p.relative_to(out))))
    save_json(receipts,out/'r2_receipts.json')
    upload(out/'r2_receipts.json','experiments/gaussian67_20260914/r2_receipts.json')
    save_json(dict(complete=True,arms=['gaussian','boundary001'],steps=5000,
                   r2_verified_files=len(receipts),comparison_sha256=sha(out/'comparison.json')),out/'complete.json')
    upload(out/'complete.json','experiments/gaussian67_20260914/complete.json')
    print('GAUSSIAN67_COMPLETE_AND_ARCHIVED',flush=True)


if __name__=='__main__':main()
