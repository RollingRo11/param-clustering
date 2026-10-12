"""Archive GIM experiment checkpoints, results, and exact code after GPU release."""
import json
import tarfile
from common import ROOT,HERE,save_json,upload,sha,credentials


def main():
    credentials(); out=ROOT/'output/gim_1h'; out.mkdir(parents=True,exist_ok=True)
    sources=list(HERE.glob('*.py'))+list(HERE.glob('*.yaml'))+list(HERE.glob('slurm_*gim*.sh'))
    sources.append(HERE/'gim_1h.md')
    sources+=list((HERE.parent/'par-auto').glob('*.py'))
    save_json({str(p.relative_to(HERE.parent)):sha(p) for p in sources},out/'source_manifest.json')
    with tarfile.open(out/'source.tar.gz','w:gz') as archive:
        for path in sources: archive.add(path,arcname=str(path.relative_to(HERE.parent)),recursive=False)
    receipts=[]
    for path in sorted(out.rglob('*')):
        if path.is_file() and 'wandb' not in path.parts and path.suffix in ('.pt','.pth','.json','.png','.gz'):
            receipts.append(upload(path,'experiments/gim_1h/'+str(path.relative_to(out))))
    for path in sorted((ROOT.parents[2]/'logs').glob('parauto_gim_*')):
        if path.is_file(): receipts.append(upload(path,'experiments/gim_1h/logs/'+path.name))
    save_json(receipts,out/'r2_receipts.json'); upload(out/'r2_receipts.json','experiments/gim_1h/r2_receipts.json')
    print('GIM_R2_BACKUP_VERIFIED',flush=True)


if __name__=='__main__': main()
