"""Archive only source/configuration trees, excluding credentials and outputs."""
import json
from pathlib import Path
import subprocess
import tarfile
from common import ROOT, HERE, VPD_COMMIT, credentials, save_json, sha, upload


def main():
    credentials()
    paths = []
    for root in (HERE, HERE.parent/'par-auto', HERE.parent/'vpd/param_decomp'):
        for path in root.rglob('*'):
            if not path.is_file() or path.is_symlink(): continue
            if any(p in ('out', 'wandb', '__pycache__', '.pytest_cache', 'runpod_compare') for p in path.relative_to(root).parts): continue
            if path.suffix in ('.py', '.md', '.yaml', '.toml', '.lock'): paths.append(path)
    paths.extend(HERE.parent/'vpd'/name for name in ('pyproject.toml', 'uv.lock'))
    info = dict(vpd_commit=VPD_COMMIT, files={str(p.relative_to(HERE.parent)): sha(p) for p in sorted(paths)})
    import torch
    info.update(torch=torch.__version__, cuda=torch.version.cuda)
    save_json(info, ROOT/'output/source_manifest.json')
    archive = ROOT/'output/source.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for path in sorted(paths): tar.add(path, arcname=str(path.relative_to(HERE.parent)), recursive=False)
    save_json(upload(archive, 'source.tar.gz'), ROOT/'output/source_r2.json')
    upload(ROOT/'output/source_manifest.json', 'output/source_manifest.json')


if __name__ == '__main__': main()
