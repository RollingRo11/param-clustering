"""Copy VPD's pieces (V read [d_in, C], U write [C, d_out]) for chosen matrices out of the 2.9 GB checkpoint.
Usage: extract_vpd_pieces67.py --checkpoint vpd.pth --modules h.1.mlp.c_fc ... --output OUT.pt"""
import argparse
import torch

ap = argparse.ArgumentParser(); ap.add_argument('--checkpoint'); ap.add_argument('--modules', nargs='+'); ap.add_argument('--output')
a = ap.parse_args()
saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False, mmap=True)
sd = saved['factors']
out = dict(target_sha256=saved['target_sha256'], completed=saved.get('completed'), pieces={})
for m in a.modules:
    keys = [k for k in sd if m.replace('.', '-') in k or m in k]
    print(m, [(k, tuple(sd[k].shape)) for k in keys], flush=True)
    V = next(sd[k] for k in keys if k.endswith('.V')); U = next(sd[k] for k in keys if k.endswith('.U'))
    out['pieces'][m] = dict(V=V.float().clone(), U=U.float().clone())
torch.save(out, a.output); print('EXTRACT_DONE', flush=True)
