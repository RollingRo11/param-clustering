"""Check copied VPD streams against the sha256 values in manifest.json (written where the streams were built).

Usage: python verify_vpd_stream.py DIR
"""
import hashlib
import json
import sys
from pathlib import Path

d = Path(sys.argv[1])
m = json.loads((d/'manifest.json').read_text())
assert m['complete'] and len(m['ranks']) == m['world_size'] == 8
ok = True
for r in m['ranks']:
    p = d/f"rank{r['rank']}.uint16"
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(64 << 20), b''): h.update(block)
    good = h.hexdigest() == r['sha256'] and p.stat().st_size == r['rows']*r['row_tokens']*2
    ok &= good
    print(r['rank'], 'MATCH' if good else 'MISMATCH', h.hexdigest()[:16], flush=True)
print('ALL_STREAMS_VERIFIED' if ok else 'STREAM_VERIFY_FAILED', flush=True)
sys.exit(0 if ok else 1)
