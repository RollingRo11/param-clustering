#!/bin/bash
set -e
cd /Users/rohan/Developer/param-clustering/geo-attribution
U="uv run --with beam-client --python 3.12 python"
F=/private/tmp/claude-501/-Users-rohan-Developer-param-clustering/30021c40-d2f5-4c1b-89c6-097ef77996b9/tasks/bmg0smawq.output
echo skip-fetch-gate
echo FETCH_OK
echo "=== sanity ==="; $U beam_cofac67.py sanity
echo "=== atoms ===";  $U beam_cofac67.py atoms
echo "=== verify ==="
$U beam_cofac67.py verify | tee /tmp/cofac67_verify.json
grep -q '"worst_relerr"' /tmp/cofac67_verify.json
python3 -c "
import json,sys
t=open('/tmp/cofac67_verify.json').read()
d=json.loads(t[t.index('{'):])
assert d['worst_relerr'] < 1e-3, f'verify failed: {d}'
print('VERIFY_OK', d)"
echo "=== collect 32 chunks (16384 events) ==="
$U beam_cofac67.py collect 32
echo "=== fit ==="; $U beam_cofac67.py fit
echo "=== eval ==="; $U beam_cofac67.py eval
echo CHAIN67_DONE
