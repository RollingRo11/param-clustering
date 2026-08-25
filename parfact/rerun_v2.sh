#!/bin/bash
set -e
cd /Users/rohan/Developer/param-clustering/parfact
U="uv run --with beam-client --python 3.12 python"
echo "=== v2 fits (rescaled init) ==="
$U beam_v2.py fit
echo "=== curves ==="
$U beam_v2.py curves
echo "=== fetch ==="
$U beam_v2.py fetch
echo RERUN_DONE
