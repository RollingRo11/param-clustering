#!/bin/bash
set -e
cd /Users/rohan/Developer/param-clustering/geo-attribution
U="uv run --with beam-client --python 3.12 python"
echo "=== sync all ==="
$U beam_r2_sync.py sync
echo "=== verify all ==="
$U beam_r2_sync.py verify
echo R2_MIGRATION_DONE
