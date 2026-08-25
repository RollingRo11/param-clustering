#!/bin/bash
# Independent chain for the second L40S: concat census -> v2 fits -> curves.
# (cc-census waits separately on the their-code run's disk/machine.)
set -e
cd /Users/rohan/Developer/param-clustering/parfact
U="uv run --with beam-client --python 3.12 python"
echo "=== concat census + loss eval ==="
$U concat_census.py
echo "=== v2 fits ==="
$U beam_v2.py fit
echo "=== curves ==="
$U beam_v2.py curves
echo "=== fetch ==="
$U beam_v2.py fetch
echo CHAIN_DONE
