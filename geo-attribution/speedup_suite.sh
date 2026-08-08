#!/bin/bash
# Benchmark + fidelity suite for the fast collection path.
set -x
cd "$(dirname "$0")"

# Reduced-D specs (instant, CPU-side).
python3.12 collect_fast.py spec_subset --spec_tag stream_spec \
  --tag stream_spec_d8192 --feat_dim 8192
python3.12 collect_fast.py spec_subset --spec_tag stream_spec \
  --tag stream_spec_d4096 --feat_dim 4096

# Throughput sweep at production pos_per_seq=506 (both GPUs).
for bs in 4 8 16 32; do
  torchrun --nproc_per_node=2 collect_fast.py benchmark --profile optimized \
    --spec_tag stream_spec --synthetic_data --pos_per_seq 506 \
    --batch_seqs $bs --benchmark_batches 8 \
    --output out/bench_fast_bs$bs.json
done

# Compile-mode and master-dtype variants at the two largest batches.
for bs in 16 32; do
  torchrun --nproc_per_node=2 collect_fast.py benchmark --profile optimized \
    --spec_tag stream_spec --synthetic_data --pos_per_seq 506 \
    --batch_seqs $bs --benchmark_batches 8 --compile_mode max-autotune \
    --output out/bench_fast_bs${bs}_autotune.json
  torchrun --nproc_per_node=2 collect_fast.py benchmark --profile optimized \
    --spec_tag stream_spec --synthetic_data --pos_per_seq 506 \
    --batch_seqs $bs --benchmark_batches 8 --master_dtype bfloat16 \
    --output out/bench_fast_bs${bs}_bf16master.json
done

# Reduced-D throughput at the largest batch.
torchrun --nproc_per_node=2 collect_fast.py benchmark --profile optimized \
  --spec_tag stream_spec_d8192 --synthetic_data --pos_per_seq 506 \
  --batch_seqs 32 --benchmark_batches 8 \
  --output out/bench_fast_bs32_d8192.json

# Fidelity gates: exact gram on reservoir rows vs candidate specs.
for tag in stream_spec stream_spec_d8192 stream_spec_d4096; do
  CUDA_VISIBLE_DEVICES=0 python3.12 collect_fast.py validate \
    --spec_tag $tag --val_tag run1m_stream --val_n 512 --val_gate 0.0
done

# fp32-vs-bf16 master weights A/B on real Pile tokens.
CUDA_VISIBLE_DEVICES=0 python3.12 collect_fast.py ab_master \
  --spec_tag stream_spec --tag ab_master_check --pos_per_seq 506 \
  --batch_seqs 4 --val_gate 0.0

echo SUITE_DONE
