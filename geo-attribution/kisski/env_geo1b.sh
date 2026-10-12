# Sourced by the KISSKI geo1b jobs. Compute nodes have no internet.
G=/mnt/vast-kisski/projects/hammerhai-parallax/geo1b
export HF_HOME=$G/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
export GEO_ATTRIBUTION_ARTIFACT_ROOT=$G/data
export GEO_ATTRIBUTION_DATA_PATH=$G/data/pile_llama_u32.bin
export TORCHINDUCTOR_CACHE_DIR=$G/.torchinductor_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="$G/code/.venv/bin/python -u"
TR="$G/code/.venv/bin/torchrun --standalone --nproc_per_node=4"
cd $G/code
