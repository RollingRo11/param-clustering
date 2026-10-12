# param-clustering

Research repo for parameter-decomposition experiments on small language models: the current parameter autoencoder in `par-auto/`, Parfact toy/67M experiments in `parfact/`, and shared 67M utilities in `geo-attribution/`. Scripts, not frameworks.

## Layout

- `par-auto/`, `parfact/` and `geo-attribution/` hold the scripts. Result JSONs go in `<dir>/out/`, figures in `<dir>/figures/`. Both are gitignored artifacts: never `git add` them.
- Data root for the 67M workstream is the `COFAC_DATA` env var (target checkpoint in `$COFAC_DATA/target`, VPD checkpoint in `$COFAC_DATA/vpd`).

## Compute: Northeastern Explorer cluster

- Access: `ssh explorer` (aliased and connection-multiplexed). The full workflow lives in the `northeastern-cluster` skill; read it before submitting.
- Project root on the cluster: `/projects/RohanKathuria/param-clustering/` with `code/` (sources plus the venv at `code/.venv`), `data/` (`COFAC_DATA` points here), `logs/`, `slurm_*.sh` at the root, and `env.sh` (secrets, sourced by every job).
- Python for jobs: `code/.venv/bin/python -u`. Always `-u`, or logs look empty.
- SLURM: `sbatch slurm_*.sh`, `squeue -u kathuria.r`. `gpu-short` caps at 2 h, `gpu` at 8 h, CPU work on `short`. Login nodes are load-balanced and OOM-kill heavy processes: anything that loads real data runs as a job.
- Never scancel or touch jobs you did not launch.
- Keep local and cluster copies of a script in sync (scp) and verify with a checksum before submitting jobs against it.

## Storage and secrets

- Cloudflare R2 bucket `param-clustering` is the durable store for large artifacts (`code/r2_push.py` and `code/r2_pull.py` on the cluster, run as jobs). Beam.cloud is legacy: do not create Beam resources or delete Beam disks unless asked.
- `env.sh` on the cluster holds `HF_TOKEN`, R2 credentials, `COFAC_DATA`, `HF_HOME`. Never print it or echo those variables.

## Conventions

- Report times in US Central.
- The local shell is fish: brace expansion fails in scp paths, so list files explicitly.
- Commit only when asked.
