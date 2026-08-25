# param-clustering

Research repo for **"Attribution-Based Co-Factorization for Parameter Decomposition"** (ICLR 2027 draft). We decompose model weights into components via attribution-based co-factorization (M̄ ≈ USVᵀ over SVD atoms) and compare against SPD/VPD baselines. This is a solo research codebase: scripts over frameworks, results as JSONs, figures for slides/paper.

## Layout

- `parfact/` — toy-model workstream: 2-layer induction model, C=600 decompositions, co-fac v1/v2/U-simplex vs SPD (their code). Plot scripts (`plot_*.py`) read `parfact/out/` JSONs → `parfact/figures/`.
- `geo-attribution/` — main workstream: 67M-param VPD-target Pile model, C=4096 co-factorization ("cofac67"). Core pipeline is **`geo-attribution/cofac67.py`** (single module: prestage → collect attributions → fit → KLKeep evals → analysis). Plot scripts read `geo-attribution/out/` → `geo-attribution/figures/`.
- Result JSONs live in `<dir>/out/`, figures in `<dir>/figures/`. Plots use matplotlib with `Agg` backend.

### cofac67 pipeline essentials

- Data root comes from `COFAC_DATA` env var; `RUN = $COFAC_DATA/cofac67` (small runs), `BIG = $COFAC_DATA/cofac67_big` (1M-event run). Targets in `$COFAC_DATA/target`, VPD baseline checkpoint in `$COFAC_DATA/vpd`.
- Method versions: v1/v2 (softmax V, mega-component collapse — one component takes ~83% of weight mass), v3 `fit_centered` (**failed** — residual drains to zero, don't revisit), v3b `fit_pinned` (**works** — backbone fraction f pinned via rank-1 least squares; keep `f_cap ≥ 0.9`).
- The same `cofac67.py` runs locally and on the cluster (`code/cofac67.py` there). **Keep the two copies in sync via scp** after any edit; verify with grep/checksum before submitting jobs against it.

## Compute: Northeastern Explorer cluster (primary)

All heavy compute runs here. Full how-to lives in the `northeastern-cluster` skill — invoke it before cluster work. Core facts:

- Access: `ssh -o BatchMode=yes kathuria.r@login.explorer.northeastern.edu '<cmd>'` (keys set up).
- Everything lives in `/projects/RohanKathuria/param-clustering/`: `code/` (sources + venv), `data/` (COFAC_DATA points here), `logs/`, `slurm_*.sh` job scripts at the root, `env.sh` (secrets, sourced by every job).
- Python for jobs: `code/.venv/bin/python` (torch 2.7.1+cu126), **always with `-u`** (else logs block-buffer and look empty).
- SLURM: submit `sbatch slurm_*.sh`; monitor `squeue -u kathuria.r`. GPU default: `-p gpu-short,gpu --gres=gpu:h200:1` (H200s schedule in minutes; A100 queue stalls). `gpu-short` caps at 2 h; `gpu` at 8 h; CPU jobs on `short`. Chain with `--dependency=afterok:<id>`.
- Login nodes are load-balanced (per-node `/tmp` — shared state goes in `/projects`) and OOM-kill heavy processes — anything loading real data runs as a SLURM job, even quick CPU work (`sbatch -p short --wrap=...`).
- **Never scancel or touch jobs you didn't launch** — the user runs their own jobs on the same account (a job pending on `QOSMaxGRESPerJob` is quota-waiting, not broken).

Beam.cloud is **legacy** — earlier runs lived there and some `beam_*.py` scripts reference its durable disks. Do not spin up Beam resources or delete Beam disks without being asked; everything needed has been mirrored to R2. If Beam is ever used: machine release ≠ billing stopped — pool delete + dashboard check.

## R2 storage (source of truth for artifacts)

Cloudflare R2, bucket **`param-clustering`**, accessed via boto3 S3 API with env vars `R2_ENDPOINT`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY` (defined in cluster `env.sh` and in Beam secrets; not set locally).

- Top-level prefixes: `cofac67/` (67M workstream: `big/`, `cofac67/`, `piledata/`, `target/`, `vpd/`), `spd-c600*/` (toy SPD runs incl. `spd-c600-cc/runs/full1m` = 1M-step SPD), `cofact-c600/`, `parfact-disk/` (Beam disk mirror).
- Transfer scripts on the cluster: `code/r2_push.py <local_dir> <prefix>` and `code/r2_pull.py <prefix> <local_dir>` — both idempotent (skip if size matches), safe to re-run. Run them as SLURM jobs, not on login nodes.
- R2 is the durable backup: after producing new large artifacts on the cluster, push them (`slurm_push.sh` pattern).

## Secrets

`env.sh` on the cluster (chmod 600) holds `HF_TOKEN`, R2 credentials, `COFAC_DATA`, `HF_HOME`. **Never print env.sh contents or echo those variables.** Small result JSONs come back via scp; secrets never leave the cluster.

## Conventions

- Report times to the user in **US Central**, not UTC.
- Local shell is fish: brace expansion like `{a,b}.py` fails in scp paths — list files explicitly.
- Long remote jobs: poll at a cadence matching job length; `squeue` is ground truth, not log tails.
- When a figure request is for slides, favor readability over paper rigor, but flag anything that changes the story (e.g., a baseline swap that flips who wins) before the user presents it.
