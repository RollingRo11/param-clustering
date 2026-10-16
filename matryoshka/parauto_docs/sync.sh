#!/bin/bash
# A dedicated current directory leaves already submitted jobs' sources intact.
set -euo pipefail
directory="$(cd "$(dirname "$0")" && pwd)"
destination="explorer:/projects/RohanKathuria/param-clustering"
rsync -az --delete --exclude out --exclude __pycache__ --exclude .pytest_cache --exclude .ruff_cache --exclude '*.pt' \
  "$directory/" "$destination/code/par-auto-current/"
for name in train benchmark profile validate spd_assets klkeep short_schedules spd_short precision precision_sweep pile pile_smoke pile_last pile_vpd_budget scalar_sensor wandb_smoke; do
  scp -q "$directory/slurm_$name.sh" "$destination/slurm_parauto_$name.sh"
done
manifest="$(mktemp)"
trap 'rm -f "$manifest"' EXIT
(cd "$directory" && rg --files -g '*.py' -g '*.md' -g '*.sh' -g '!out/**' \
  | sort | while IFS= read -r file; do shasum -a 256 "$file"; done) > "$manifest"
scp -q "$manifest" "$destination/code/par-auto-current/SHA256SUMS"
ssh -o BatchMode=yes explorer 'cd /projects/RohanKathuria/param-clustering/code/par-auto-current && sha256sum -c SHA256SUMS'
